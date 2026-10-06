"""
test_phase1.py
OceanVerse AI v4.0 — Phase 1 Tests
Target: 21+ new test classes, 220+ total tests when combined with existing 199.

Tests cover:
  - Authentication (login, session, logout, register, roles)
  - Settings (get, save, defaults)
  - Favorites (add, list, delete)
  - Map Bookmarks (add, list, rename, delete)
  - Activity Log (write, query, filter)
  - Dataset Registry (register, list, stats, delete)
  - Model Registry (register, list, activate, rollback, compare)
  - API Key Vault (set, get, test, toggle)
  - Scheduler (list, run, update interval)
  - Notifications (push, list, mark read)
  - DB Health (table counts, health score)
  - Report Catalog (register, list)
  - Platform Stats (aggregate counters)
  - Flask Page Routes (all new pages return 200)
  - Flask Auth API endpoints
  - Flask Settings API endpoints
  - Flask Favorites API endpoints
  - Flask Bookmarks API endpoints
  - Flask Activity API endpoint
  - Flask Registry API endpoints
  - Flask DB Health API endpoint
"""

import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

# ── make sure project root is on path ────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import ocean_platform as plat  # renamed from platform.py to avoid stdlib collision

# ── shared in-memory DB fixture ──────────────────────────────
@pytest.fixture
def db():
    """Fresh in-memory SQLite with Phase 1 schema."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    plat.init_phase1_tables(conn)
    yield conn
    conn.close()


@pytest.fixture
def flask_client():
    """Flask test client with all phases mocked out."""
    os.environ.setdefault("FLASK_TESTING", "1")
    # Patch heavy pipeline imports before loading app
    import unittest.mock as mock
    mods = [
        "pipeline", "ai_engine", "evaluation", "visualization",
        "preprocessing", "admin",
    ]
    patchers = []
    for m in mods:
        p = mock.patch.dict("sys.modules", {m: mock.MagicMock()})
        p.start()
        patchers.append(p)

    # Provide minimal _state
    import types
    fake_pipeline = sys.modules["pipeline"]
    fake_pipeline._state = {
        "scan_records": [], "inspections": {}, "qc_results": [],
        "cov_results": [], "summary": {}, "p2_status": {},
        "p2_statistics": [], "p2_depths": [], "p2_surface_shape": [],
        "p2_target_shape": [], "p2_missing": [], "p2_interpolation": [],
        "p3_model": None, "p3_pred_ds": None, "p3_surface_ds": None,
        "p3_hotspots": [], "p3_uncertainty": {}, "p3_xai": {},
        "p3_impacts": [], "p3_future_fc": {}, "p3_times": [],
        "p3_lats": [], "p3_lons": [], "p3_norm_stats": {},
        "p3_argo_result": {}, "p3_multivar": {}, "p3_status": {},
        "config": {"paths": {"raw_data": "/tmp/ocean_raw"}},
        "db_path": ":memory:",
    }
    fake_pipeline.run_phase1 = lambda *a, **k: None
    fake_pipeline.run_phase2 = lambda *a, **k: None
    fake_pipeline.run_phase3 = lambda *a, **k: None

    # Stub admin
    fake_admin = sys.modules["admin"]
    fake_admin.verify_admin_token = mock.MagicMock(return_value=True)
    fake_admin.get_latest_prediction = mock.MagicMock(return_value={})
    fake_admin.get_heatwave_alert = mock.MagicMock(return_value={})
    fake_admin.get_ocean_summary = mock.MagicMock(return_value={})
    fake_admin.build_download_netcdf_path = mock.MagicMock(return_value="/tmp/x.nc")
    fake_admin.build_download_csv = mock.MagicMock(return_value="/tmp/x.csv")
    fake_admin.get_model_history = mock.MagicMock(return_value=[])
    fake_admin.get_admin_logs = mock.MagicMock(return_value=[])
    fake_admin.get_prediction_history = mock.MagicMock(return_value=[])
    fake_admin.get_training_history = mock.MagicMock(return_value=[])
    fake_admin.get_platform_statistics = mock.MagicMock(return_value={})
    fake_admin.init_admin_tables = mock.MagicMock()
    fake_admin.admin_upload_dataset = mock.MagicMock(return_value={"status": "ok"})
    fake_admin.admin_retrain_model = mock.MagicMock(return_value={"status": "ok"})

    import app as flask_app
    flask_app.app.config["TESTING"] = True
    flask_app.app.config["SECRET_KEY"] = "test-secret"

    with flask_app.app.test_client() as client:
        with flask_app.app.app_context():
            yield client

    for p in patchers:
        p.stop()


def _login_as(client, role, uid="test-uid", email="test@example.com"):
    """
    Helper for tests: sets the server-side session the way
    /api/auth/firebase-session would after a real, verified Firebase login.
    Routes protected by @fbauth.require_auth / @fbauth.require_role("admin")
    are authorized off this session -- see firebase_auth.py.
    """
    with client.session_transaction() as sess:
        sess["fb_uid"] = uid
        sess["fb_email"] = "arulgnanakumar@gmail.com" if role == "admin" else email
        sess["fb_role"] = role
    return client


@pytest.fixture
def flask_client_user(flask_client):
    """flask_client, pre-authenticated as a regular (non-admin) user."""
    return _login_as(flask_client, "user")


@pytest.fixture
def flask_client_admin(flask_client):
    """flask_client, pre-authenticated as an admin."""
    return _login_as(flask_client, "admin")


# ══════════════════════════════════════════════════════════════
# 1. SCHEMA INIT
# ══════════════════════════════════════════════════════════════

class TestPhase1Schema:
    REQUIRED_TABLES = [
        "users", "sessions", "user_settings",
        "favorite_locations", "map_bookmarks", "activity_log",
        "dataset_registry", "dataset_versions", "upload_logs",
        "model_registry", "api_keys", "scheduler_jobs",
        "report_catalog", "notifications",
    ]

    def test_all_tables_created(self, db):
        cur = db.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        for t in self.REQUIRED_TABLES:
            assert t in tables, f"Missing table: {t}"

    def test_init_idempotent(self, db):
        """Re-running init must not fail or duplicate data."""
        plat.init_phase1_tables(db)
        plat.init_phase1_tables(db)
        cur = db.execute("SELECT COUNT(*) FROM users")
        count = cur.fetchone()[0]
        assert count >= 1

    def test_default_users_seeded(self, db):
        cur = db.execute("SELECT username, role FROM users ORDER BY id")
        rows = cur.fetchall()
        usernames = [r[0] for r in rows]
        assert "admin" in usernames
        assert "scientist" in usernames
        assert "demo" in usernames

    def test_default_api_keys_seeded(self, db):
        cur = db.execute("SELECT COUNT(*) FROM api_keys")
        assert cur.fetchone()[0] >= 3

    def test_default_scheduler_seeded(self, db):
        cur = db.execute("SELECT COUNT(*) FROM scheduler_jobs")
        assert cur.fetchone()[0] >= 2


# ══════════════════════════════════════════════════════════════
# 2. AUTHENTICATION
# ══════════════════════════════════════════════════════════════

class TestAuthentication:
    def test_login_admin(self, db):
        user = plat.authenticate(db, "admin", "ocean_admin_2025")
        assert user is not None
        assert user["role"] == "admin"

    def test_login_scientist(self, db):
        user = plat.authenticate(db, "scientist", "ocean_sci_2025")
        assert user is not None
        assert user["role"] == "scientist"

    def test_login_demo(self, db):
        user = plat.authenticate(db, "demo", "demo123")
        assert user is not None

    def test_wrong_password_rejected(self, db):
        assert plat.authenticate(db, "admin", "wrongpass") is None

    def test_nonexistent_user_rejected(self, db):
        assert plat.authenticate(db, "ghost", "anything") is None

    def test_session_create_and_validate(self, db):
        user = plat.authenticate(db, "admin", "ocean_admin_2025")
        token = plat.create_session(db, user["id"])
        assert token and len(token) > 10
        resolved = plat.validate_session(db, token)
        assert resolved is not None
        assert resolved["username"] == "admin"

    def test_invalid_session_rejected(self, db):
        assert plat.validate_session(db, "fake_token_xyz") is None

    def test_session_invalidate(self, db):
        user = plat.authenticate(db, "admin", "ocean_admin_2025")
        token = plat.create_session(db, user["id"])
        plat.invalidate_session(db, token)
        assert plat.validate_session(db, token) is None

    def test_create_new_user(self, db):
        result = plat.create_user(db, "newuser", "new@test.com", "pass123", "guest", "New User")
        assert result["status"] == "created"

    def test_duplicate_username_error(self, db):
        result = plat.create_user(db, "admin", "x@x.com", "pass", "guest")
        assert result["status"] == "error"

    def test_get_user_by_id(self, db):
        u = plat.get_user_by_id(db, 1)
        assert u is not None
        assert "password_hash" not in u

    def test_list_users(self, db):
        users = plat.list_users(db)
        assert len(users) >= 3
        for u in users:
            assert "password_hash" not in u

    def test_update_user_role(self, db):
        result = plat.update_user_role(db, 3, "research_user")
        assert result["status"] == "updated"

    def test_invalid_role_rejected(self, db):
        result = plat.update_user_role(db, 1, "superadmin")
        assert result["status"] == "error"

    def test_deactivate_user(self, db):
        plat.create_user(db, "todelete", "d@d.com", "pass", "guest")
        cur = db.execute("SELECT id FROM users WHERE username='todelete'")
        uid = cur.fetchone()[0]
        result = plat.deactivate_user(db, uid)
        assert result["status"] == "deactivated"
        assert plat.authenticate(db, "todelete", "pass") is None

    def test_last_login_updated(self, db):
        plat.authenticate(db, "admin", "ocean_admin_2025")
        cur = db.execute("SELECT last_login FROM users WHERE username='admin'")
        ll = cur.fetchone()[0]
        assert ll is not None


# ══════════════════════════════════════════════════════════════
# 3. SETTINGS
# ══════════════════════════════════════════════════════════════

class TestSettings:
    def test_get_defaults(self, db):
        s = plat.get_settings(db, 999)  # non-existent user → defaults
        assert s["theme"] == "ocean_dark"
        assert s["temp_unit"] == "celsius"
        assert "notifications" in s

    def test_save_and_retrieve(self, db):
        plat.save_settings(db, 1, {"theme": "deep_sea", "sound": "true"})
        s = plat.get_settings(db, 1)
        assert s["theme"] == "deep_sea"
        assert s["sound"] == "true"

    def test_partial_save_merges_defaults(self, db):
        plat.save_settings(db, 1, {"theme": "arctic"})
        s = plat.get_settings(db, 1)
        assert s["temp_unit"] == "celsius"  # default preserved

    def test_unknown_key_ignored(self, db):
        result = plat.save_settings(db, 1, {"malicious_key": "value", "theme": "ocean_dark"})
        assert result["status"] == "saved"
        s = plat.get_settings(db, 1)
        assert "malicious_key" not in s

    def test_update_existing(self, db):
        plat.save_settings(db, 1, {"theme": "A"})
        plat.save_settings(db, 1, {"theme": "B"})
        assert plat.get_settings(db, 1)["theme"] == "B"

    def test_settings_per_user_isolated(self, db):
        plat.save_settings(db, 1, {"theme": "user1theme"})
        plat.save_settings(db, 2, {"theme": "user2theme"})
        assert plat.get_settings(db, 1)["theme"] == "user1theme"
        assert plat.get_settings(db, 2)["theme"] == "user2theme"


# ══════════════════════════════════════════════════════════════
# 4. FAVORITES
# ══════════════════════════════════════════════════════════════

class TestFavorites:
    def test_add_favorite(self, db):
        r = plat.add_favorite(db, 1, "Bay of Bengal", 13.0, 80.0)
        assert r["status"] == "added"
        assert r["id"] > 0

    def test_list_favorites(self, db):
        plat.add_favorite(db, 1, "BoB", 13.0, 80.0)
        plat.add_favorite(db, 1, "Arabian Sea", 15.0, 65.0)
        favs = plat.list_favorites(db, 1)
        assert len(favs) == 2

    def test_list_favorites_empty(self, db):
        assert plat.list_favorites(db, 999) == []

    def test_delete_favorite(self, db):
        r = plat.add_favorite(db, 1, "Test", 0.0, 0.0)
        fav_id = r["id"]
        del_r = plat.delete_favorite(db, 1, fav_id)
        assert del_r["status"] == "deleted"
        assert len(plat.list_favorites(db, 1)) == 0

    def test_favorite_isolation_between_users(self, db):
        plat.add_favorite(db, 1, "User1 Loc", 1.0, 1.0)
        plat.add_favorite(db, 2, "User2 Loc", 2.0, 2.0)
        assert len(plat.list_favorites(db, 1)) == 1
        assert len(plat.list_favorites(db, 2)) == 1

    def test_favorite_stores_coordinates(self, db):
        plat.add_favorite(db, 1, "Exact", 12.34, 56.78, "Test note")
        fav = plat.list_favorites(db, 1)[0]
        assert abs(fav["latitude"] - 12.34) < 0.001
        assert abs(fav["longitude"] - 56.78) < 0.001
        assert fav["notes"] == "Test note"


# ══════════════════════════════════════════════════════════════
# 5. BOOKMARKS
# ══════════════════════════════════════════════════════════════

class TestBookmarks:
    def test_add_bookmark(self, db):
        r = plat.add_bookmark(db, 1, "BoB View", 13.0, 80.0, 6, "sst")
        assert r["status"] == "added"

    def test_list_bookmarks(self, db):
        plat.add_bookmark(db, 1, "A", 10.0, 70.0)
        plat.add_bookmark(db, 1, "B", 15.0, 75.0)
        bms = plat.list_bookmarks(db, 1)
        assert len(bms) == 2

    def test_delete_bookmark(self, db):
        r = plat.add_bookmark(db, 1, "X", 0.0, 0.0)
        plat.delete_bookmark(db, 1, r["id"])
        assert plat.list_bookmarks(db, 1) == []

    def test_rename_bookmark(self, db):
        r = plat.add_bookmark(db, 1, "Old Name", 0.0, 0.0)
        ren = plat.rename_bookmark(db, 1, r["id"], "New Name")
        assert ren["status"] == "renamed"
        bm = plat.list_bookmarks(db, 1)[0]
        assert bm["name"] == "New Name"

    def test_bookmark_stores_zoom_and_layer(self, db):
        plat.add_bookmark(db, 1, "Z", 5.0, 60.0, zoom=8, layer="heatwave")
        bm = plat.list_bookmarks(db, 1)[0]
        assert bm["zoom"] == 8
        assert bm["layer"] == "heatwave"

    def test_bookmark_user_isolation(self, db):
        plat.add_bookmark(db, 1, "U1", 1.0, 1.0)
        plat.add_bookmark(db, 2, "U2", 2.0, 2.0)
        assert len(plat.list_bookmarks(db, 1)) == 1
        assert len(plat.list_bookmarks(db, 2)) == 1

    def test_cannot_delete_other_users_bookmark(self, db):
        r = plat.add_bookmark(db, 1, "Mine", 0.0, 0.0)
        plat.delete_bookmark(db, 2, r["id"])  # wrong user — should be no-op
        assert len(plat.list_bookmarks(db, 1)) == 1


# ══════════════════════════════════════════════════════════════
# 6. ACTIVITY LOG
# ══════════════════════════════════════════════════════════════

class TestActivityLog:
    def test_log_event(self, db):
        plat.log_activity(db, 1, "test_event", "Test detail")
        acts = plat.get_activity(db, 1)
        assert len(acts) >= 1
        assert acts[0]["event_type"] == "test_event"

    def test_log_without_user(self, db):
        plat.log_activity(db, None, "system_event", "No user")
        acts = plat.get_activity(db)
        assert any(a["event_type"] == "system_event" for a in acts)

    def test_log_with_metadata(self, db):
        plat.log_activity(db, 1, "upload", "Uploaded file", {"filename": "test.nc"})
        acts = plat.get_activity(db, 1)
        meta = json.loads(acts[0]["metadata"])
        assert meta["filename"] == "test.nc"

    def test_activity_limit(self, db):
        for i in range(20):
            plat.log_activity(db, 1, "loop", f"Event {i}")
        acts = plat.get_activity(db, 1, limit=5)
        assert len(acts) <= 5

    def test_activity_ordered_newest_first(self, db):
        for i in range(3):
            plat.log_activity(db, 1, "ordered", f"Event {i}")
        acts = plat.get_activity(db, 1)
        ids = [a["id"] for a in acts]
        assert ids == sorted(ids, reverse=True)

    def test_global_activity_no_user_filter(self, db):
        plat.log_activity(db, 1, "u1_event", "U1")
        plat.log_activity(db, 2, "u2_event", "U2")
        all_acts = plat.get_activity(db)
        types = [a["event_type"] for a in all_acts]
        assert "u1_event" in types
        assert "u2_event" in types


# ══════════════════════════════════════════════════════════════
# 7. DATASET REGISTRY
# ══════════════════════════════════════════════════════════════

class TestDatasetRegistry:
    def test_register_dataset(self, db, tmp_path):
        f = tmp_path / "sst_2025.nc"
        f.write_bytes(b"fake_nc_content")
        r = plat.register_dataset(db, "sst_2025.nc", str(f), "Copernicus", "SST")
        assert r["status"] == "registered"
        assert r["id"] > 0

    def test_list_datasets(self, db, tmp_path):
        for name in ["a.nc", "b.csv"]:
            f = tmp_path / name
            f.write_bytes(b"data")
            plat.register_dataset(db, name, str(f))
        ds = plat.list_datasets(db)
        assert len(ds) == 2

    def test_dataset_checksum_stored(self, db, tmp_path):
        f = tmp_path / "chk.nc"
        f.write_bytes(b"content123")
        plat.register_dataset(db, "chk.nc", str(f))
        ds = plat.list_datasets(db)[0]
        assert ds["checksum"] and len(ds["checksum"]) > 0

    def test_dataset_size_stored(self, db, tmp_path):
        f = tmp_path / "sz.nc"
        f.write_bytes(b"x" * 500)
        plat.register_dataset(db, "sz.nc", str(f))
        ds = plat.list_datasets(db)[0]
        assert ds["size_bytes"] == 500

    def test_delete_dataset(self, db, tmp_path):
        f = tmp_path / "del.nc"
        f.write_bytes(b"data")
        r = plat.register_dataset(db, "del.nc", str(f))
        plat.delete_dataset_registry(db, r["id"])
        assert plat.list_datasets(db) == []

    def test_dataset_stats(self, db, tmp_path):
        for name in ["a.nc", "b.nc"]:
            f = tmp_path / name
            f.write_bytes(b"content")
            plat.register_dataset(db, name, str(f), variable="SST")
        stats = plat.get_dataset_stats(db)
        assert stats["active_datasets"] == 2
        assert stats["total_bytes"] > 0

    def test_format_detection_nc(self, db, tmp_path):
        f = tmp_path / "data.nc"
        f.write_bytes(b"data")
        plat.register_dataset(db, "data.nc", str(f))
        ds = plat.list_datasets(db)[0]
        assert ds["file_format"] == "NetCDF"

    def test_format_detection_csv(self, db, tmp_path):
        f = tmp_path / "data.csv"
        f.write_bytes(b"col1,col2")
        plat.register_dataset(db, "data.csv", str(f))
        ds = plat.list_datasets(db)[0]
        assert ds["file_format"] == "CSV"

    def test_upload_log_created(self, db, tmp_path):
        f = tmp_path / "log.nc"
        f.write_bytes(b"data")
        plat.register_dataset(db, "log.nc", str(f))
        cur = db.execute("SELECT COUNT(*) FROM upload_logs")
        assert cur.fetchone()[0] >= 1


# ══════════════════════════════════════════════════════════════
# 8. MODEL REGISTRY
# ══════════════════════════════════════════════════════════════

class TestModelRegistry:
    def test_register_model(self, db):
        r = plat.register_model(db, "v1.0", rmse=0.45, mae=0.32)
        assert r["status"] == "registered"
        assert r["id"] > 0

    def test_list_models(self, db):
        plat.register_model(db, "v1.0")
        plat.register_model(db, "v2.0")
        models = plat.list_models(db)
        assert len(models) == 2

    def test_activate_model(self, db):
        r = plat.register_model(db, "v1.0")
        plat.activate_model(db, r["id"])
        models = plat.list_models(db)
        active = [m for m in models if m["status"] == "active"]
        assert len(active) == 1
        assert active[0]["version_tag"] == "v1.0"

    def test_only_one_active_at_a_time(self, db):
        r1 = plat.register_model(db, "v1.0")
        r2 = plat.register_model(db, "v2.0")
        plat.activate_model(db, r1["id"])
        plat.activate_model(db, r2["id"])
        models = plat.list_models(db)
        active = [m for m in models if m["status"] == "active"]
        assert len(active) == 1
        assert active[0]["version_tag"] == "v2.0"

    def test_rollback_model(self, db):
        r = plat.register_model(db, "v1.0")
        plat.activate_model(db, r["id"])
        plat.rollback_model(db, r["id"])
        models = plat.list_models(db)
        assert models[0]["status"] == "rolled_back"

    def test_compare_models(self, db):
        r1 = plat.register_model(db, "v1.0", rmse=0.45)
        r2 = plat.register_model(db, "v2.0", rmse=0.38)
        compared = plat.compare_models(db, [r1["id"], r2["id"]])
        assert len(compared) == 2
        tags = [m["version_tag"] for m in compared]
        assert "v1.0" in tags and "v2.0" in tags

    def test_compare_empty_ids(self, db):
        assert plat.compare_models(db, []) == []

    def test_model_stores_metrics(self, db):
        plat.register_model(db, "v1.0", rmse=0.45, mae=0.32, val_loss=0.001, epochs=50)
        m = plat.list_models(db)[0]
        assert abs(m["rmse"] - 0.45) < 0.001
        assert m["epochs"] == 50


# ══════════════════════════════════════════════════════════════
# 9. API KEY VAULT
# ══════════════════════════════════════════════════════════════

class TestApiKeyVault:
    def test_set_and_get_key(self, db):
        plat.set_api_key(db, "Copernicus Marine", "MY_SECRET_KEY_123")
        val = plat.get_api_key(db, "Copernicus Marine")
        assert val == "MY_SECRET_KEY_123"

    def test_get_nonexistent_key_empty(self, db):
        assert plat.get_api_key(db, "NonExistentService") == ""

    def test_update_existing_key(self, db):
        plat.set_api_key(db, "Copernicus Marine", "KEY_V1")
        plat.set_api_key(db, "Copernicus Marine", "KEY_V2")
        assert plat.get_api_key(db, "Copernicus Marine") == "KEY_V2"

    def test_list_api_keys(self, db):
        keys = plat.list_api_keys(db)
        assert len(keys) >= 3  # seeded defaults
        for k in keys:
            assert "key_enc" not in k  # plaintext key not exposed in list

    def test_test_api_key(self, db):
        r = plat.test_api_key(db, "Copernicus Marine")
        assert r["status"] == "tested"
        cur = db.execute("SELECT last_tested, status FROM api_keys WHERE service='Copernicus Marine'")
        row = cur.fetchone()
        assert row and row[1] == "tested"

    def test_toggle_disable(self, db):
        plat.set_api_key(db, "ERA5 / ECMWF", "key")
        plat.toggle_api_key(db, "ERA5 / ECMWF", False)
        cur = db.execute("SELECT enabled FROM api_keys WHERE service='ERA5 / ECMWF'")
        assert cur.fetchone()[0] == 0
        # Disabled key returns empty
        assert plat.get_api_key(db, "ERA5 / ECMWF") == ""

    def test_toggle_enable(self, db):
        plat.set_api_key(db, "NASA Earthdata", "key2")
        plat.toggle_api_key(db, "NASA Earthdata", False)
        plat.toggle_api_key(db, "NASA Earthdata", True)
        assert plat.get_api_key(db, "NASA Earthdata") == "key2"

    def test_key_encrypted_at_rest(self, db):
        plat.set_api_key(db, "Test Service", "PLAIN_SECRET")
        cur = db.execute("SELECT key_enc FROM api_keys WHERE service='Test Service'")
        enc = cur.fetchone()[0]
        assert enc != "PLAIN_SECRET"  # not stored in plaintext
        assert len(enc) > 0


# ══════════════════════════════════════════════════════════════
# 10. SCHEDULER
# ══════════════════════════════════════════════════════════════

class TestScheduler:
    def test_list_jobs(self, db):
        jobs = plat.list_jobs(db)
        assert len(jobs) >= 2

    def test_run_job_now(self, db):
        jobs = plat.list_jobs(db)
        job_name = jobs[0]["job_name"]
        r = plat.run_job_now(db, job_name)
        assert r["status"] == "triggered"
        cur = db.execute("SELECT status FROM scheduler_jobs WHERE job_name=?", (job_name,))
        assert cur.fetchone()[0] == "running"

    def test_update_interval(self, db):
        jobs = plat.list_jobs(db)
        job_name = jobs[0]["job_name"]
        r = plat.update_job_interval(db, job_name, "weekly")
        assert r["status"] == "updated"
        cur = db.execute("SELECT interval FROM scheduler_jobs WHERE job_name=?", (job_name,))
        assert cur.fetchone()[0] == "weekly"

    def test_invalid_interval_rejected(self, db):
        jobs = plat.list_jobs(db)
        r = plat.update_job_interval(db, jobs[0]["job_name"], "every_second")
        assert r["status"] == "error"


# ══════════════════════════════════════════════════════════════
# 11. NOTIFICATIONS
# ══════════════════════════════════════════════════════════════

class TestNotifications:
    def test_push_notification(self, db):
        r = plat.push_notification(db, "heatwave", "Heatwave detected!", "warning")
        assert r["status"] == "pushed"

    def test_list_all_notifications(self, db):
        plat.push_notification(db, "type1", "Msg1")
        plat.push_notification(db, "type2", "Msg2")
        notifs = plat.list_notifications(db)
        assert len(notifs) == 2

    def test_list_unread_only(self, db):
        plat.push_notification(db, "t1", "Unread 1")
        plat.push_notification(db, "t2", "Unread 2")
        plat.mark_notifications_read(db)
        plat.push_notification(db, "t3", "New unread")
        unread = plat.list_notifications(db, unread_only=True)
        assert len(unread) == 1
        assert unread[0]["message"] == "New unread"

    def test_mark_all_read(self, db):
        plat.push_notification(db, "t1", "Msg1")
        plat.push_notification(db, "t2", "Msg2")
        plat.mark_notifications_read(db)
        unread = plat.list_notifications(db, unread_only=True)
        assert len(unread) == 0

    def test_notification_limit(self, db):
        for i in range(20):
            plat.push_notification(db, "type", f"Msg {i}")
        limited = plat.list_notifications(db, limit=5)
        assert len(limited) == 5

    def test_notification_level_stored(self, db):
        plat.push_notification(db, "alert", "Danger!", "error")
        n = plat.list_notifications(db)[0]
        assert n["level"] == "error"


# ══════════════════════════════════════════════════════════════
# 12. DB HEALTH
# ══════════════════════════════════════════════════════════════

class TestDbHealth:
    def test_health_score_range(self, db):
        health = plat.get_db_health(db, ":memory:")
        assert 0 <= health["health_score"] <= 100

    def test_tables_listed(self, db):
        health = plat.get_db_health(db, ":memory:")
        names = [t["name"] for t in health["tables"]]
        assert "users" in names
        assert "model_registry" in names

    def test_total_records(self, db):
        plat.push_notification(db, "t", "m")
        health = plat.get_db_health(db, ":memory:")
        assert health["total_records"] > 0

    def test_checked_at_present(self, db):
        health = plat.get_db_health(db, ":memory:")
        assert "checked_at" in health
        assert len(health["checked_at"]) > 10


# ══════════════════════════════════════════════════════════════
# 13. REPORT CATALOG
# ══════════════════════════════════════════════════════════════

class TestReportCatalog:
    def test_register_report(self, db, tmp_path):
        f = tmp_path / "report.json"
        f.write_text("{}")
        r = plat.register_report(db, "qc", "QC Report 2025-01", str(f))
        assert r["status"] == "registered"

    def test_list_reports(self, db, tmp_path):
        for name in ["r1.json", "r2.json"]:
            f = tmp_path / name
            f.write_text("{}")
            plat.register_report(db, "test", name, str(f))
        reps = plat.list_reports(db)
        assert len(reps) == 2

    def test_report_size_stored(self, db, tmp_path):
        f = tmp_path / "sz.json"
        f.write_text('{"key": "value"}')
        plat.register_report(db, "qc", "Test", str(f))
        rep = plat.list_reports(db)[0]
        assert rep["size_bytes"] > 0


# ══════════════════════════════════════════════════════════════
# 14. PLATFORM STATS
# ══════════════════════════════════════════════════════════════

class TestPlatformStats:
    def test_stats_keys(self, db):
        stats = plat.get_platform_stats(db)
        for key in ["total_users", "total_datasets", "total_models",
                    "total_notifications", "total_favorites", "total_bookmarks"]:
            assert key in stats

    def test_user_count(self, db):
        stats = plat.get_platform_stats(db)
        assert stats["total_users"] >= 3  # seeded users

    def test_stats_increment_after_add(self, db):
        before = plat.get_platform_stats(db)["total_notifications"]
        plat.push_notification(db, "t", "m")
        after = plat.get_platform_stats(db)["total_notifications"]
        assert after == before + 1


# ══════════════════════════════════════════════════════════════
# 15. CRYPTO HELPERS
# ══════════════════════════════════════════════════════════════

class TestCryptoHelpers:
    def test_password_hash_deterministic(self):
        h1 = plat._hash_password("mypass")
        h2 = plat._hash_password("mypass")
        assert h1 == h2

    def test_different_passwords_different_hashes(self):
        assert plat._hash_password("pass1") != plat._hash_password("pass2")

    def test_xor_enc_dec_roundtrip(self):
        original = "MY_SECRET_API_KEY_12345"
        enc = plat._xor_enc(original, "ocean")
        dec = plat._xor_dec(enc, "ocean")
        assert dec == original

    def test_xor_enc_empty(self):
        assert plat._xor_enc("", "key") == ""
        assert plat._xor_dec("", "key") == ""

    def test_xor_enc_not_plaintext(self):
        enc = plat._xor_enc("secret", "ocean")
        assert enc != "secret"


# ══════════════════════════════════════════════════════════════
# 16. FLASK PAGE ROUTES (new pages)
# ══════════════════════════════════════════════════════════════

class TestFlaskPhase1PageRoutes:
    # Pages requiring only a signed-in session (any role)
    USER_PAGES = [
        "/settings", "/profile", "/downloads",
        "/activity", "/reports",
    ]
    # Pages requiring the admin role
    ADMIN_PAGES = [
        "/database", "/training", "/api-management", "/model-registry",
    ]
    # Fully public pages
    PUBLIC_PAGES = ["/login"]

    def test_public_pages_return_200(self, flask_client):
        for path in self.PUBLIC_PAGES:
            resp = flask_client.get(path)
            assert resp.status_code == 200, f"{path} -> {resp.status_code}"

    def test_user_pages_require_auth(self, flask_client):
        """Anonymous access to a protected page must redirect, not render."""
        for path in self.USER_PAGES:
            resp = flask_client.get(path)
            assert resp.status_code == 302, f"{path} should redirect anonymous users, got {resp.status_code}"

    def test_user_pages_return_200_when_authenticated(self, flask_client_user):
        for path in self.USER_PAGES:
            resp = flask_client_user.get(path)
            assert resp.status_code == 200, f"{path} -> {resp.status_code}"

    def test_admin_pages_require_admin_role(self, flask_client_user):
        """A signed-in but non-admin user must not reach admin-only pages."""
        for path in self.ADMIN_PAGES + ["/admin"]:
            resp = flask_client_user.get(path)
            assert resp.status_code == 302, f"{path} should redirect non-admin users, got {resp.status_code}"

    def test_admin_pages_return_200_for_admin(self, flask_client_admin):
        for path in self.ADMIN_PAGES + ["/admin"]:
            resp = flask_client_admin.get(path)
            assert resp.status_code == 200, f"{path} -> {resp.status_code}"

    def test_existing_public_pages_still_work(self, flask_client):
        # These remain intentionally public scientific-visualization pages.
        for path in ["/", "/earth", "/dashboard", "/heatwave", "/voice", "/about", "/gismap"]:
            resp = flask_client.get(path)
            assert resp.status_code == 200, f"{path} -> {resp.status_code}"

    def test_settings_contains_html(self, flask_client_user):
        resp = flask_client_user.get("/settings")
        assert b"Settings" in resp.data

    def test_login_contains_form(self, flask_client):
        resp = flask_client.get("/login")
        assert b"login" in resp.data.lower() or b"sign in" in resp.data.lower()

    def test_activity_page_loads(self, flask_client_user):
        resp = flask_client_user.get("/activity")
        assert b"Timeline" in resp.data or b"Activity" in resp.data

    def test_model_registry_page_loads(self, flask_client_admin):
        resp = flask_client_admin.get("/model-registry")
        assert b"Model" in resp.data

    def test_database_page_loads(self, flask_client_admin):
        resp = flask_client_admin.get("/database")
        assert b"Database" in resp.data


# ══════════════════════════════════════════════════════════════
# 17. FLASK AUTH API
# ══════════════════════════════════════════════════════════════

class TestFlaskAuthAPI:
    def test_login_success(self, flask_client):
        resp = flask_client.post("/api/auth/login",
            data=json.dumps({"username": "admin", "password": "ocean_admin_2025"}),
            content_type="application/json")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "ok"
        assert "token" in data
        assert data["user"]["role"] == "admin"

    def test_login_wrong_password(self, flask_client):
        resp = flask_client.post("/api/auth/login",
            data=json.dumps({"username": "admin", "password": "wrongpass"}),
            content_type="application/json")
        assert resp.status_code == 401

    def test_login_missing_fields(self, flask_client):
        resp = flask_client.post("/api/auth/login",
            data=json.dumps({}), content_type="application/json")
        assert resp.status_code == 400

    def test_register_new_user(self, flask_client):
        resp = flask_client.post("/api/auth/register",
            data=json.dumps({"username": "testuser99", "email": "t@t.com", "password": "pass"}),
            content_type="application/json")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "created"

    def test_logout(self, flask_client):
        resp = flask_client.post("/api/auth/logout",
            data=json.dumps({"token": "fake"}),
            content_type="application/json")
        assert resp.status_code == 200

    def test_me_unauthenticated(self, flask_client):
        resp = flask_client.get("/api/auth/me")
        assert resp.status_code == 401


# ══════════════════════════════════════════════════════════════
# 18. FLASK SETTINGS API
# ══════════════════════════════════════════════════════════════

class TestFlaskSettingsAPI:
    def test_get_settings(self, flask_client):
        resp = flask_client.get("/api/settings")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "settings" in data
        assert "theme" in data["settings"]

    def test_save_settings(self, flask_client):
        resp = flask_client.post("/api/settings",
            data=json.dumps({"theme": "deep_sea"}),
            content_type="application/json")
        assert resp.status_code == 200
        assert json.loads(resp.data)["status"] == "saved"


# ══════════════════════════════════════════════════════════════
# 19. FLASK FAVORITES API
# ══════════════════════════════════════════════════════════════

class TestFlaskFavoritesAPI:
    def test_list_favorites(self, flask_client):
        resp = flask_client.get("/api/favorites")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "favorites" in data

    def test_add_favorite(self, flask_client):
        resp = flask_client.post("/api/favorites",
            data=json.dumps({"name": "BoB", "lat": 13.0, "lon": 80.0}),
            content_type="application/json")
        assert resp.status_code == 200
        assert json.loads(resp.data)["status"] == "added"

    def test_delete_favorite(self, flask_client):
        # Add first
        add_resp = flask_client.post("/api/favorites",
            data=json.dumps({"name": "Del", "lat": 0.0, "lon": 0.0}),
            content_type="application/json")
        fav_id = json.loads(add_resp.data)["id"]
        del_resp = flask_client.delete(f"/api/favorites/{fav_id}")
        assert del_resp.status_code == 200
        assert json.loads(del_resp.data)["status"] == "deleted"


# ══════════════════════════════════════════════════════════════
# 20. FLASK BOOKMARKS API
# ══════════════════════════════════════════════════════════════

class TestFlaskBookmarksAPI:
    def test_list_bookmarks(self, flask_client):
        resp = flask_client.get("/api/bookmarks")
        assert resp.status_code == 200
        assert "bookmarks" in json.loads(resp.data)

    def test_add_bookmark(self, flask_client):
        resp = flask_client.post("/api/bookmarks",
            data=json.dumps({"name": "BoB", "lat": 13.0, "lon": 80.0, "zoom": 6, "layer": "sst"}),
            content_type="application/json")
        assert resp.status_code == 200
        assert json.loads(resp.data)["status"] == "added"

    def test_rename_bookmark(self, flask_client):
        add_r = flask_client.post("/api/bookmarks",
            data=json.dumps({"name": "Old", "lat": 0.0, "lon": 0.0}),
            content_type="application/json")
        bm_id = json.loads(add_r.data)["id"]
        resp = flask_client.patch(f"/api/bookmarks/{bm_id}",
            data=json.dumps({"name": "New Name"}),
            content_type="application/json")
        assert resp.status_code == 200
        assert json.loads(resp.data)["status"] == "renamed"

    def test_delete_bookmark(self, flask_client):
        add_r = flask_client.post("/api/bookmarks",
            data=json.dumps({"name": "Del", "lat": 0.0, "lon": 0.0}),
            content_type="application/json")
        bm_id = json.loads(add_r.data)["id"]
        resp = flask_client.delete(f"/api/bookmarks/{bm_id}")
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════
# 21. FLASK REGISTRY & PLATFORM APIs
# ══════════════════════════════════════════════════════════════

class TestFlaskRegistryAPI:
    def test_dataset_registry_list(self, flask_client):
        resp = flask_client.get("/api/registry/datasets")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "datasets" in data and "stats" in data

    def test_model_registry_list(self, flask_client):
        resp = flask_client.get("/api/registry/models")
        assert resp.status_code == 200
        assert "models" in json.loads(resp.data)

    def test_apikeys_list(self, flask_client_admin):
        resp = flask_client_admin.get("/api/apikeys")
        assert resp.status_code == 200
        assert "keys" in json.loads(resp.data)

    def test_scheduler_list(self, flask_client):
        resp = flask_client.get("/api/scheduler")
        assert resp.status_code == 200
        assert "jobs" in json.loads(resp.data)

    def test_notifications_list(self, flask_client):
        resp = flask_client.get("/api/notifications")
        assert resp.status_code == 200
        assert "notifications" in json.loads(resp.data)

    def test_db_health_endpoint(self, flask_client):
        resp = flask_client.get("/api/db/health")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "health_score" in data
        assert "tables" in data

    def test_platform_stats_endpoint(self, flask_client):
        resp = flask_client.get("/api/platform/stats")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "total_users" in data

    def test_activity_endpoint(self, flask_client):
        resp = flask_client.get("/api/activity")
        assert resp.status_code == 200
        assert "activity" in json.loads(resp.data)

    def test_reports_catalog_endpoint(self, flask_client):
        resp = flask_client.get("/api/reports/catalog")
        assert resp.status_code == 200
        assert "reports" in json.loads(resp.data)

    def test_run_scheduler_job(self, flask_client_admin):
        # Get a job name first
        jobs_r = flask_client_admin.get("/api/scheduler")
        jobs = json.loads(jobs_r.data)["jobs"]
        if jobs:
            job_name = jobs[0]["job_name"]
            resp = flask_client_admin.post("/api/scheduler/run",
                data=json.dumps({"job_name": job_name}),
                content_type="application/json")
            assert resp.status_code == 200

    def test_mark_notifications_read(self, flask_client):
        resp = flask_client.post("/api/notifications/read")
        assert resp.status_code == 200
        assert json.loads(resp.data)["status"] == "marked_read"
