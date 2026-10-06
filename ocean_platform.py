"""
platform.py
OceanVerse AI v4.0 — Phase 1
Platform layer: authentication, user roles, settings, favorites,
bookmarks, activity log, dataset registry, model registry,
API key vault, scheduler, report center, database health.

All functions are additive — no existing modules are touched.
SQLite-only, no external auth dependencies.
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ocean_platform")

# ──────────────────────────────────────────────────────────────
# SCHEMA  (Phase 1 tables — additive only)
# ──────────────────────────────────────────────────────────────

SCHEMA_PHASE1 = """
-- ── User accounts & sessions ──────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    username      TEXT UNIQUE NOT NULL,
    email         TEXT UNIQUE,
    password_hash TEXT NOT NULL,
    role          TEXT DEFAULT 'guest',
    display_name  TEXT,
    avatar_seed   TEXT,
    created_at    TEXT,
    last_login    TEXT,
    is_active     INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    token      TEXT UNIQUE NOT NULL,
    user_id    INTEGER,
    created_at TEXT,
    expires_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- ── Settings (per user or global key=value) ───────────────
CREATE TABLE IF NOT EXISTS user_settings (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id  INTEGER,
    key      TEXT NOT NULL,
    value    TEXT,
    UNIQUE(user_id, key),
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- ── Favorite locations ────────────────────────────────────
CREATE TABLE IF NOT EXISTS favorite_locations (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER,
    name       TEXT NOT NULL,
    latitude   REAL,
    longitude  REAL,
    notes      TEXT,
    created_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- ── Map bookmarks ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS map_bookmarks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER,
    name       TEXT NOT NULL,
    latitude   REAL,
    longitude  REAL,
    zoom       INTEGER DEFAULT 6,
    layer      TEXT DEFAULT 'sst',
    created_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
);

-- ── Activity / event timeline ─────────────────────────────
CREATE TABLE IF NOT EXISTS activity_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER,
    event_type TEXT,
    detail     TEXT,
    metadata   TEXT,
    created_at TEXT
);

-- ── Dataset registry (Phase 1 extension) ─────────────────
CREATE TABLE IF NOT EXISTS dataset_registry (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT NOT NULL,
    source        TEXT,
    variable      TEXT,
    coverage      TEXT,
    date_start    TEXT,
    date_end      TEXT,
    resolution    TEXT,
    checksum      TEXT,
    status        TEXT DEFAULT 'pending',
    uploader      TEXT,
    file_format   TEXT,
    size_bytes    INTEGER DEFAULT 0,
    nc_variables  TEXT,
    lat_range     TEXT,
    lon_range     TEXT,
    missing_pct   REAL,
    qc_score      REAL,
    notes         TEXT,
    uploaded_at   TEXT
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id   INTEGER,
    version_tag  TEXT,
    filepath     TEXT,
    created_at   TEXT,
    FOREIGN KEY(dataset_id) REFERENCES dataset_registry(id)
);

CREATE TABLE IF NOT EXISTS upload_logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id  INTEGER,
    action      TEXT,
    detail      TEXT,
    performed_by TEXT,
    performed_at TEXT
);

-- ── Model registry ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    version_tag    TEXT NOT NULL,
    checkpoint_path TEXT,
    architecture   TEXT,
    training_date  TEXT,
    epochs         INTEGER,
    rmse           REAL,
    mae            REAL,
    r2             REAL,
    val_loss       REAL,
    status         TEXT DEFAULT 'inactive',
    notes          TEXT,
    created_at     TEXT
);

-- ── API key vault (encrypted at rest via XOR+hex) ─────────
CREATE TABLE IF NOT EXISTS api_keys (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    service     TEXT UNIQUE NOT NULL,
    key_enc     TEXT,
    enabled     INTEGER DEFAULT 1,
    last_tested TEXT,
    last_sync   TEXT,
    status      TEXT DEFAULT 'unknown',
    notes       TEXT
);

-- ── Scheduler jobs ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS scheduler_jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name    TEXT UNIQUE NOT NULL,
    interval    TEXT DEFAULT 'daily',
    last_run    TEXT,
    next_run    TEXT,
    status      TEXT DEFAULT 'idle',
    log         TEXT
);

-- ── Reports ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS report_catalog (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    report_type TEXT,
    title       TEXT,
    filepath    TEXT,
    generated_at TEXT,
    generated_by TEXT,
    size_bytes  INTEGER DEFAULT 0
);

-- ── Notification inbox (persisted) ───────────────────────
CREATE TABLE IF NOT EXISTS notifications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT,
    message     TEXT,
    level       TEXT DEFAULT 'info',
    read        INTEGER DEFAULT 0,
    created_at  TEXT
);
"""


def init_phase1_tables(conn: sqlite3.Connection) -> None:
    """Create all Phase 1 tables. Idempotent (IF NOT EXISTS)."""
    conn.executescript(SCHEMA_PHASE1)
    conn.commit()
    _seed_default_data(conn)
    logger.info("Phase 1 SQLite tables initialised")


def _seed_default_data(conn: sqlite3.Connection) -> None:
    """Insert default rows if tables are empty."""
    cur = conn.cursor()

    # Default admin user (password: ocean_admin_2025)
    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        cur.execute("""INSERT INTO users
            (username, email, password_hash, role, display_name, avatar_seed, created_at)
            VALUES (?,?,?,?,?,?,?)""",
            ("admin", "admin@oceanverse.ai",
             _hash_password("ocean_admin_2025"),
             "admin", "OceanVerse Admin", "admin",
             _now()))
        cur.execute("""INSERT INTO users
            (username, email, password_hash, role, display_name, avatar_seed, created_at)
            VALUES (?,?,?,?,?,?,?)""",
            ("scientist", "scientist@oceanverse.ai",
             _hash_password("ocean_sci_2025"),
             "scientist", "Lead Scientist", "scientist",
             _now()))
        cur.execute("""INSERT INTO users
            (username, email, password_hash, role, display_name, avatar_seed, created_at)
            VALUES (?,?,?,?,?,?,?)""",
            ("demo", "demo@oceanverse.ai",
             _hash_password("demo123"),
             "guest", "Demo User", "demo",
             _now()))

    # Default API key entries (no real keys)
    cur.execute("SELECT COUNT(*) FROM api_keys")
    if cur.fetchone()[0] == 0:
        services = [
            ("Copernicus Marine", ""),
            ("ERA5 / ECMWF", ""),
            ("NASA Earthdata", ""),
            ("NOAA ERDDAP", ""),
            ("INCOIS ODAS", ""),
        ]
        for svc, key in services:
            cur.execute("""INSERT OR IGNORE INTO api_keys (service, key_enc, status)
                VALUES (?,?,?)""", (svc, _xor_enc(key, "ocean"), "not_configured"))

    # Default scheduler jobs
    cur.execute("SELECT COUNT(*) FROM scheduler_jobs")
    if cur.fetchone()[0] == 0:
        for name, interval in [
            ("SST Daily Sync", "daily"),
            ("ARGO Weekly Download", "weekly"),
            ("Model Health Check", "daily"),
            ("Report Generation", "weekly"),
        ]:
            cur.execute("""INSERT OR IGNORE INTO scheduler_jobs
                (job_name, interval, status) VALUES (?,?,?)""",
                (name, interval, "idle"))

    conn.commit()


# ──────────────────────────────────────────────────────────────
# CRYPTO HELPERS
# ──────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    salt = "oceanverse_salt_v1"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def _xor_enc(text: str, key: str) -> str:
    """Very lightweight XOR obfuscation for API key storage."""
    if not text:
        return ""
    key_bytes = (key * (len(text) // len(key) + 1))[:len(text)]
    return "".join(f"{ord(c) ^ ord(k):02x}" for c, k in zip(text, key_bytes))


def _xor_dec(enc: str, key: str) -> str:
    if not enc:
        return ""
    try:
        raw = bytes(int(enc[i:i+2], 16) for i in range(0, len(enc), 2))
        key_bytes = (key * (len(raw) // len(key) + 1))[:len(raw)]
        return "".join(chr(b ^ ord(k)) for b, k in zip(raw, key_bytes))
    except Exception:
        return ""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────
# AUTHENTICATION
# ──────────────────────────────────────────────────────────────

def authenticate(conn: sqlite3.Connection, username: str, password: str) -> Optional[dict]:
    """Return user dict on success, None on failure."""
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username=? AND is_active=1", (username,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    user = dict(zip(cols, row))
    if user["password_hash"] != _hash_password(password):
        return None
    # Update last_login
    conn.execute("UPDATE users SET last_login=? WHERE id=?", (_now(), user["id"]))
    conn.commit()
    log_activity(conn, user["id"], "login", f"User {username} logged in")
    return user


def create_session(conn: sqlite3.Connection, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    conn.execute("""INSERT INTO sessions (token, user_id, created_at, expires_at)
        VALUES (?,?,?,datetime('now','+7 days'))""",
        (token, user_id, _now()))
    conn.commit()
    return token


def validate_session(conn: sqlite3.Connection, token: str) -> Optional[dict]:
    cur = conn.cursor()
    cur.execute("""SELECT u.* FROM users u
        JOIN sessions s ON s.user_id=u.id
        WHERE s.token=? AND s.expires_at > datetime('now') AND u.is_active=1""",
        (token,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    return dict(zip(cols, row))


def invalidate_session(conn: sqlite3.Connection, token: str) -> None:
    conn.execute("DELETE FROM sessions WHERE token=?", (token,))
    conn.commit()


def create_user(conn: sqlite3.Connection, username: str, email: str,
                password: str, role: str = "guest", display_name: str = "") -> dict:
    try:
        conn.execute("""INSERT INTO users
            (username, email, password_hash, role, display_name, avatar_seed, created_at)
            VALUES (?,?,?,?,?,?,?)""",
            (username, email, _hash_password(password), role,
             display_name or username, username[:6], _now()))
        conn.commit()
        return {"status": "created", "username": username}
    except sqlite3.IntegrityError as e:
        return {"status": "error", "error": str(e)}


def get_user_by_id(conn: sqlite3.Connection, user_id: int) -> Optional[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    if not row:
        return None
    cols = [d[0] for d in cur.description]
    u = dict(zip(cols, row))
    u.pop("password_hash", None)
    return u


def list_users(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute("SELECT id,username,email,role,display_name,created_at,last_login,is_active FROM users ORDER BY id")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def update_user_role(conn: sqlite3.Connection, user_id: int, role: str) -> dict:
    allowed = {"guest", "research_user", "scientist", "admin"}
    if role not in allowed:
        return {"status": "error", "error": "Invalid role"}
    conn.execute("UPDATE users SET role=? WHERE id=?", (role, user_id))
    conn.commit()
    return {"status": "updated", "user_id": user_id, "role": role}


def deactivate_user(conn: sqlite3.Connection, user_id: int) -> dict:
    conn.execute("UPDATE users SET is_active=0 WHERE id=?", (user_id,))
    conn.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
    conn.commit()
    return {"status": "deactivated", "user_id": user_id}


# ──────────────────────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────────────────────

_DEFAULT_SETTINGS = {
    "theme":           "ocean_dark",
    "language":        "en",
    "temp_unit":       "celsius",
    "wind_unit":       "ms",
    "map_style":       "ocean",
    "animation_speed": "normal",
    "notifications":   "true",
    "sound":           "false",
    "auto_refresh":    "true",
    "grid_visible":    "true",
    "ocean_opacity":   "0.7",
}


def get_settings(conn: sqlite3.Connection, user_id: int) -> dict:
    cur = conn.cursor()
    cur.execute("SELECT key, value FROM user_settings WHERE user_id=?", (user_id,))
    rows = dict(cur.fetchall())
    merged = {**_DEFAULT_SETTINGS, **rows}
    return merged


def save_settings(conn: sqlite3.Connection, user_id: int, settings: dict) -> dict:
    allowed = set(_DEFAULT_SETTINGS.keys())
    for k, v in settings.items():
        if k not in allowed:
            continue
        conn.execute("""INSERT INTO user_settings (user_id, key, value)
            VALUES (?,?,?)
            ON CONFLICT(user_id, key) DO UPDATE SET value=excluded.value""",
            (user_id, k, str(v)))
    conn.commit()
    log_activity(conn, user_id, "settings_save", f"Saved {len(settings)} settings")
    return {"status": "saved", "count": len(settings)}


# ──────────────────────────────────────────────────────────────
# FAVORITE LOCATIONS
# ──────────────────────────────────────────────────────────────

def add_favorite(conn: sqlite3.Connection, user_id: int,
                 name: str, lat: float, lon: float, notes: str = "") -> dict:
    cur = conn.execute("""INSERT INTO favorite_locations
        (user_id, name, latitude, longitude, notes, created_at) VALUES (?,?,?,?,?,?)""",
        (user_id, name, lat, lon, notes, _now()))
    conn.commit()
    fav_id = cur.lastrowid
    log_activity(conn, user_id, "add_favorite", f"Added {name} ({lat},{lon})")
    return {"status": "added", "id": fav_id}


def list_favorites(conn: sqlite3.Connection, user_id: int) -> list:
    cur = conn.cursor()
    cur.execute("SELECT * FROM favorite_locations WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def delete_favorite(conn: sqlite3.Connection, user_id: int, fav_id: int) -> dict:
    conn.execute("DELETE FROM favorite_locations WHERE id=? AND user_id=?", (fav_id, user_id))
    conn.commit()
    return {"status": "deleted", "id": fav_id}


# ──────────────────────────────────────────────────────────────
# MAP BOOKMARKS
# ──────────────────────────────────────────────────────────────

def add_bookmark(conn: sqlite3.Connection, user_id: int,
                 name: str, lat: float, lon: float,
                 zoom: int = 6, layer: str = "sst") -> dict:
    cur = conn.execute("""INSERT INTO map_bookmarks
        (user_id, name, latitude, longitude, zoom, layer, created_at) VALUES (?,?,?,?,?,?,?)""",
        (user_id, name, lat, lon, zoom, layer, _now()))
    conn.commit()
    return {"status": "added", "id": cur.lastrowid}


def list_bookmarks(conn: sqlite3.Connection, user_id: int) -> list:
    cur = conn.cursor()
    cur.execute("SELECT * FROM map_bookmarks WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def delete_bookmark(conn: sqlite3.Connection, user_id: int, bm_id: int) -> dict:
    conn.execute("DELETE FROM map_bookmarks WHERE id=? AND user_id=?", (bm_id, user_id))
    conn.commit()
    return {"status": "deleted", "id": bm_id}


def rename_bookmark(conn: sqlite3.Connection, user_id: int, bm_id: int, new_name: str) -> dict:
    conn.execute("UPDATE map_bookmarks SET name=? WHERE id=? AND user_id=?",
                 (new_name, bm_id, user_id))
    conn.commit()
    return {"status": "renamed", "id": bm_id, "name": new_name}


# ──────────────────────────────────────────────────────────────
# ACTIVITY LOG
# ──────────────────────────────────────────────────────────────

def log_activity(conn: sqlite3.Connection, user_id: Optional[int],
                 event_type: str, detail: str, metadata: dict = None) -> None:
    conn.execute("""INSERT INTO activity_log (user_id, event_type, detail, metadata, created_at)
        VALUES (?,?,?,?,?)""",
        (user_id, event_type, detail,
         json.dumps(metadata) if metadata else None, _now()))
    conn.commit()


def get_activity(conn: sqlite3.Connection, user_id: Optional[int] = None,
                 limit: int = 100) -> list:
    cur = conn.cursor()
    if user_id:
        cur.execute("""SELECT a.*, u.username FROM activity_log a
            LEFT JOIN users u ON a.user_id=u.id
            WHERE a.user_id=? ORDER BY a.id DESC LIMIT ?""", (user_id, limit))
    else:
        cur.execute("""SELECT a.*, u.username FROM activity_log a
            LEFT JOIN users u ON a.user_id=u.id
            ORDER BY a.id DESC LIMIT ?""", (limit,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ──────────────────────────────────────────────────────────────
# DATASET REGISTRY
# ──────────────────────────────────────────────────────────────

def register_dataset(conn: sqlite3.Connection, filename: str, filepath: str,
                     source: str = "", variable: str = "",
                     uploader: str = "admin", extra: dict = None) -> dict:
    """Register a new dataset upload into the registry."""
    size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    checksum = _file_checksum(filepath)
    ext = Path(filename).suffix.lower()
    file_format = "NetCDF" if ext in {".nc", ".nc4"} else "CSV" if ext == ".csv" else ext[1:].upper()

    meta = extra or {}
    cur = conn.execute("""INSERT INTO dataset_registry
        (filename, source, variable, checksum, status, uploader,
         file_format, size_bytes, nc_variables, lat_range, lon_range,
         missing_pct, qc_score, notes, uploaded_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (filename, source, variable, checksum, "active", uploader,
         file_format, size,
         meta.get("nc_variables"), meta.get("lat_range"),
         meta.get("lon_range"), meta.get("missing_pct"),
         meta.get("qc_score"), meta.get("notes"), _now()))
    ds_id = cur.lastrowid
    conn.execute("""INSERT INTO upload_logs (dataset_id, action, detail, performed_by, performed_at)
        VALUES (?,?,?,?,?)""", (ds_id, "register", f"Registered {filename}", uploader, _now()))
    conn.commit()
    log_activity(conn, None, "dataset_register", f"Registered {filename} (id={ds_id})")
    return {"status": "registered", "id": ds_id, "filename": filename}


def list_datasets(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute("SELECT * FROM dataset_registry ORDER BY uploaded_at DESC")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def delete_dataset_registry(conn: sqlite3.Connection, ds_id: int, filepath: str = "") -> dict:
    if filepath and os.path.exists(filepath):
        os.remove(filepath)
    conn.execute("DELETE FROM dataset_registry WHERE id=?", (ds_id,))
    conn.execute("""INSERT INTO upload_logs (dataset_id, action, detail, performed_at)
        VALUES (?,?,?,?)""", (ds_id, "delete", f"Deleted dataset id={ds_id}", _now()))
    conn.commit()
    return {"status": "deleted", "id": ds_id}


def get_dataset_stats(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM dataset_registry WHERE status='active'")
    active = cur.fetchone()[0]
    cur.execute("SELECT SUM(size_bytes) FROM dataset_registry")
    total_size = cur.fetchone()[0] or 0
    cur.execute("SELECT variable, COUNT(*) as cnt FROM dataset_registry GROUP BY variable ORDER BY cnt DESC LIMIT 5")
    by_var = [{"variable": r[0], "count": r[1]} for r in cur.fetchall()]
    return {"active_datasets": active, "total_bytes": total_size,
            "total_mb": round(total_size / 1_048_576, 2), "by_variable": by_var}


def _file_checksum(filepath: str) -> str:
    if not os.path.exists(filepath):
        return ""
    h = hashlib.sha256()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return ""


# ──────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ──────────────────────────────────────────────────────────────

def register_model(conn: sqlite3.Connection, version_tag: str,
                   checkpoint_path: str = "", architecture: str = "OceanTransformer",
                   rmse: float = None, mae: float = None, val_loss: float = None,
                   epochs: int = None, notes: str = "") -> dict:
    cur = conn.execute("""INSERT INTO model_registry
        (version_tag, checkpoint_path, architecture, training_date,
         epochs, rmse, mae, val_loss, status, notes, created_at)
        VALUES (?,?,?,?,?,?,?,?,'inactive',?,?)""",
        (version_tag, checkpoint_path, architecture, _now()[:10],
         epochs, rmse, mae, val_loss, notes, _now()))
    model_id = cur.lastrowid
    conn.commit()
    log_activity(conn, None, "model_register", f"Registered model {version_tag}")
    return {"status": "registered", "id": model_id, "version_tag": version_tag}


def list_models(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute("SELECT * FROM model_registry ORDER BY created_at DESC")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def activate_model(conn: sqlite3.Connection, model_id: int) -> dict:
    conn.execute("UPDATE model_registry SET status='inactive'")
    conn.execute("UPDATE model_registry SET status='active' WHERE id=?", (model_id,))
    conn.commit()
    log_activity(conn, None, "model_activate", f"Activated model id={model_id}")
    return {"status": "activated", "id": model_id}


def rollback_model(conn: sqlite3.Connection, model_id: int) -> dict:
    conn.execute("UPDATE model_registry SET status='rolled_back' WHERE id=?", (model_id,))
    conn.commit()
    return {"status": "rolled_back", "id": model_id}


def compare_models(conn: sqlite3.Connection, ids: list) -> list:
    if not ids:
        return []
    placeholders = ",".join("?" * len(ids))
    cur = conn.cursor()
    cur.execute(f"SELECT * FROM model_registry WHERE id IN ({placeholders}) ORDER BY id", ids)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ──────────────────────────────────────────────────────────────
# API KEY VAULT
# ──────────────────────────────────────────────────────────────

def set_api_key(conn: sqlite3.Connection, service: str, key: str, notes: str = "") -> dict:
    conn.execute("""INSERT INTO api_keys (service, key_enc, enabled, notes)
        VALUES (?,?,1,?)
        ON CONFLICT(service) DO UPDATE SET key_enc=excluded.key_enc, notes=excluded.notes""",
        (service, _xor_enc(key, "ocean"), notes))
    conn.commit()
    return {"status": "saved", "service": service}


def get_api_key(conn: sqlite3.Connection, service: str) -> str:
    cur = conn.cursor()
    cur.execute("SELECT key_enc FROM api_keys WHERE service=? AND enabled=1", (service,))
    row = cur.fetchone()
    return _xor_dec(row[0], "ocean") if row and row[0] else ""


def list_api_keys(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute("SELECT id, service, enabled, last_tested, last_sync, status, notes FROM api_keys ORDER BY service")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def test_api_key(conn: sqlite3.Connection, service: str) -> dict:
    """Mark key as tested (actual HTTP test would need external call)."""
    conn.execute("UPDATE api_keys SET last_tested=?, status='tested' WHERE service=?",
                 (_now(), service))
    conn.commit()
    return {"status": "tested", "service": service, "timestamp": _now()}


def toggle_api_key(conn: sqlite3.Connection, service: str, enabled: bool) -> dict:
    conn.execute("UPDATE api_keys SET enabled=? WHERE service=?",
                 (1 if enabled else 0, service))
    conn.commit()
    return {"status": "updated", "service": service, "enabled": enabled}


# ──────────────────────────────────────────────────────────────
# SCHEDULER
# ──────────────────────────────────────────────────────────────

def list_jobs(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute("SELECT * FROM scheduler_jobs ORDER BY job_name")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def run_job_now(conn: sqlite3.Connection, job_name: str) -> dict:
    conn.execute("""UPDATE scheduler_jobs
        SET last_run=?, status='running', log='Manual run triggered'
        WHERE job_name=?""", (_now(), job_name))
    conn.commit()
    log_activity(conn, None, "scheduler_run", f"Manual trigger: {job_name}")
    return {"status": "triggered", "job": job_name, "timestamp": _now()}


def update_job_interval(conn: sqlite3.Connection, job_name: str, interval: str) -> dict:
    allowed = {"hourly", "daily", "weekly", "manual"}
    if interval not in allowed:
        return {"status": "error", "error": "Invalid interval"}
    conn.execute("UPDATE scheduler_jobs SET interval=? WHERE job_name=?", (interval, job_name))
    conn.commit()
    return {"status": "updated", "job": job_name, "interval": interval}


# ──────────────────────────────────────────────────────────────
# NOTIFICATIONS (persisted)
# ──────────────────────────────────────────────────────────────

def push_notification(conn: sqlite3.Connection, kind: str, message: str,
                      level: str = "info") -> dict:
    conn.execute("""INSERT INTO notifications (kind, message, level, read, created_at)
        VALUES (?,?,?,0,?)""", (kind, message, level, _now()))
    conn.commit()
    return {"status": "pushed", "kind": kind}


def list_notifications(conn: sqlite3.Connection, unread_only: bool = False,
                       limit: int = 50) -> list:
    cur = conn.cursor()
    if unread_only:
        cur.execute("SELECT * FROM notifications WHERE read=0 ORDER BY id DESC LIMIT ?", (limit,))
    else:
        cur.execute("SELECT * FROM notifications ORDER BY id DESC LIMIT ?", (limit,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def mark_notifications_read(conn: sqlite3.Connection) -> dict:
    conn.execute("UPDATE notifications SET read=1")
    conn.commit()
    return {"status": "marked_read"}


# ──────────────────────────────────────────────────────────────
# DATABASE HEALTH
# ──────────────────────────────────────────────────────────────

def get_db_health(conn: sqlite3.Connection, db_path: str) -> dict:
    cur = conn.cursor()
    health = {"tables": [], "total_records": 0, "storage_bytes": 0, "health_score": 100}

    # All Phase 1 tables to check
    all_tables = [
        "users", "sessions", "user_settings", "favorite_locations",
        "map_bookmarks", "activity_log", "dataset_registry",
        "dataset_versions", "upload_logs", "model_registry",
        "api_keys", "scheduler_jobs", "report_catalog",
        "notifications",
        # Existing tables
        "datasets", "admin_uploads", "admin_actions",
        "model_history", "dataset_history", "prediction_history",
    ]

    for table in all_tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            count = cur.fetchone()[0]
            health["tables"].append({"name": table, "records": count, "ok": True})
            health["total_records"] += count
        except Exception:
            health["tables"].append({"name": table, "records": 0, "ok": False})
            health["health_score"] -= 2

    if os.path.exists(db_path):
        health["storage_bytes"] = os.path.getsize(db_path)
        health["storage_kb"] = round(health["storage_bytes"] / 1024, 1)

    health["health_score"] = max(0, health["health_score"])
    health["checked_at"] = _now()
    return health


# ──────────────────────────────────────────────────────────────
# REPORT CATALOG
# ──────────────────────────────────────────────────────────────

def register_report(conn: sqlite3.Connection, report_type: str,
                    title: str, filepath: str, generated_by: str = "system") -> dict:
    size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    conn.execute("""INSERT INTO report_catalog
        (report_type, title, filepath, generated_at, generated_by, size_bytes)
        VALUES (?,?,?,?,?,?)""",
        (report_type, title, filepath, _now(), generated_by, size))
    conn.commit()
    return {"status": "registered", "type": report_type}


def list_reports(conn: sqlite3.Connection) -> list:
    cur = conn.cursor()
    cur.execute("SELECT * FROM report_catalog ORDER BY generated_at DESC")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


# ──────────────────────────────────────────────────────────────
# PLATFORM STATS (for admin dashboard counters)
# ──────────────────────────────────────────────────────────────

def get_platform_stats(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    stats = {}
    table_counts = {
        "total_users":        "SELECT COUNT(*) FROM users WHERE is_active=1",
        "total_datasets":     "SELECT COUNT(*) FROM dataset_registry WHERE status='active'",
        "total_models":       "SELECT COUNT(*) FROM model_registry",
        "active_model_count": "SELECT COUNT(*) FROM model_registry WHERE status='active'",
        "total_notifications":"SELECT COUNT(*) FROM notifications",
        "unread_notifications":"SELECT COUNT(*) FROM notifications WHERE read=0",
        "total_favorites":   "SELECT COUNT(*) FROM favorite_locations",
        "total_bookmarks":   "SELECT COUNT(*) FROM map_bookmarks",
        "total_reports":     "SELECT COUNT(*) FROM report_catalog",
        "total_activity":    "SELECT COUNT(*) FROM activity_log",
    }
    for key, sql in table_counts.items():
        try:
            cur.execute(sql)
            stats[key] = cur.fetchone()[0]
        except Exception:
            stats[key] = 0
    return stats
