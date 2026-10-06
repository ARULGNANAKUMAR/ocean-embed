"""
test_phase3_twin.py
OceanVerse AI v4.0 — Phase 3 Digital Twin Tests
150+ tests covering:
  - digital_twin.py module (all data generators)
  - Ocean volume (voxels, depth layers, thermocline)
  - Particle data (current + wind)
  - ARGO animation keyframes
  - Satellite orbits
  - Marine objects
  - Bathymetry terrain
  - Heatwave glow field
  - Coral hotspots
  - Timeline frames
  - AI explanation HUD
  - Wave surface parameters
  - Scene descriptor
  - VR descriptor
  - LOD descriptor
  - All Flask /api/twin/* and /api/ocean-volume etc. endpoints
  - /twin page renders 200
  - lat/lon to XYZ conversions
  - colour mapping utilities
  - SST synthetic generator
"""

import json
import math
import sys
from pathlib import Path
import unittest.mock as mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import digital_twin as dt


# ──────────────────────────────────────────────────────────────
# FLASK FIXTURE
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def flask_client():
    mods = ["pipeline","ai_engine","evaluation","visualization",
            "preprocessing","admin","flask_socketio","xarray","numpy","pandas"]
    patchers = []
    for m in mods:
        p = mock.patch.dict("sys.modules", {m: mock.MagicMock()})
        p.start(); patchers.append(p)

    np_mock = sys.modules["numpy"]
    np_mock.floating = float; np_mock.integer = int; np_mock.ndarray = list
    np_mock.argmin = lambda x: 0; np_mock.abs = abs

    fp = sys.modules["pipeline"]
    fp._state = {
        "p3_hotspots": [], "p3_argo_result": {"floats": []},
        "p3_pred_ds": None, "p3_surface_ds": None,
        "p3_uncertainty": {}, "p3_multivar": {}, "p3_status": {},
        "p2_status": {}, "p3_times": [], "p3_lats": [], "p3_lons": [],
        "p3_norm_stats": {}, "p3_impacts": [], "p3_future_fc": {},
        "p3_xai": {}, "p2_statistics": [], "p2_depths": [],
        "p2_surface_shape": [], "p2_target_shape": [], "p2_missing": [],
        "p2_interpolation": [], "scan_records": [], "inspections": {},
        "qc_results": [], "cov_results": [], "summary": {}, "p3_model": None,
        "config": {"paths": {"raw_data": "/tmp"}}, "db_path": ":memory:",
    }
    fp.run_phase1 = fp.run_phase2 = fp.run_phase3 = lambda *a, **k: None

    fa = sys.modules["admin"]
    fa.verify_admin_token = mock.MagicMock(return_value=True)
    for attr in ["get_latest_prediction","get_heatwave_alert","get_ocean_summary","get_platform_statistics"]:
        setattr(fa, attr, mock.MagicMock(return_value={}))
    for attr in ["get_model_history","get_admin_logs","get_prediction_history","get_training_history"]:
        setattr(fa, attr, mock.MagicMock(return_value=[]))
    fa.build_download_netcdf_path = mock.MagicMock(return_value="/tmp/x.nc")
    fa.build_download_csv = mock.MagicMock(return_value="/tmp/x.csv")
    fa.init_admin_tables = mock.MagicMock()
    fa.admin_upload_dataset = fa.admin_retrain_model = mock.MagicMock(return_value={"status":"ok"})

    import app as flask_app
    flask_app.app.config["TESTING"] = True
    flask_app.app.config["SECRET_KEY"] = "twin-test"

    with flask_app.app.test_client() as client:
        with flask_app.app.app_context():
            yield client

    for p in patchers:
        p.stop()


def _login_as(client, role, uid="twin-test-uid", email="twin-test@example.com"):
    with client.session_transaction() as sess:
        sess["fb_uid"] = uid
        sess["fb_email"] = "arulgnanakumar@gmail.com" if role == "admin" else email
        sess["fb_role"] = role
    return client


@pytest.fixture
def flask_client_user(flask_client):
    return _login_as(flask_client, "user")


@pytest.fixture
def flask_client_admin(flask_client):
    return _login_as(flask_client, "admin")


# ══════════════════════════════════════════════════════════════
# 1. UTILITIES
# ══════════════════════════════════════════════════════════════

class TestUtilities:
    def test_ll2xyz_north_pole(self):
        pos = dt._lat_lon_to_xyz(90, 0, 5.0)
        assert abs(pos["y"] - 5.0) < 0.01
        assert abs(pos["x"]) < 0.01
        assert abs(pos["z"]) < 0.01

    def test_ll2xyz_equator_prime(self):
        pos = dt._lat_lon_to_xyz(0, 0, 5.0)
        assert abs(pos["y"]) < 0.01
        length = math.sqrt(pos["x"]**2 + pos["y"]**2 + pos["z"]**2)
        assert abs(length - 5.0) < 0.05

    def test_ll2xyz_radius(self):
        for lat in [-30, 0, 13, 30]:
            for lon in [60, 80, 100]:
                pos = dt._lat_lon_to_xyz(lat, lon, 5.0)
                length = math.sqrt(pos["x"]**2 + pos["y"]**2 + pos["z"]**2)
                assert abs(length - 5.0) < 0.05, f"Bad radius at {lat},{lon}"

    def test_ll2xyz_returns_dict(self):
        pos = dt._lat_lon_to_xyz(13, 80)
        assert "x" in pos and "y" in pos and "z" in pos

    def test_sst_color_cold(self):
        r, g, b = dt._sst_color(0.0)
        assert r < 0.2  # blue-dominant

    def test_sst_color_hot(self):
        r, g, b = dt._sst_color(1.0)
        assert r > 0.5  # red-dominant

    def test_sst_color_mid(self):
        r, g, b = dt._sst_color(0.5)
        # mid should be greenish
        assert 0.0 <= r <= 1.0
        assert 0.0 <= g <= 1.0
        assert 0.0 <= b <= 1.0

    def test_sst_color_clamps(self):
        c_low = dt._sst_color(-0.5)
        c_high = dt._sst_color(1.5)
        c_zero = dt._sst_color(0.0)
        c_one = dt._sst_color(1.0)
        assert c_low == c_zero
        assert c_high == c_one

    def test_sst_at_surface(self):
        sst = dt._sst_at(13.0, 80.0, 0)
        assert 24 <= sst <= 36

    def test_sst_at_deep(self):
        sst_sfc = dt._sst_at(13.0, 80.0, 0)
        sst_deep = dt._sst_at(13.0, 80.0, 14)  # 1500m
        assert sst_deep < sst_sfc

    def test_sst_at_hotspot(self):
        hs = [{"latitude": 13.0, "longitude": 80.0, "severity": "EXTREME"}]
        sst_hw = dt._sst_at(13.0, 80.0, 0, hs)
        sst_no = dt._sst_at(13.0, 80.0, 0, [])
        assert sst_hw >= sst_no

    def test_lerp_zero(self):
        assert dt._lerp(0, 10, 0) == 0

    def test_lerp_one(self):
        assert dt._lerp(0, 10, 1) == 10

    def test_lerp_half(self):
        assert dt._lerp(0, 10, 0.5) == 5.0


# ══════════════════════════════════════════════════════════════
# 2. OCEAN VOLUME
# ══════════════════════════════════════════════════════════════

class TestOceanVolume:
    def test_returns_dict(self):
        data = dt.get_ocean_volume(depth_indices=[0, 1])
        assert isinstance(data, dict)

    def test_type_key(self):
        data = dt.get_ocean_volume(depth_indices=[0])
        assert data["type"] == "ocean_volume"

    def test_voxels_present(self):
        data = dt.get_ocean_volume(depth_indices=[0])
        assert "voxels" in data
        assert len(data["voxels"]) > 0

    def test_voxel_fields(self):
        data = dt.get_ocean_volume(depth_indices=[0])
        v = data["voxels"][0]
        for field in ["x","y","z","r","g","b","a","sst","depth_m","lat","lon"]:
            assert field in v, f"Missing field: {field}"

    def test_voxel_xyz_on_sphere(self):
        data = dt.get_ocean_volume(depth_indices=[0])
        for v in data["voxels"][:20]:
            r = math.sqrt(v["x"]**2 + v["y"]**2 + v["z"]**2)
            assert 4.8 <= r <= 5.2, f"Voxel off sphere: r={r}"

    def test_voxel_sst_range(self):
        data = dt.get_ocean_volume(depth_indices=[0])
        for v in data["voxels"][:30]:
            assert 4 <= v["sst"] <= 36

    def test_voxel_alpha_range(self):
        data = dt.get_ocean_volume(depth_indices=[0])
        for v in data["voxels"][:10]:
            assert 0.0 <= v["a"] <= 1.0

    def test_depth_indices_filter(self):
        single = dt.get_ocean_volume(depth_indices=[0])
        multi  = dt.get_ocean_volume(depth_indices=[0, 3, 7])
        assert single["depth_levels"] == 1
        assert multi["depth_levels"] == 3
        assert multi["voxel_count"] > single["voxel_count"]

    def test_hotspot_increases_sst(self):
        hs = [{"latitude": 13.0, "longitude": 80.0, "severity": "EXTREME"}]
        v_hw = dt.get_ocean_volume(hotspots=hs, depth_indices=[0])
        v_no = dt.get_ocean_volume(hotspots=[],  depth_indices=[0])
        avg_hw = sum(v["sst"] for v in v_hw["voxels"]) / v_hw["voxel_count"]
        avg_no = sum(v["sst"] for v in v_no["voxels"]) / v_no["voxel_count"]
        assert avg_hw >= avg_no

    def test_voxel_count_matches(self):
        data = dt.get_ocean_volume(depth_indices=[0])
        assert data["voxel_count"] == len(data["voxels"])

    def test_voxel_rgb_range(self):
        data = dt.get_ocean_volume(depth_indices=[0])
        for v in data["voxels"][:20]:
            assert 0.0 <= v["r"] <= 1.0
            assert 0.0 <= v["g"] <= 1.0
            assert 0.0 <= v["b"] <= 1.0


# ══════════════════════════════════════════════════════════════
# 3. DEPTH LAYER
# ══════════════════════════════════════════════════════════════

class TestDepthLayer:
    def test_type(self):
        data = dt.get_depth_layer(0)
        assert data["type"] == "depth_layer"

    def test_depth_idx(self):
        for i in [0, 5, 14]:
            data = dt.get_depth_layer(i)
            assert data["depth_idx"] == i

    def test_depth_m_correct(self):
        data = dt.get_depth_layer(4)  # 50m
        assert data["depth_m"] == 50

    def test_cells_present(self):
        data = dt.get_depth_layer(0)
        assert "cells" in data
        assert len(data["cells"]) > 0

    def test_cell_fields(self):
        data = dt.get_depth_layer(0)
        c = data["cells"][0]
        for f in ["x","y","z","r","g","b","sst","lat","lon"]:
            assert f in c

    def test_clamps_idx(self):
        d_neg = dt.get_depth_layer(-1)
        d_over = dt.get_depth_layer(100)
        assert d_neg["depth_idx"] == 0
        assert d_over["depth_idx"] == len(dt.DEPTH_LEVELS_M) - 1

    def test_deeper_is_colder(self):
        surface = dt.get_depth_layer(0)
        deep = dt.get_depth_layer(13)
        avg_s = sum(c["sst"] for c in surface["cells"]) / len(surface["cells"])
        avg_d = sum(c["sst"] for c in deep["cells"]) / len(deep["cells"])
        assert avg_d < avg_s

    def test_cell_count_matches(self):
        data = dt.get_depth_layer(0)
        assert data["cell_count"] == len(data["cells"])


# ══════════════════════════════════════════════════════════════
# 4. THERMOCLINE
# ══════════════════════════════════════════════════════════════

class TestThermocline:
    def test_type(self):
        data = dt.get_thermocline_surface()
        assert data["type"] == "thermocline_surface"

    def test_points_present(self):
        data = dt.get_thermocline_surface()
        assert "points" in data
        assert len(data["points"]) > 0

    def test_point_fields(self):
        data = dt.get_thermocline_surface()
        p = data["points"][0]
        for f in ["x","y","z","thermocline_depth_m","sst_surface","r","g","b","lat","lon"]:
            assert f in p

    def test_depth_range(self):
        data = dt.get_thermocline_surface()
        for p in data["points"]:
            assert 20 <= p["thermocline_depth_m"] <= 120

    def test_avg_depth_present(self):
        data = dt.get_thermocline_surface()
        assert "avg_depth_m" in data
        assert 20 <= data["avg_depth_m"] <= 120

    def test_sst_range(self):
        data = dt.get_thermocline_surface()
        for p in data["points"][:20]:
            assert 4 <= p["sst_surface"] <= 36

    def test_xyz_on_sphere(self):
        data = dt.get_thermocline_surface()
        for p in data["points"][:10]:
            r = math.sqrt(p["x"]**2 + p["y"]**2 + p["z"]**2)
            assert 4.8 <= r <= 5.1


# ══════════════════════════════════════════════════════════════
# 5. PARTICLE DATA
# ══════════════════════════════════════════════════════════════

class TestParticleData:
    def test_current_type(self):
        data = dt.get_particle_data(50, "current")
        assert data["type"] == "particle_system"
        assert data["particle_type"] == "current"

    def test_wind_type(self):
        data = dt.get_particle_data(50, "wind")
        assert data["particle_type"] == "wind"

    def test_count_matches(self):
        data = dt.get_particle_data(100, "current")
        assert data["count"] == 100
        assert len(data["particles"]) == 100

    def test_particle_fields(self):
        data = dt.get_particle_data(10, "current")
        p = data["particles"][0]
        for f in ["x","y","z","vx","vy","vz","speed","age","max_age","r","g","b"]:
            assert f in p

    def test_particle_speed_positive(self):
        data = dt.get_particle_data(20, "current")
        for p in data["particles"]:
            assert p["speed"] >= 0

    def test_particle_age_range(self):
        data = dt.get_particle_data(20, "current")
        for p in data["particles"]:
            assert 0 <= p["age"] <= p["max_age"]

    def test_particle_rgb(self):
        data = dt.get_particle_data(20, "current")
        for p in data["particles"]:
            assert 0.0 <= p["r"] <= 1.0
            assert 0.0 <= p["g"] <= 1.0
            assert 0.0 <= p["b"] <= 1.0

    def test_deterministic(self):
        d1 = dt.get_particle_data(20, "current")
        d2 = dt.get_particle_data(20, "current")
        assert d1["particles"][0]["x"] == d2["particles"][0]["x"]

    def test_color_hint(self):
        c = dt.get_particle_data(5, "current")
        w = dt.get_particle_data(5, "wind")
        assert "#" in c["color_hint"]
        assert "#" in w["color_hint"]
        assert c["color_hint"] != w["color_hint"]


# ══════════════════════════════════════════════════════════════
# 6. ARGO ANIMATION
# ══════════════════════════════════════════════════════════════

class TestArgoAnimation:
    def test_type(self):
        data = dt.get_argo_animation(n_demo=5)
        assert data["type"] == "argo_animation"

    def test_float_count(self):
        data = dt.get_argo_animation(n_demo=10)
        assert data["float_count"] == 10

    def test_keyframes_present(self):
        data = dt.get_argo_animation(n_demo=3)
        for f in data["floats"]:
            assert "keyframes" in f
            assert len(f["keyframes"]) == 5  # cycle_t has 5 points

    def test_keyframe_fields(self):
        data = dt.get_argo_animation(n_demo=2)
        kf = data["floats"][0]["keyframes"][0]
        for field in ["t","x","y","z","depth_m","lat","lon"]:
            assert field in kf

    def test_keyframe_t_range(self):
        data = dt.get_argo_animation(n_demo=3)
        for f in data["floats"]:
            ts = [kf["t"] for kf in f["keyframes"]]
            assert ts[0] == 0.0
            assert ts[-1] == 1.0

    def test_depth_cycle(self):
        data = dt.get_argo_animation(n_demo=2)
        depths = [kf["depth_m"] for kf in data["floats"][0]["keyframes"]]
        assert depths[0] == 0     # surface
        assert max(depths) == 2000  # max depth
        assert depths[-1] == 0    # returns to surface

    def test_cycle_duration(self):
        data = dt.get_argo_animation(n_demo=2)
        assert data["cycle_duration_hours"] == 240

    def test_caps_at_50(self):
        floats = [{"float_id": f"F{i}", "latitude": 10.0, "longitude": 80.0} for i in range(100)]
        data = dt.get_argo_animation(floats=floats)
        assert data["float_count"] <= 50

    def test_surface_position_stored(self):
        data = dt.get_argo_animation(n_demo=3)
        for f in data["floats"]:
            assert "surface_lat" in f
            assert "surface_lon" in f


# ══════════════════════════════════════════════════════════════
# 7. SATELLITE ORBITS
# ══════════════════════════════════════════════════════════════

class TestSatelliteOrbits:
    def test_type(self):
        data = dt.get_satellite_orbits()
        assert data["type"] == "satellite_orbits"

    def test_count(self):
        data = dt.get_satellite_orbits()
        assert data["satellite_count"] == len(dt.SATELLITES)
        assert data["satellite_count"] >= 4

    def test_orbit_fields(self):
        data = dt.get_satellite_orbits()
        orb = data["orbits"][0]
        for f in ["id","altitude_km","inclination_deg","period_min","color","path"]:
            assert f in orb

    def test_path_length(self):
        data = dt.get_satellite_orbits(n_frames=30)
        for orb in data["orbits"]:
            assert len(orb["path"]) == 31  # n_frames + 1

    def test_path_point_fields(self):
        data = dt.get_satellite_orbits(n_frames=10)
        p = data["orbits"][0]["path"][0]
        for f in ["t","x","y","z"]:
            assert f in p

    def test_orbit_radius(self):
        data = dt.get_satellite_orbits()
        for orb in data["orbits"]:
            for p in orb["path"][:5]:
                r = math.sqrt(p["x"]**2 + p["y"]**2 + p["z"]**2)
                assert r > dt.EARTH_RADIUS  # above Earth surface

    def test_orbit_color_hex(self):
        data = dt.get_satellite_orbits()
        for orb in data["orbits"]:
            assert orb["color"].startswith("#")

    def test_t_range(self):
        data = dt.get_satellite_orbits(n_frames=10)
        for orb in data["orbits"]:
            ts = [p["t"] for p in orb["path"]]
            assert ts[0] == 0.0
            assert ts[-1] == 1.0


# ══════════════════════════════════════════════════════════════
# 8. MARINE OBJECTS
# ══════════════════════════════════════════════════════════════

class TestMarineObjects:
    def test_type(self):
        data = dt.get_marine_objects()
        assert data["type"] == "marine_objects"

    def test_total(self):
        data = dt.get_marine_objects()
        assert data["total"] >= 8

    def test_by_type(self):
        data = dt.get_marine_objects()
        by_t = data["by_type"]
        assert "ship" in by_t
        assert "buoy" in by_t
        assert "glider" in by_t

    def test_object_fields(self):
        data = dt.get_marine_objects()
        obj = data["objects"][0]
        for f in ["id","type","x","y","z","color"]:
            assert f in obj

    def test_objects_near_surface(self):
        data = dt.get_marine_objects()
        for obj in data["objects"]:
            if obj["type"] in ("ship","buoy"):
                r = math.sqrt(obj["x"]**2 + obj["y"]**2 + obj["z"]**2)
                assert r > dt.EARTH_RADIUS - 0.1  # near surface

    def test_gliders_submerged(self):
        data = dt.get_marine_objects()
        gliders = [o for o in data["objects"] if o["type"] == "glider"]
        assert len(gliders) >= 1
        for g in gliders:
            r = math.sqrt(g["x"]**2 + g["y"]**2 + g["z"]**2)
            assert r < dt.EARTH_RADIUS  # below surface

    def test_color_hex(self):
        data = dt.get_marine_objects()
        for obj in data["objects"]:
            assert obj["color"].startswith("#")

    def test_total_matches(self):
        data = dt.get_marine_objects()
        assert data["total"] == len(data["objects"])


# ══════════════════════════════════════════════════════════════
# 9. BATHYMETRY TERRAIN
# ══════════════════════════════════════════════════════════════

class TestBathymetryTerrain:
    def test_type(self):
        data = dt.get_bathymetry_terrain("low")
        assert data["type"] == "bathymetry_terrain"

    def test_resolutions(self):
        for res in ["low","medium","high"]:
            data = dt.get_bathymetry_terrain(res)
            assert data["resolution"] == res

    def test_point_count(self):
        low  = dt.get_bathymetry_terrain("low")
        high = dt.get_bathymetry_terrain("high")
        assert high["point_count"] > low["point_count"]

    def test_terrain_fields(self):
        data = dt.get_bathymetry_terrain("low")
        p = data["terrain"][0]
        for f in ["x","y","z","depth_m","feature_type","r","g","b","lat","lon"]:
            assert f in p

    def test_depths_negative(self):
        data = dt.get_bathymetry_terrain("low")
        assert data["deepest_m"] < 0
        assert data["deepest_m"] < data["shallowest_m"]

    def test_named_features(self):
        data = dt.get_bathymetry_terrain("low")
        assert "named_features" in data
        assert len(data["named_features"]) >= 5
        names = [f["name"] for f in data["named_features"]]
        assert any("Ridge" in n or "Trench" in n for n in names)

    def test_feature_xyz(self):
        data = dt.get_bathymetry_terrain("low")
        for f in data["named_features"]:
            for key in ["x","y","z"]:
                assert key in f

    def test_count_matches(self):
        data = dt.get_bathymetry_terrain("low")
        assert data["point_count"] == len(data["terrain"])


# ══════════════════════════════════════════════════════════════
# 10. HEATWAVE GLOW
# ══════════════════════════════════════════════════════════════

class TestHeatwaveGlow:
    def test_type_no_hotspots(self):
        data = dt.get_heatwave_glow([])
        assert data["type"] == "heatwave_glow"

    def test_empty_hotspots(self):
        data = dt.get_heatwave_glow([])
        assert data["point_count"] == 0
        assert data["hotspot_count"] == 0

    def test_with_hotspots(self):
        hs = [{"latitude": 13.0, "longitude": 80.0, "severity": "EXTREME"}]
        data = dt.get_heatwave_glow(hs)
        assert data["point_count"] > 0

    def test_hotspot_markers(self):
        hs = [{"latitude": 13.0, "longitude": 80.0, "severity": "HIGH"}]
        data = dt.get_heatwave_glow(hs)
        assert data["hotspot_count"] == 1
        mk = data["hotspot_markers"][0]
        for f in ["x","y","z","severity"]:
            assert f in mk

    def test_glow_point_fields(self):
        hs = [{"latitude": 13.0, "longitude": 80.0, "severity": "HIGH"}]
        data = dt.get_heatwave_glow(hs)
        if data["points"]:
            p = data["points"][0]
            for f in ["x","y","z","intensity","r","g","b"]:
                assert f in p

    def test_intensity_range(self):
        hs = [{"latitude": 13.0, "longitude": 80.0, "severity": "EXTREME"}]
        data = dt.get_heatwave_glow(hs)
        for p in data["points"]:
            assert 0.0 < p["intensity"] <= 1.0

    def test_extreme_redder(self):
        hs_ext = [{"latitude": 13.0, "longitude": 80.0, "severity": "EXTREME"}]
        hs_low = [{"latitude": 13.0, "longitude": 80.0, "severity": "MODERATE"}]
        d_ext = dt.get_heatwave_glow(hs_ext)
        d_low = dt.get_heatwave_glow(hs_low)
        if d_ext["points"] and d_low["points"]:
            max_ext = max(p["intensity"] for p in d_ext["points"])
            max_low = max(p["intensity"] for p in d_low["points"])
            assert max_ext >= max_low


# ══════════════════════════════════════════════════════════════
# 11. CORAL HOTSPOTS
# ══════════════════════════════════════════════════════════════

class TestCoralHotspots:
    def test_type(self):
        data = dt.get_coral_hotspots()
        assert data["type"] == "coral_hotspots"

    def test_site_count(self):
        data = dt.get_coral_hotspots()
        assert data["site_count"] >= 8
        assert data["site_count"] == len(data["sites"])

    def test_health_categories(self):
        data = dt.get_coral_hotspots()
        healths = {s["health"] for s in data["sites"]}
        assert "healthy" in healths or "bleached" in healths

    def test_site_fields(self):
        data = dt.get_coral_hotspots()
        s = data["sites"][0]
        for f in ["name","health","sst_stress","x","y","z","r","g","b","intensity"]:
            assert f in s

    def test_xyz_on_sphere(self):
        data = dt.get_coral_hotspots()
        for s in data["sites"]:
            r = math.sqrt(s["x"]**2 + s["y"]**2 + s["z"]**2)
            assert 4.9 <= r <= 5.1

    def test_bleached_redder(self):
        data = dt.get_coral_hotspots()
        bleached = [s for s in data["sites"] if s["health"] == "bleached"]
        healthy  = [s for s in data["sites"] if s["health"] == "healthy"]
        if bleached and healthy:
            avg_red_b = sum(s["r"] for s in bleached) / len(bleached)
            avg_red_h = sum(s["r"] for s in healthy) / len(healthy)
            assert avg_red_b > avg_red_h

    def test_bleached_count(self):
        data = dt.get_coral_hotspots()
        assert data["bleached"] >= 0
        assert data["bleached"] <= data["site_count"]

    def test_intensity_range(self):
        data = dt.get_coral_hotspots()
        for s in data["sites"]:
            assert 0.0 <= s["intensity"] <= 1.0


# ══════════════════════════════════════════════════════════════
# 12. TIMELINE
# ══════════════════════════════════════════════════════════════

class TestTimeline:
    def test_type(self):
        data = dt.get_timeline_frames(5)
        assert data["type"] == "timeline"

    def test_frame_count(self):
        data = dt.get_timeline_frames(5)
        # 5 historical + 3 projections
        assert data["frame_count"] == 8

    def test_frame_fields(self):
        data = dt.get_timeline_frames(3)
        f = data["frames"][0]
        for field in ["year","mean_sst","heatwave_count","anomaly_c","label"]:
            assert field in f

    def test_sst_increases(self):
        data = dt.get_timeline_frames(10)
        historical = [f for f in data["frames"] if not f.get("projected")]
        sst_vals = [f["mean_sst"] for f in historical]
        # Should be generally increasing
        assert sst_vals[-1] > sst_vals[0]

    def test_projected_frames(self):
        data = dt.get_timeline_frames(5)
        proj = [f for f in data["frames"] if f.get("projected")]
        assert len(proj) == 3

    def test_year_range(self):
        data = dt.get_timeline_frames(5)
        assert data["year_start"] == 2015
        assert data["year_end"] >= 2020

    def test_heatwave_count_nonneg(self):
        data = dt.get_timeline_frames(5)
        for f in data["frames"]:
            assert f["heatwave_count"] >= 0

    def test_label_contains_year(self):
        data = dt.get_timeline_frames(3)
        for f in data["frames"]:
            assert str(f["year"]) in f["label"]


# ══════════════════════════════════════════════════════════════
# 13. AI EXPLANATION HUD
# ══════════════════════════════════════════════════════════════

class TestAIExplanation:
    def test_type(self):
        data = dt.get_ai_explanation_hud(13.0, 80.0)
        assert data["type"] == "ai_explanation"

    def test_fields(self):
        data = dt.get_ai_explanation_hud(13.0, 80.0)
        for f in ["lat","lon","sst","climatology","anomaly_c","confidence_pct",
                  "heatwave_status","explanation_text","shap_values","model_version"]:
            assert f in data

    def test_sst_range(self):
        data = dt.get_ai_explanation_hud(13.0, 80.0)
        assert 4 <= data["sst"] <= 36

    def test_confidence_range(self):
        data = dt.get_ai_explanation_hud(13.0, 80.0)
        assert 50 <= data["confidence_pct"] <= 100

    def test_shap_keys(self):
        data = dt.get_ai_explanation_hud(13.0, 80.0)
        shap = data["shap_values"]
        assert len(shap) >= 4
        assert "sst_lag1" in shap

    def test_heatwave_detected(self):
        hs = [{"latitude": 13.0, "longitude": 80.0, "severity": "EXTREME"}]
        data = dt.get_ai_explanation_hud(13.0, 80.0, hs)
        assert data["heatwave_status"] == "EXTREME"
        assert "heatwave" in data["explanation_text"].lower()

    def test_no_heatwave(self):
        data = dt.get_ai_explanation_hud(13.0, 80.0, [])
        assert data["heatwave_status"] == "NONE"

    def test_model_version(self):
        data = dt.get_ai_explanation_hud(13.0, 80.0)
        assert "OceanTransformer" in data["model_version"]

    def test_different_coords(self):
        d1 = dt.get_ai_explanation_hud(5.0, 65.0)
        d2 = dt.get_ai_explanation_hud(20.0, 90.0)
        assert d1["sst"] != d2["sst"]  # different locations give different SSTs


# ══════════════════════════════════════════════════════════════
# 14. WAVE SURFACE
# ══════════════════════════════════════════════════════════════

class TestWaveSurface:
    def test_type(self):
        data = dt.get_wave_surface()
        assert data["type"] == "wave_surface"

    def test_wave_count(self):
        data = dt.get_wave_surface(8)
        assert data["wave_count"] == 8

    def test_wave_fields(self):
        data = dt.get_wave_surface(4)
        w = data["waves"][0]
        for f in ["amplitude","frequency","speed","dir_x","dir_z","phase"]:
            assert f in w

    def test_amplitude_positive(self):
        data = dt.get_wave_surface()
        for w in data["waves"]:
            assert w["amplitude"] > 0

    def test_frequency_positive(self):
        data = dt.get_wave_surface()
        for w in data["waves"]:
            assert w["frequency"] > 0

    def test_direction_unit(self):
        data = dt.get_wave_surface()
        for w in data["waves"]:
            mag = math.sqrt(w["dir_x"]**2 + w["dir_z"]**2)
            assert 0.9 <= mag <= 1.1  # approximately unit vector

    def test_glsl_snippet(self):
        data = dt.get_wave_surface()
        assert "glsl_snippet" in data


# ══════════════════════════════════════════════════════════════
# 15. SCENE DESCRIPTOR
# ══════════════════════════════════════════════════════════════

class TestSceneDescriptor:
    def test_type(self):
        data = dt.get_scene_descriptor()
        assert data["type"] == "scene_descriptor"

    def test_fields(self):
        data = dt.get_scene_descriptor()
        for f in ["version","earth_radius","depth_levels","depth_scale","region","layers","camera","vr","performance"]:
            assert f in data

    def test_earth_radius(self):
        data = dt.get_scene_descriptor()
        assert data["earth_radius"] == dt.EARTH_RADIUS

    def test_depth_levels_count(self):
        data = dt.get_scene_descriptor()
        assert len(data["depth_levels"]) == 15

    def test_camera_fly_sequence(self):
        data = dt.get_scene_descriptor()
        assert "fly_sequence" in data["camera"]
        assert len(data["camera"]["fly_sequence"]) >= 4

    def test_layers_defined(self):
        data = dt.get_scene_descriptor()
        layers = data["layers"]
        assert "ocean_volume" in layers
        assert "current_particles" in layers
        assert "heatwave_glow" in layers

    def test_vr_hooks(self):
        data = dt.get_scene_descriptor()
        assert "enabled" in data["vr"]

    def test_lod_performance(self):
        data = dt.get_scene_descriptor()
        perf = data["performance"]
        assert "lod_levels" in perf
        assert "target_fps" in perf
        assert perf["target_fps"] == 60


# ══════════════════════════════════════════════════════════════
# 16. VR & LOD DESCRIPTORS
# ══════════════════════════════════════════════════════════════

class TestVRDescriptor:
    def test_type(self):
        data = dt.get_vr_descriptor()
        assert data["type"] == "vr_descriptor"

    def test_fields(self):
        data = dt.get_vr_descriptor()
        for f in ["xr_session_mode","xr_features","controllers","scene_scale","notes"]:
            assert f in data

    def test_session_mode(self):
        data = dt.get_vr_descriptor()
        assert data["xr_session_mode"] == "immersive-vr"

    def test_controllers(self):
        data = dt.get_vr_descriptor()
        assert "left" in data["controllers"]
        assert "right" in data["controllers"]


class TestLODDescriptor:
    def test_type(self):
        data = dt.get_lod_descriptor()
        assert data["type"] == "lod_descriptor"

    def test_four_levels(self):
        data = dt.get_lod_descriptor()
        assert len(data["levels"]) == 4

    def test_level_fields(self):
        data = dt.get_lod_descriptor()
        for level in data["levels"]:
            for f in ["level","label","grid_res_deg","particles","camera_dist_max"]:
                assert f in level

    def test_lod_increasing_res(self):
        data = dt.get_lod_descriptor()
        levels = data["levels"]
        # Higher level = coarser resolution
        assert levels[0]["grid_res_deg"] <= levels[-1]["grid_res_deg"]

    def test_gpu_instancing(self):
        data = dt.get_lod_descriptor()
        assert data["gpu_instancing"] is True


# ══════════════════════════════════════════════════════════════
# 17. FLASK /twin PAGE
# ══════════════════════════════════════════════════════════════

class TestFlaskTwinPage:
    def test_twin_page_200(self, flask_client):
        r = flask_client.get("/twin")
        assert r.status_code == 200

    def test_twin_has_threejs(self, flask_client):
        r = flask_client.get("/twin")
        assert b"three" in r.data.lower() or b"THREE" in r.data

    def test_twin_has_canvas(self, flask_client):
        r = flask_client.get("/twin")
        assert b"twin-canvas" in r.data

    def test_twin_has_hud(self, flask_client):
        r = flask_client.get("/twin")
        assert b"hud" in r.data.lower() or b"HUD" in r.data


# ══════════════════════════════════════════════════════════════
# 18. FLASK /api/twin/* ENDPOINTS
# ══════════════════════════════════════════════════════════════

class TestFlaskTwinScene:
    def test_scene_200(self, flask_client):
        r = flask_client.get("/api/twin/scene")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "scene_descriptor"

    def test_scene_has_layers(self, flask_client):
        r = flask_client.get("/api/twin/scene")
        d = json.loads(r.data)
        assert "layers" in d
        assert "ocean_volume" in d["layers"]

    def test_scene_has_camera(self, flask_client):
        r = flask_client.get("/api/twin/scene")
        d = json.loads(r.data)
        assert "camera" in d
        assert "fly_sequence" in d["camera"]


class TestFlaskOceanVolume:
    def test_volume_200(self, flask_client):
        r = flask_client.get("/api/ocean-volume")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "ocean_volume"
        assert d["voxel_count"] > 0

    def test_volume_depth_indices(self, flask_client):
        r = flask_client.get("/api/ocean-volume?depth_indices=0,1")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["depth_levels"] == 2

    def test_volume_voxels_have_fields(self, flask_client):
        r = flask_client.get("/api/ocean-volume?depth_indices=0")
        d = json.loads(r.data)
        v = d["voxels"][0]
        for f in ["x","y","z","sst","r","g","b"]:
            assert f in v


class TestFlaskDepthLayer:
    def test_depth_layer_default(self, flask_client):
        r = flask_client.get("/api/depth-layer")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "depth_layer"
        assert d["depth_idx"] == 0
        assert d["depth_m"] == 0

    def test_depth_layer_index(self, flask_client):
        r = flask_client.get("/api/depth-layer?depth_idx=5")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["depth_idx"] == 5
        assert d["depth_m"] == dt.DEPTH_LEVELS_M[5]

    def test_depth_layer_cells(self, flask_client):
        r = flask_client.get("/api/depth-layer?depth_idx=0")
        d = json.loads(r.data)
        assert d["cell_count"] > 0


class TestFlaskThermocline:
    def test_thermocline_200(self, flask_client):
        r = flask_client.get("/api/thermocline")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "thermocline_surface"
        assert d["point_count"] > 0


class TestFlaskParticleData:
    def test_current_200(self, flask_client):
        r = flask_client.get("/api/particle-data?type=current&n=50")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["count"] == 50
        assert d["particle_type"] == "current"

    def test_wind_200(self, flask_client):
        r = flask_client.get("/api/particle-data?type=wind&n=50")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["particle_type"] == "wind"

    def test_n_caps_at_800(self, flask_client):
        r = flask_client.get("/api/particle-data?n=9999")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["count"] <= 800

    def test_n_floor_at_10(self, flask_client):
        r = flask_client.get("/api/particle-data?n=1")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["count"] >= 10


class TestFlaskArgoAnimation:
    def test_argo_anim_200(self, flask_client):
        r = flask_client.get("/api/argo-animation")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "argo_animation"
        assert d["float_count"] > 0

    def test_argo_keyframes(self, flask_client):
        r = flask_client.get("/api/argo-animation")
        d = json.loads(r.data)
        assert len(d["floats"][0]["keyframes"]) == 5


class TestFlaskSatelliteOrbits:
    def test_satellite_orbits_200(self, flask_client):
        r = flask_client.get("/api/satellite-orbits")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "satellite_orbits"
        assert d["satellite_count"] >= 4

    def test_satellite_n_frames(self, flask_client):
        r = flask_client.get("/api/satellite-orbits?n_frames=30")
        d = json.loads(r.data)
        assert len(d["orbits"][0]["path"]) == 31


class TestFlaskMarineObjects:
    def test_marine_objects_200(self, flask_client):
        r = flask_client.get("/api/marine-objects")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "marine_objects"
        assert d["total"] >= 8


class TestFlaskBathymetryTerrain:
    def test_bathymetry_terrain_200(self, flask_client):
        r = flask_client.get("/api/bathymetry-terrain?resolution=low")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "bathymetry_terrain"
        assert d["point_count"] > 0

    def test_bathymetry_named_features(self, flask_client):
        r = flask_client.get("/api/bathymetry-terrain?resolution=low")
        d = json.loads(r.data)
        assert len(d["named_features"]) >= 5


class TestFlaskHeatwaveGlow:
    def test_glow_200(self, flask_client):
        r = flask_client.get("/api/heatwave-glow")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "heatwave_glow"

    def test_glow_no_hotspots(self, flask_client):
        r = flask_client.get("/api/heatwave-glow")
        d = json.loads(r.data)
        assert d["point_count"] == 0  # state has empty hotspots


class TestFlaskCoralHotspots:
    def test_coral_200(self, flask_client):
        r = flask_client.get("/api/coral-hotspots")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "coral_hotspots"
        assert d["site_count"] >= 8


class TestFlaskWaveSurface:
    def test_wave_200(self, flask_client):
        r = flask_client.get("/api/wave-surface")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "wave_surface"
        assert d["wave_count"] == 8

    def test_wave_n_param(self, flask_client):
        r = flask_client.get("/api/wave-surface?n_waves=4")
        d = json.loads(r.data)
        assert d["wave_count"] == 4


class TestFlaskTwinTimeline:
    def test_timeline_200(self, flask_client):
        r = flask_client.get("/api/twin/timeline")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "timeline"
        assert d["frame_count"] > 0

    def test_timeline_n_years(self, flask_client):
        r = flask_client.get("/api/twin/timeline?n_years=5")
        d = json.loads(r.data)
        assert d["frame_count"] == 8  # 5 + 3 projected


class TestFlaskTwinExplain:
    def test_explain_200(self, flask_client):
        r = flask_client.get("/api/twin/explain?lat=13.0&lon=80.0")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "ai_explanation"

    def test_explain_lat_lon(self, flask_client):
        r = flask_client.get("/api/twin/explain?lat=15.0&lon=70.0")
        d = json.loads(r.data)
        assert abs(d["lat"] - 15.0) < 0.01
        assert abs(d["lon"] - 70.0) < 0.01

    def test_explain_shap(self, flask_client):
        r = flask_client.get("/api/twin/explain?lat=13.0&lon=80.0")
        d = json.loads(r.data)
        assert "shap_values" in d
        assert len(d["shap_values"]) >= 4


class TestFlaskTwinVR:
    def test_vr_200(self, flask_client):
        r = flask_client.get("/api/twin/vr")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "vr_descriptor"
        assert d["xr_session_mode"] == "immersive-vr"


class TestFlaskTwinLOD:
    def test_lod_200(self, flask_client):
        r = flask_client.get("/api/twin/lod")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "lod_descriptor"
        assert len(d["levels"]) == 4


class TestFlaskTwinStats:
    def test_stats_200(self, flask_client):
        r = flask_client.get("/api/twin/stats")
        assert r.status_code == 200
        d = json.loads(r.data)
        for key in ["depth_levels","satellites","coral_sites","marine_objects","region"]:
            assert key in d

    def test_stats_depth_levels(self, flask_client):
        r = flask_client.get("/api/twin/stats")
        d = json.loads(r.data)
        assert d["depth_levels"] == 15

    def test_stats_satellites(self, flask_client):
        r = flask_client.get("/api/twin/stats")
        d = json.loads(r.data)
        assert d["satellites"] >= 4


# ══════════════════════════════════════════════════════════════
# 19. ALL PREVIOUS ROUTES STILL INTACT
# ══════════════════════════════════════════════════════════════

class TestAllPreviousRoutes:
    PUBLIC_PAGES = ["/","/earth","/dashboard","/heatwave","/voice",
                    "/about","/gismap","/map","/twin","/login"]
    USER_PAGES = ["/settings","/profile","/downloads","/activity","/reports"]
    ADMIN_PAGES = ["/admin","/database","/training","/api-management","/model-registry"]

    def test_public_pages_200(self, flask_client):
        for path in self.PUBLIC_PAGES:
            r = flask_client.get(path)
            assert r.status_code == 200, f"{path} → {r.status_code}"

    def test_user_pages_200_when_authenticated(self, flask_client_user):
        for path in self.USER_PAGES:
            r = flask_client_user.get(path)
            assert r.status_code == 200, f"{path} → {r.status_code}"

    def test_admin_pages_200_for_admin(self, flask_client_admin):
        for path in self.ADMIN_PAGES:
            r = flask_client_admin.get(path)
            assert r.status_code == 200, f"{path} → {r.status_code}"

    def test_protected_pages_redirect_when_anonymous(self, flask_client):
        for path in self.USER_PAGES + self.ADMIN_PAGES:
            r = flask_client.get(path)
            assert r.status_code == 302, f"{path} should redirect anonymous users, got {r.status_code}"

    def test_phase2_apis_intact(self, flask_client):
        for ep in ["/api/map/layers","/api/map/ports","/api/map/routes",
                   "/api/map/areas","/api/map/argo","/api/map/ships",
                   "/api/map/stats","/api/map/bathymetry"]:
            r = flask_client.get(ep)
            assert r.status_code == 200, f"{ep} → {r.status_code}"

    def test_phase1_apis_intact(self, flask_client_user):
        for ep in ["/api/settings","/api/favorites","/api/bookmarks",
                   "/api/registry/datasets","/api/registry/models",
                   "/api/platform/stats","/api/db/health"]:
            r = flask_client_user.get(ep)
            assert r.status_code == 200, f"{ep} → {r.status_code}"

    def test_existing_apis_intact(self, flask_client):
        for ep in ["/api/health","/api/status","/api/heatwave",
                   "/api/datasets","/api/reports","/api/summary"]:
            r = flask_client.get(ep)
            assert r.status_code == 200, f"{ep} → {r.status_code}"
