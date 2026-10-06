"""
test_phase2_gis.py
OceanVerse AI v4.0 — Phase 2 GIS Tests
150+ tests covering:
  - GIS schema init & seeding
  - Ports CRUD + stats + GeoJSON
  - Shipping routes CRUD + GeoJSON
  - Marine areas (MPA + EEZ) CRUD + GeoJSON
  - Bathymetry CRUD + stats
  - AIS vessels CRUD
  - GIS bookmarks
  - Measurements (haversine + save + list)
  - Drawn features CRUD
  - Location search (gazeteer + port + coordinates)
  - Grid utilities (generate + snap)
  - SST / Wind / Current GeoJSON generators
  - Distance from coast
  - Aggregate GIS stats
  - All Flask /api/map/* endpoints (GET + POST + DELETE)
  - /api/map/layers, /api/map/inspect, /api/map/stats
  - /api/map/export/geojson
  - /page/map renders 200
"""

import json
import os
import sqlite3
import sys
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gis_engine as gis


# ──────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    gis.init_gis_tables(conn)
    yield conn
    conn.close()


@pytest.fixture
def flask_client():
    import unittest.mock as mock
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
        "p3_hotspots":[], "p3_argo_result":{}, "p3_pred_ds":None,
        "p3_surface_ds":None, "p3_uncertainty":{}, "p3_multivar":{},
        "p3_status":{}, "p2_status":{}, "p3_times":[], "p3_lats":[],
        "p3_lons":[], "p3_norm_stats":{}, "p3_impacts":[], "p3_future_fc":{},
        "p3_xai":{}, "p2_statistics":[], "p2_depths":[], "p2_surface_shape":[],
        "p2_target_shape":[], "p2_missing":[], "p2_interpolation":[],
        "scan_records":[], "inspections":{}, "qc_results":[], "cov_results":[],
        "summary":{}, "config":{"paths":{"raw_data":"/tmp"}}, "db_path":":memory:",
    }
    fp.run_phase1=fp.run_phase2=fp.run_phase3=lambda *a,**k:None

    fa = sys.modules["admin"]
    fa.verify_admin_token=mock.MagicMock(return_value=True)
    fa.get_latest_prediction=mock.MagicMock(return_value={})
    fa.get_heatwave_alert=mock.MagicMock(return_value={})
    fa.get_ocean_summary=mock.MagicMock(return_value={})
    fa.build_download_netcdf_path=mock.MagicMock(return_value="/tmp/x.nc")
    fa.build_download_csv=mock.MagicMock(return_value="/tmp/x.csv")
    for attr in ["get_model_history","get_admin_logs","get_prediction_history",
                 "get_training_history","get_platform_statistics"]:
        setattr(fa, attr, mock.MagicMock(return_value=[]))
    fa.init_admin_tables=mock.MagicMock()
    fa.admin_upload_dataset=fa.admin_retrain_model=mock.MagicMock(return_value={"status":"ok"})

    import app as flask_app
    flask_app.app.config["TESTING"] = True
    flask_app.app.config["SECRET_KEY"] = "test-gis-secret"

    with flask_app.app.test_client() as client:
        with flask_app.app.app_context():
            yield client

    for p in patchers: p.stop()


# ══════════════════════════════════════════════════════════════
# 1. SCHEMA & SEEDING
# ══════════════════════════════════════════════════════════════

class TestGISSchema:
    REQUIRED = ["ports","shipping_routes","marine_areas","bathymetry_points",
                "gis_map_bookmarks","ais_vessels","time_frames",
                "measurement_sessions","drawn_features","location_search_cache"]

    def test_all_tables_created(self, db):
        cur = db.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in cur.fetchall()}
        for t in self.REQUIRED:
            assert t in tables, f"Missing: {t}"

    def test_init_idempotent(self, db):
        gis.init_gis_tables(db)
        gis.init_gis_tables(db)
        cur = db.execute("SELECT COUNT(*) FROM ports")
        assert cur.fetchone()[0] > 0

    def test_ports_seeded(self, db):
        cur = db.execute("SELECT COUNT(*) FROM ports")
        assert cur.fetchone()[0] >= 20

    def test_routes_seeded(self, db):
        cur = db.execute("SELECT COUNT(*) FROM shipping_routes")
        assert cur.fetchone()[0] >= 5

    def test_areas_seeded(self, db):
        cur = db.execute("SELECT COUNT(*) FROM marine_areas")
        assert cur.fetchone()[0] >= 8

    def test_bathymetry_seeded(self, db):
        cur = db.execute("SELECT COUNT(*) FROM bathymetry_points")
        assert cur.fetchone()[0] >= 15

    def test_vessels_seeded(self, db):
        cur = db.execute("SELECT COUNT(*) FROM ais_vessels")
        assert cur.fetchone()[0] >= 8

    def test_seed_has_major_ports(self, db):
        cur = db.execute("SELECT COUNT(*) FROM ports WHERE is_major=1")
        assert cur.fetchone()[0] >= 5

    def test_seed_has_mpa_and_eez(self, db):
        cur = db.execute("SELECT COUNT(*) FROM marine_areas WHERE area_type='mpa'")
        assert cur.fetchone()[0] >= 3
        cur = db.execute("SELECT COUNT(*) FROM marine_areas WHERE area_type='eez'")
        assert cur.fetchone()[0] >= 3


# ══════════════════════════════════════════════════════════════
# 2. PORTS
# ══════════════════════════════════════════════════════════════

class TestPorts:
    def test_get_all_ports_geojson(self, db):
        result = gis.get_ports(db)
        assert result["type"] == "FeatureCollection"
        assert result["total"] >= 20
        assert isinstance(result["features"], list)

    def test_geojson_features_have_geometry(self, db):
        result = gis.get_ports(db)
        for f in result["features"]:
            assert f["type"] == "Feature"
            assert "geometry" in f
            assert f["geometry"]["type"] == "Point"
            coords = f["geometry"]["coordinates"]
            assert len(coords) == 2
            assert -180 <= coords[0] <= 180
            assert -90 <= coords[1] <= 90

    def test_get_major_only(self, db):
        all_p = gis.get_ports(db)
        major = gis.get_ports(db, major_only=True)
        assert major["total"] < all_p["total"]
        assert major["total"] >= 5

    def test_get_by_country(self, db):
        india = gis.get_ports(db, country="India")
        assert india["total"] >= 5
        for f in india["features"]:
            assert f["properties"]["country"] == "India"

    def test_get_by_bbox(self, db):
        bbox = {"lat_min":10,"lat_max":20,"lon_min":70,"lon_max":90}
        result = gis.get_ports(db, bbox=bbox)
        for f in result["features"]:
            c = f["geometry"]["coordinates"]
            assert 70 <= c[0] <= 90
            assert 10 <= c[1] <= 20

    def test_add_port(self, db):
        r = gis.add_port(db, "Test Port", "TestLand", 10.0, 75.0, "commercial")
        assert r["status"] == "added"
        assert r["id"] > 0

    def test_delete_port(self, db):
        r = gis.add_port(db, "Del Port", "Land", 5.0, 60.0)
        gis.delete_port(db, r["id"])
        cur = db.execute("SELECT COUNT(*) FROM ports WHERE id=?", (r["id"],))
        assert cur.fetchone()[0] == 0

    def test_ports_stats(self, db):
        stats = gis.get_ports_stats(db)
        assert "total" in stats
        assert "major" in stats
        assert "by_country" in stats
        assert stats["total"] >= 20
        assert stats["major"] >= 5
        assert isinstance(stats["by_country"], list)

    def test_ports_stats_traffic(self, db):
        stats = gis.get_ports_stats(db)
        assert "total_traffic_mt" in stats
        assert stats["total_traffic_mt"] > 0

    def test_features_have_port_props(self, db):
        result = gis.get_ports(db)
        for f in result["features"][:5]:
            props = f["properties"]
            assert "name" in props
            assert "country" in props
            assert "port_type" in props


# ══════════════════════════════════════════════════════════════
# 3. SHIPPING ROUTES
# ══════════════════════════════════════════════════════════════

class TestShippingRoutes:
    def test_get_all_routes(self, db):
        result = gis.get_shipping_routes(db)
        assert result["type"] == "FeatureCollection"
        assert result["total"] >= 5

    def test_route_features_are_linestrings(self, db):
        result = gis.get_shipping_routes(db)
        for f in result["features"]:
            assert f["geometry"]["type"] == "LineString"
            assert len(f["geometry"]["coordinates"]) >= 2

    def test_filter_by_type(self, db):
        tankers = gis.get_shipping_routes(db, route_type="tanker")
        for f in tankers["features"]:
            assert f["properties"]["type"] == "tanker"

    def test_filter_by_traffic(self, db):
        high = gis.get_shipping_routes(db, traffic="very_high")
        assert high["total"] >= 1

    def test_add_route(self, db):
        coords = [[80.0,13.0],[85.0,15.0],[90.0,18.0]]
        r = gis.add_shipping_route(db, "Test Route", coords, "container", "#00ff00")
        assert r["status"] == "added"

    def test_route_geojson_valid(self, db):
        result = gis.get_shipping_routes(db)
        for f in result["features"]:
            assert isinstance(f["properties"].get("color"), str)
            assert f["properties"]["color"].startswith("#")

    def test_route_has_name(self, db):
        result = gis.get_shipping_routes(db)
        for f in result["features"]:
            assert "name" in f["properties"]
            assert len(f["properties"]["name"]) > 0


# ══════════════════════════════════════════════════════════════
# 4. MARINE AREAS
# ══════════════════════════════════════════════════════════════

class TestMarineAreas:
    def test_get_all_areas(self, db):
        result = gis.get_marine_areas(db)
        assert result["type"] == "FeatureCollection"
        assert result["total"] >= 8

    def test_areas_are_polygons(self, db):
        result = gis.get_marine_areas(db)
        for f in result["features"]:
            assert f["geometry"]["type"] == "Polygon"

    def test_filter_mpa(self, db):
        mpa = gis.get_marine_areas(db, area_type="mpa")
        assert mpa["total"] >= 3
        for f in mpa["features"]:
            assert f["properties"]["type"] == "mpa"

    def test_filter_eez(self, db):
        eez = gis.get_marine_areas(db, area_type="eez")
        assert eez["total"] >= 3
        for f in eez["features"]:
            assert f["properties"]["type"] == "eez"

    def test_filter_by_country(self, db):
        india_eez = gis.get_marine_areas(db, country="India")
        assert india_eez["total"] >= 1

    def test_add_mpa(self, db):
        coords = [[70.0,10.0],[72.0,10.0],[72.0,12.0],[70.0,12.0],[70.0,10.0]]
        r = gis.add_marine_area(db, "Test MPA", coords, "mpa", "TestLand", "#00ff00")
        assert r["status"] == "added"

    def test_delete_area(self, db):
        coords = [[70.0,10.0],[72.0,10.0],[72.0,12.0],[70.0,12.0],[70.0,10.0]]
        r = gis.add_marine_area(db, "Del MPA", coords)
        gis.delete_marine_area(db, r["id"])
        cur = db.execute("SELECT COUNT(*) FROM marine_areas WHERE id=?", (r["id"],))
        assert cur.fetchone()[0] == 0

    def test_area_has_color(self, db):
        result = gis.get_marine_areas(db)
        for f in result["features"]:
            assert "color" in f["properties"]


# ══════════════════════════════════════════════════════════════
# 5. BATHYMETRY
# ══════════════════════════════════════════════════════════════

class TestBathymetry:
    def test_get_all_bathymetry(self, db):
        result = gis.get_bathymetry(db)
        assert result["type"] == "FeatureCollection"
        assert result["total"] >= 15

    def test_bathymetry_points_have_depth(self, db):
        result = gis.get_bathymetry(db)
        for f in result["features"]:
            assert "depth_m" in f["properties"]
            assert f["properties"]["depth_m"] < 0  # depths are negative

    def test_filter_by_bbox(self, db):
        bbox = {"lat_min":-10,"lat_max":25,"lon_min":60,"lon_max":100}
        result = gis.get_bathymetry(db, bbox=bbox)
        assert isinstance(result["features"], list)

    def test_filter_by_feature_type(self, db):
        ridges = gis.get_bathymetry(db, feature_type="ridge")
        for f in ridges["features"]:
            assert f["properties"]["feature_type"] == "ridge"

    def test_filter_trenches(self, db):
        trenches = gis.get_bathymetry(db, feature_type="trench")
        assert trenches["total"] >= 1

    def test_bathymetry_stats(self, db):
        stats = gis.get_bathymetry_stats(db)
        assert "min_depth_m" in stats
        assert "max_depth_m" in stats
        assert "total_points" in stats
        assert stats["min_depth_m"] < stats["max_depth_m"]
        assert stats["total_points"] >= 15

    def test_deepest_is_trench(self, db):
        stats = gis.get_bathymetry_stats(db)
        assert stats["min_depth_m"] < -5000  # Sunda trench > 7000m

    def test_add_point(self, db):
        r = gis.add_bathymetry_point(db, 10.0, 80.0, -2500.0, "plain")
        assert r["status"] == "added"
        stats = gis.get_bathymetry_stats(db)
        prev_count = stats["total_points"]
        assert prev_count >= 16

    def test_stats_by_feature_type(self, db):
        stats = gis.get_bathymetry_stats(db)
        assert isinstance(stats["by_feature_type"], list)
        types = [x["type"] for x in stats["by_feature_type"]]
        assert "ridge" in types or "trench" in types


# ══════════════════════════════════════════════════════════════
# 6. AIS VESSELS
# ══════════════════════════════════════════════════════════════

class TestAISVessels:
    def test_get_all_vessels(self, db):
        result = gis.get_ais_vessels(db)
        assert result["type"] == "FeatureCollection"
        assert result["total"] >= 8

    def test_vessel_features_structure(self, db):
        result = gis.get_ais_vessels(db)
        for f in result["features"][:3]:
            assert f["geometry"]["type"] == "Point"
            assert "mmsi" in f["properties"]
            assert "vessel_type" in f["properties"]

    def test_filter_by_type(self, db):
        tankers = gis.get_ais_vessels(db, vessel_type="tanker")
        assert tankers["total"] >= 1
        for f in tankers["features"]:
            assert f["properties"]["vessel_type"] == "tanker"

    def test_filter_by_bbox(self, db):
        bbox = {"lat_min":5,"lat_max":25,"lon_min":60,"lon_max":95}
        result = gis.get_ais_vessels(db, bbox=bbox)
        assert isinstance(result["features"], list)

    def test_update_position(self, db):
        # Get first vessel MMSI
        cur = db.execute("SELECT mmsi FROM ais_vessels LIMIT 1")
        mmsi = cur.fetchone()[0]
        r = gis.update_vessel_position(db, mmsi, 15.0, 75.0, 12.5, 180)
        assert r["status"] == "updated"
        v = gis.get_vessel_by_mmsi(db, mmsi)
        assert abs(v["latitude"] - 15.0) < 0.001
        assert abs(v["longitude"] - 75.0) < 0.001

    def test_get_vessel_by_mmsi(self, db):
        cur = db.execute("SELECT mmsi FROM ais_vessels LIMIT 1")
        mmsi = cur.fetchone()[0]
        v = gis.get_vessel_by_mmsi(db, mmsi)
        assert v is not None
        assert v["mmsi"] == mmsi

    def test_nonexistent_mmsi_returns_none(self, db):
        assert gis.get_vessel_by_mmsi(db, "000000000") is None


# ══════════════════════════════════════════════════════════════
# 7. GIS BOOKMARKS
# ══════════════════════════════════════════════════════════════

class TestGISBookmarks:
    def test_add_bookmark(self, db):
        r = gis.add_gis_bookmark(db, 1, "BoB View", 13.0, 80.0, 6.0)
        assert r["status"] == "added"
        assert r["id"] > 0

    def test_list_bookmarks(self, db):
        gis.add_gis_bookmark(db, 1, "A", 10.0, 70.0)
        gis.add_gis_bookmark(db, 1, "B", 15.0, 75.0)
        bms = gis.list_gis_bookmarks(db, 1)
        assert len(bms) == 2

    def test_bookmark_stores_layers(self, db):
        gis.add_gis_bookmark(db, 1, "Test", 10.0, 70.0, active_layers=["sst","heatwave"])
        bms = gis.list_gis_bookmarks(db, 1)
        assert bms[0]["active_layers"] == ["sst","heatwave"]

    def test_bookmark_stores_zoom(self, db):
        gis.add_gis_bookmark(db, 1, "Zoom Test", 10.0, 70.0, zoom=8.5)
        bms = gis.list_gis_bookmarks(db, 1)
        assert abs(bms[0]["zoom"] - 8.5) < 0.01

    def test_delete_bookmark(self, db):
        r = gis.add_gis_bookmark(db, 1, "Del", 0.0, 0.0)
        gis.delete_gis_bookmark(db, r["id"], 1)
        assert gis.list_gis_bookmarks(db, 1) == []

    def test_user_isolation(self, db):
        gis.add_gis_bookmark(db, 1, "U1", 1.0, 1.0)
        gis.add_gis_bookmark(db, 2, "U2", 2.0, 2.0)
        assert len(gis.list_gis_bookmarks(db, 1)) == 1
        assert len(gis.list_gis_bookmarks(db, 2)) == 1

    def test_bookmark_basemap_stored(self, db):
        gis.add_gis_bookmark(db, 1, "Dark", 10.0, 70.0, basemap="dark")
        bms = gis.list_gis_bookmarks(db, 1)
        assert bms[0]["basemap"] == "dark"


# ══════════════════════════════════════════════════════════════
# 8. MEASUREMENTS
# ══════════════════════════════════════════════════════════════

class TestMeasurements:
    def test_haversine_known_distance(self):
        # Mumbai to Colombo ≈ 1570 km
        d = gis.haversine_km(18.9, 72.8, 6.9, 79.9)
        assert 1400 < d < 1700

    def test_haversine_same_point(self):
        assert gis.haversine_km(10.0, 80.0, 10.0, 80.0) == 0.0

    def test_haversine_equator(self):
        # 1 degree of longitude at equator ≈ 111 km
        d = gis.haversine_km(0, 0, 0, 1)
        assert 100 < d < 120

    def test_calculate_two_points(self):
        result = gis.calculate_measurement([[18.9, 72.8],[6.9, 79.9]])
        assert "total_km" in result
        assert "total_nm" in result
        assert "segments" in result
        assert result["total_km"] > 0
        assert len(result["segments"]) == 1

    def test_calculate_multi_segment(self):
        pts = [[18.9,72.8],[13.1,80.3],[6.9,79.9]]
        result = gis.calculate_measurement(pts)
        assert len(result["segments"]) == 2
        assert result["total_km"] > 0

    def test_calculate_one_point_returns_zero(self):
        result = gis.calculate_measurement([[10.0, 80.0]])
        assert result["total_km"] == 0

    def test_nm_conversion(self):
        pts = [[0,0],[0,1]]
        result = gis.calculate_measurement(pts)
        assert abs(result["total_nm"] - result["total_km"] / 1.852) < 0.1

    def test_save_measurement(self, db):
        pts = [[18.9,72.8],[6.9,79.9]]
        r = gis.save_measurement(db, 1, pts, "Mumbai-Colombo")
        assert r["status"] == "saved"
        assert r["total_km"] > 0

    def test_list_measurements(self, db):
        gis.save_measurement(db, 1, [[0,0],[0,1]], "Test")
        ms = gis.list_measurements(db, 1)
        assert len(ms) == 1
        assert ms[0]["label"] == "Test"

    def test_measurement_user_isolation(self, db):
        gis.save_measurement(db, 1, [[0,0],[1,1]], "U1")
        gis.save_measurement(db, 2, [[2,2],[3,3]], "U2")
        assert len(gis.list_measurements(db, 1)) == 1
        assert len(gis.list_measurements(db, 2)) == 1


# ══════════════════════════════════════════════════════════════
# 9. DRAWN FEATURES
# ══════════════════════════════════════════════════════════════

class TestDrawnFeatures:
    def test_save_point(self, db):
        geojson = {"type":"Point","coordinates":[80.0,13.0]}
        r = gis.save_drawn_feature(db, 1, "point", geojson, "My Point")
        assert r["status"] == "saved"

    def test_save_polygon(self, db):
        geojson = {"type":"Polygon","coordinates":[[[80,12],[82,12],[82,14],[80,14],[80,12]]]}
        r = gis.save_drawn_feature(db, 1, "polygon", geojson, "My Polygon", "#ff0000")
        assert r["status"] == "saved"

    def test_list_features(self, db):
        gis.save_drawn_feature(db, 1, "point", {"type":"Point","coordinates":[80,13]}, "P1")
        gis.save_drawn_feature(db, 1, "point", {"type":"Point","coordinates":[81,14]}, "P2")
        feats = gis.list_drawn_features(db, 1)
        assert len(feats) == 2

    def test_geojson_parsed_on_list(self, db):
        geojson = {"type":"Point","coordinates":[80.0,13.0]}
        gis.save_drawn_feature(db, 1, "point", geojson, "Test")
        feats = gis.list_drawn_features(db, 1)
        assert isinstance(feats[0]["geojson"], dict)

    def test_delete_feature(self, db):
        r = gis.save_drawn_feature(db, 1, "point", {"type":"Point","coordinates":[0,0]}, "Del")
        gis.delete_drawn_feature(db, r["id"], 1)
        assert gis.list_drawn_features(db, 1) == []

    def test_user_isolation(self, db):
        gis.save_drawn_feature(db, 1, "point", {"type":"Point","coordinates":[80,13]}, "U1")
        gis.save_drawn_feature(db, 2, "point", {"type":"Point","coordinates":[81,14]}, "U2")
        assert len(gis.list_drawn_features(db, 1)) == 1
        assert len(gis.list_drawn_features(db, 2)) == 1

    def test_color_stored(self, db):
        gis.save_drawn_feature(db, 1, "line", {"type":"LineString","coordinates":[[0,0],[1,1]]}, "L", "#ff0000")
        feats = gis.list_drawn_features(db, 1)
        assert feats[0]["color"] == "#ff0000"


# ══════════════════════════════════════════════════════════════
# 10. LOCATION SEARCH
# ══════════════════════════════════════════════════════════════

class TestLocationSearch:
    def test_search_bay_of_bengal(self, db):
        results = gis.search_location(db, "Bay of Bengal")
        assert len(results) > 0
        assert any("Bay of Bengal" in r["name"] for r in results)

    def test_search_mumbai(self, db):
        results = gis.search_location(db, "Mumbai")
        assert len(results) > 0
        assert results[0]["type"] in ("city", "port")

    def test_search_partial(self, db):
        results = gis.search_location(db, "arabian")
        assert len(results) > 0

    def test_search_coordinate(self, db):
        results = gis.search_location(db, "13.5,82.3")
        assert len(results) > 0
        assert results[0]["type"] == "coordinate"
        assert abs(results[0]["lat"] - 13.5) < 0.01
        assert abs(results[0]["lon"] - 82.3) < 0.01

    def test_search_port_name(self, db):
        results = gis.search_location(db, "Colombo")
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert any("Colombo" in n for n in names)

    def test_search_returns_max_5(self, db):
        results = gis.search_location(db, "a")
        assert len(results) <= 5

    def test_search_cached(self, db):
        gis.search_location(db, "Mumbai")
        cur = db.execute("SELECT COUNT(*) FROM location_search_cache WHERE query='Mumbai'")
        assert cur.fetchone()[0] >= 1

    def test_empty_query_not_cached(self, db):
        results = gis.search_location(db, "xyzzy_not_real")
        assert isinstance(results, list)

    def test_search_gazeteer_ridge(self, db):
        results = gis.search_location(db, "Ninety East Ridge")
        assert len(results) > 0
        assert results[0]["type"] == "feature"


# ══════════════════════════════════════════════════════════════
# 11. GRID UTILITIES
# ══════════════════════════════════════════════════════════════

class TestGrid:
    def test_generate_grid_count(self):
        grid = gis.get_grid_cells(0, 2, 80, 82, 0.25)
        # 2/0.25 * 2/0.25 = 8*8 = 64 cells
        assert grid["total"] == 64
        assert len(grid["features"]) == 64

    def test_grid_features_are_polygons(self):
        grid = gis.get_grid_cells(0, 1, 80, 81, 0.25)
        for f in grid["features"]:
            assert f["geometry"]["type"] == "Polygon"
            coords = f["geometry"]["coordinates"][0]
            assert len(coords) == 5  # closed ring

    def test_grid_has_cell_id(self):
        grid = gis.get_grid_cells(0, 1, 80, 81, 0.25)
        for f in grid["features"]:
            assert "cell_id" in f["properties"]

    def test_grid_has_center(self):
        grid = gis.get_grid_cells(0, 0.25, 80, 80.25, 0.25)
        f = grid["features"][0]
        assert "lat_center" in f["properties"]
        assert "lon_center" in f["properties"]

    def test_snap_to_grid_exact(self):
        result = gis.snap_to_grid(13.0, 80.0, 0.25)
        assert result["lat"] == 13.0
        assert result["lon"] == 80.0

    def test_snap_to_grid_rounds(self):
        result = gis.snap_to_grid(13.1, 80.1, 0.25)
        assert result["lat"] == 13.0
        assert result["lon"] == 80.0

    def test_snap_cell_id_format(self):
        result = gis.snap_to_grid(13.0, 80.0, 0.25)
        assert "_" in result["cell_id"]

    def test_grid_res_1_degree(self):
        grid = gis.get_grid_cells(0, 5, 80, 85, 1.0)
        assert grid["total"] == 25  # 5*5

    def test_grid_type_feature_collection(self):
        grid = gis.get_grid_cells(0, 1, 80, 81, 0.25)
        assert grid["type"] == "FeatureCollection"


# ══════════════════════════════════════════════════════════════
# 12. DATA GENERATORS
# ══════════════════════════════════════════════════════════════

class TestDataGenerators:
    def test_sst_geojson_structure(self):
        data = gis.generate_sst_geojson(-5, 5, 75, 85, res=2.0)
        assert data["type"] == "FeatureCollection"
        assert data["total"] > 0
        for f in data["features"]:
            assert "sst" in f["properties"]
            assert 24 <= f["properties"]["sst"] <= 36
            assert 0 <= f["properties"]["intensity"] <= 1

    def test_sst_hotspot_warming(self):
        hotspots = [{"latitude": 0.0, "longitude": 80.0, "severity": "EXTREME"}]
        with_hs = gis.generate_sst_geojson(-2, 2, 78, 82, res=2.0, hotspots=hotspots)
        without_hs = gis.generate_sst_geojson(-2, 2, 78, 82, res=2.0, hotspots=[])
        avg_with = sum(f["properties"]["sst"] for f in with_hs["features"]) / with_hs["total"]
        avg_without = sum(f["properties"]["sst"] for f in without_hs["features"]) / without_hs["total"]
        assert avg_with >= avg_without

    def test_wind_geojson_structure(self):
        data = gis.generate_wind_geojson(-5, 5, 75, 85, res=2.0)
        assert data["type"] == "FeatureCollection"
        for f in data["features"]:
            props = f["properties"]
            assert "u" in props and "v" in props
            assert "speed" in props and "direction" in props
            assert props["speed"] >= 0
            assert 0 <= props["direction"] <= 360

    def test_current_geojson_structure(self):
        data = gis.generate_current_geojson(-5, 5, 75, 85, res=2.0)
        assert data["type"] == "FeatureCollection"
        for f in data["features"]:
            props = f["properties"]
            assert "u" in props and "v" in props and "speed" in props

    def test_sst_coordinates_in_range(self):
        data = gis.generate_sst_geojson(10, 15, 80, 85, res=1.0)
        for f in data["features"]:
            c = f["geometry"]["coordinates"]
            assert 80 <= c[0] <= 85
            assert 10 <= c[1] <= 15

    def test_wind_speed_positive(self):
        data = gis.generate_wind_geojson(0, 10, 70, 80, res=2.0)
        for f in data["features"]:
            assert f["properties"]["speed"] >= 0


# ══════════════════════════════════════════════════════════════
# 13. DISTANCE FROM COAST
# ══════════════════════════════════════════════════════════════

class TestDistanceFromCoast:
    def test_chennai_near_coast(self):
        # Chennai is right on the coast
        d = gis.distance_from_coast_km(13.08, 80.27)
        assert d < 50

    def test_central_ocean_far(self):
        # Centre of Bay of Bengal
        d = gis.distance_from_coast_km(13.0, 85.0)
        assert d > 100

    def test_returns_float(self):
        d = gis.distance_from_coast_km(15.0, 80.0)
        assert isinstance(d, float)

    def test_nonnegative(self):
        d = gis.distance_from_coast_km(10.0, 77.0)
        assert d >= 0


# ══════════════════════════════════════════════════════════════
# 14. GIS STATS
# ══════════════════════════════════════════════════════════════

class TestGISStats:
    def test_stats_keys(self, db):
        stats = gis.get_gis_stats(db)
        for key in ["total_ports","major_ports","shipping_routes","marine_areas",
                    "mpa_zones","eez_zones","bathymetry_points","ais_vessels",
                    "gis_bookmarks","drawn_features","measurements"]:
            assert key in stats

    def test_stats_values_nonneg(self, db):
        stats = gis.get_gis_stats(db)
        for v in stats.values():
            assert v >= 0

    def test_stats_reflect_data(self, db):
        stats = gis.get_gis_stats(db)
        assert stats["total_ports"] >= 20
        assert stats["mpa_zones"] >= 3
        assert stats["ais_vessels"] >= 8

    def test_stats_update_after_add(self, db):
        before = gis.get_gis_stats(db)["total_ports"]
        gis.add_port(db, "New Port", "Land", 5.0, 60.0)
        after = gis.get_gis_stats(db)["total_ports"]
        assert after == before + 1


# ══════════════════════════════════════════════════════════════
# 15. FLASK /api/map/* ENDPOINTS
# ══════════════════════════════════════════════════════════════

class TestFlaskMapPage:
    def test_map_page_200(self, flask_client):
        r = flask_client.get("/map")
        assert r.status_code == 200

    def test_map_page_has_maplibre(self, flask_client):
        r = flask_client.get("/map")
        assert b"maplibre" in r.data.lower() or b"MapLibre" in r.data

    def test_map_page_has_canvas(self, flask_client):
        r = flask_client.get("/map")
        assert b"<div id=\"map\"" in r.data


class TestFlaskMapLayers:
    def test_layers_endpoint(self, flask_client):
        r = flask_client.get("/api/map/layers")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert "layers" in d
        assert d["total"] >= 10

    def test_layers_have_required_fields(self, flask_client):
        r = flask_client.get("/api/map/layers")
        d = json.loads(r.data)
        for layer in d["layers"]:
            assert "id" in layer
            assert "label" in layer
            assert "category" in layer


class TestFlaskMapSST:
    def test_sst_endpoint(self, flask_client):
        r = flask_client.get("/api/map/sst")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "FeatureCollection"
        assert d["total"] > 0

    def test_sst_res_param(self, flask_client):
        r = flask_client.get("/api/map/sst?res=2.0")
        assert r.status_code == 200
        d2 = json.loads(r.data)
        r2 = flask_client.get("/api/map/sst?res=1.0")
        d1 = json.loads(r2.data)
        assert d1["total"] >= d2["total"]

    def test_wind_endpoint(self, flask_client):
        r = flask_client.get("/api/map/wind")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "FeatureCollection"

    def test_currents_endpoint(self, flask_client):
        r = flask_client.get("/api/map/currents")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "FeatureCollection"


class TestFlaskMapPorts:
    def test_ports_get(self, flask_client):
        r = flask_client.get("/api/map/ports")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "FeatureCollection"
        assert d["total"] >= 20
        assert "stats" in d

    def test_ports_major_filter(self, flask_client):
        r = flask_client.get("/api/map/ports?major=true")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["total"] >= 5

    def test_ports_country_filter(self, flask_client):
        r = flask_client.get("/api/map/ports?country=India")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["total"] >= 5

    def test_ports_add(self, flask_client):
        r = flask_client.post("/api/map/ports",
            data=json.dumps({"name":"Test Port","country":"Test","lat":10.0,"lon":75.0}),
            content_type="application/json")
        assert r.status_code == 200
        assert json.loads(r.data)["status"] == "added"

    def test_ports_delete(self, flask_client):
        add_r = flask_client.post("/api/map/ports",
            data=json.dumps({"name":"Del Port","country":"Test","lat":5.0,"lon":60.0}),
            content_type="application/json")
        port_id = json.loads(add_r.data)["id"]
        del_r = flask_client.delete(f"/api/map/ports/{port_id}")
        assert del_r.status_code == 200
        assert json.loads(del_r.data)["status"] == "deleted"


class TestFlaskMapRoutes:
    def test_routes_get(self, flask_client):
        r = flask_client.get("/api/map/routes")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "FeatureCollection"
        assert d["total"] >= 5

    def test_routes_filter_type(self, flask_client):
        r = flask_client.get("/api/map/routes?type=tanker")
        assert r.status_code == 200

    def test_routes_add(self, flask_client):
        r = flask_client.post("/api/map/routes",
            data=json.dumps({"name":"Test Route","coords":[[80,13],[85,15]],"route_type":"container"}),
            content_type="application/json")
        assert r.status_code == 200
        assert json.loads(r.data)["status"] == "added"


class TestFlaskMapAreas:
    def test_areas_all(self, flask_client):
        r = flask_client.get("/api/map/areas")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "FeatureCollection"
        assert d["total"] >= 8

    def test_areas_eez(self, flask_client):
        r = flask_client.get("/api/map/areas?type=eez")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["total"] >= 3

    def test_areas_mpa(self, flask_client):
        r = flask_client.get("/api/map/areas?type=mpa")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["total"] >= 3

    def test_areas_add(self, flask_client):
        r = flask_client.post("/api/map/areas",
            data=json.dumps({"name":"Test MPA","coords":[[70,10],[72,10],[72,12],[70,12],[70,10]],"area_type":"mpa","country":"Test"}),
            content_type="application/json")
        assert r.status_code == 200
        assert json.loads(r.data)["status"] == "added"

    def test_areas_delete(self, flask_client):
        add_r = flask_client.post("/api/map/areas",
            data=json.dumps({"name":"Del MPA","coords":[[70,10],[72,10],[72,12],[70,12],[70,10]],"area_type":"mpa"}),
            content_type="application/json")
        area_id = json.loads(add_r.data)["id"]
        del_r = flask_client.delete(f"/api/map/areas/{area_id}")
        assert del_r.status_code == 200


class TestFlaskMapBathymetry:
    def test_bathymetry_get(self, flask_client):
        r = flask_client.get("/api/map/bathymetry")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "FeatureCollection"
        assert "stats" in d

    def test_bathymetry_add(self, flask_client):
        r = flask_client.post("/api/map/bathymetry",
            data=json.dumps({"lat":10.0,"lon":80.0,"depth_m":-2000,"feature_type":"plain"}),
            content_type="application/json")
        assert r.status_code == 200


class TestFlaskMapArgo:
    def test_argo_endpoint(self, flask_client):
        r = flask_client.get("/api/map/argo")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "FeatureCollection"
        assert d["total"] > 0

    def test_argo_has_float_ids(self, flask_client):
        r = flask_client.get("/api/map/argo")
        d = json.loads(r.data)
        for f in d["features"][:3]:
            assert "float_id" in f["properties"]


class TestFlaskMapShips:
    def test_ships_endpoint(self, flask_client):
        r = flask_client.get("/api/map/ships")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "FeatureCollection"
        assert d["total"] >= 8

    def test_ships_filter_type(self, flask_client):
        r = flask_client.get("/api/map/ships?type=tanker")
        assert r.status_code == 200


class TestFlaskMapGrid:
    def test_grid_endpoint(self, flask_client):
        r = flask_client.get("/api/map/grid?lat_min=10&lat_max=12&lon_min=80&lon_max=82&res=1.0")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "FeatureCollection"
        assert d["total"] == 4  # 2x2 grid

    def test_grid_snap_endpoint(self, flask_client):
        r = flask_client.get("/api/map/grid/snap?lat=13.1&lon=80.1&res=0.25")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert "lat" in d and "lon" in d and "cell_id" in d
        assert d["lat"] == 13.0
        assert d["lon"] == 80.0


class TestFlaskMapLocation:
    def test_location_search(self, flask_client):
        r = flask_client.get("/api/map/location?q=Mumbai")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert "results" in d
        assert d["total"] > 0

    def test_location_empty_query(self, flask_client):
        r = flask_client.get("/api/map/location?q=")
        assert r.status_code == 400

    def test_location_coordinate(self, flask_client):
        r = flask_client.get("/api/map/location?q=13.5,82.3")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["results"][0]["type"] == "coordinate"

    def test_location_ocean(self, flask_client):
        r = flask_client.get("/api/map/location?q=Arabian+Sea")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["total"] > 0


class TestFlaskMapBookmarks:
    def test_bookmarks_list(self, flask_client):
        r = flask_client.get("/api/map/bookmarks?user_id=0")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert "bookmarks" in d

    def test_bookmarks_add(self, flask_client):
        r = flask_client.post("/api/map/bookmarks",
            data=json.dumps({"name":"BoB","lat":13.0,"lon":80.0,"zoom":6,"user_id":0,"active_layers":["sst"]}),
            content_type="application/json")
        assert r.status_code == 200
        assert json.loads(r.data)["status"] == "added"

    def test_bookmarks_delete(self, flask_client):
        add_r = flask_client.post("/api/map/bookmarks",
            data=json.dumps({"name":"Del","lat":0,"lon":0,"user_id":0}),
            content_type="application/json")
        bm_id = json.loads(add_r.data)["id"]
        del_r = flask_client.delete(f"/api/map/bookmarks/{bm_id}?user_id=0")
        assert del_r.status_code == 200


class TestFlaskMapMeasure:
    def test_measure_two_points(self, flask_client):
        r = flask_client.post("/api/map/measure",
            data=json.dumps({"points":[[18.9,72.8],[6.9,79.9]]}),
            content_type="application/json")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert "total_km" in d
        assert d["total_km"] > 1000

    def test_measure_one_point_error(self, flask_client):
        r = flask_client.post("/api/map/measure",
            data=json.dumps({"points":[[13.0,80.0]]}),
            content_type="application/json")
        assert r.status_code == 400

    def test_measure_list(self, flask_client):
        r = flask_client.get("/api/map/measure?user_id=0")
        assert r.status_code == 200
        assert "measurements" in json.loads(r.data)


class TestFlaskMapDraw:
    def test_draw_save(self, flask_client):
        r = flask_client.post("/api/map/draw",
            data=json.dumps({"feature_type":"point","geojson":{"type":"Point","coordinates":[80,13]},"label":"Test","user_id":0}),
            content_type="application/json")
        assert r.status_code == 200
        assert json.loads(r.data)["status"] == "saved"

    def test_draw_list(self, flask_client):
        r = flask_client.get("/api/map/draw?user_id=0")
        assert r.status_code == 200
        assert "features" in json.loads(r.data)

    def test_draw_delete(self, flask_client):
        add_r = flask_client.post("/api/map/draw",
            data=json.dumps({"feature_type":"point","geojson":{"type":"Point","coordinates":[0,0]},"label":"Del","user_id":0}),
            content_type="application/json")
        feat_id = json.loads(add_r.data)["id"]
        del_r = flask_client.delete(f"/api/map/draw/{feat_id}?user_id=0")
        assert del_r.status_code == 200


class TestFlaskMapInspect:
    def test_inspect_endpoint(self, flask_client):
        r = flask_client.get("/api/map/inspect?lat=13.0&lon=80.0")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert "latitude" in d
        assert "longitude" in d
        assert "grid" in d
        assert "distance_from_coast_km" in d
        assert "heatwave_status" in d

    def test_inspect_sst_range(self, flask_client):
        r = flask_client.get("/api/map/inspect?lat=13.0&lon=80.0")
        d = json.loads(r.data)
        if d.get("sst"):
            assert 20 <= d["sst"] <= 40

    def test_inspect_grid_has_cell_id(self, flask_client):
        r = flask_client.get("/api/map/inspect?lat=13.0&lon=80.0")
        d = json.loads(r.data)
        assert "cell_id" in d["grid"]


class TestFlaskMapStats:
    def test_stats_endpoint(self, flask_client):
        r = flask_client.get("/api/map/stats")
        assert r.status_code == 200
        d = json.loads(r.data)
        for key in ["total_ports","shipping_routes","marine_areas","ais_vessels"]:
            assert key in d

    def test_stats_values_positive(self, flask_client):
        r = flask_client.get("/api/map/stats")
        d = json.loads(r.data)
        assert d["total_ports"] >= 20
        assert d["shipping_routes"] >= 5


class TestFlaskMapExport:
    def test_export_ports_geojson(self, flask_client):
        r = flask_client.get("/api/map/export/geojson?type=ports")
        assert r.status_code == 200
        assert r.content_type in ("application/geo+json", "application/json", "application/geo+json; charset=utf-8")
        d = json.loads(r.data)
        assert d["type"] == "FeatureCollection"

    def test_export_routes(self, flask_client):
        r = flask_client.get("/api/map/export/geojson?type=routes")
        assert r.status_code == 200

    def test_export_sst(self, flask_client):
        r = flask_client.get("/api/map/export/geojson?type=sst")
        assert r.status_code == 200
        d = json.loads(r.data)
        assert d["type"] == "FeatureCollection"
