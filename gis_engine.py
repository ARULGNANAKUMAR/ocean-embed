"""
gis_engine.py
OceanVerse AI v4.0 — Phase 2 GIS Engine
Pure-Python, SQLite-backed geospatial data module.

All functions are additive — no existing modules touched.
Provides:
  - Schema for ports, shipping_routes, marine_areas,
    bathymetry_points, gis_map_bookmarks, ais_vessels,
    time_animation_frames, measurement_sessions, drawn_features
  - Seed data for all Indian Ocean features
  - Spatial query helpers
  - GeoJSON builders
  - Grid utilities
"""

import json
import math
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

# ──────────────────────────────────────────────────────────────
# SCHEMA
# ──────────────────────────────────────────────────────────────

SCHEMA_GIS = """
-- ── Ports ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    country     TEXT,
    port_type   TEXT DEFAULT 'commercial',
    latitude    REAL NOT NULL,
    longitude   REAL NOT NULL,
    capacity_teu INTEGER DEFAULT 0,
    annual_traffic_mt REAL DEFAULT 0,
    is_major    INTEGER DEFAULT 0,
    notes       TEXT
);

-- ── Shipping routes ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS shipping_routes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    route_type  TEXT DEFAULT 'commercial',
    traffic_level TEXT DEFAULT 'medium',
    geojson     TEXT,
    color       TEXT DEFAULT '#f1c40f',
    notes       TEXT
);

-- ── Marine protected areas & EEZ zones ──────────────────────
CREATE TABLE IF NOT EXISTS marine_areas (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    area_type   TEXT DEFAULT 'mpa',
    country     TEXT,
    geojson     TEXT,
    color       TEXT DEFAULT '#27ae60',
    fill_opacity REAL DEFAULT 0.1,
    notes       TEXT
);

-- ── Bathymetry sample points ─────────────────────────────────
CREATE TABLE IF NOT EXISTS bathymetry_points (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    latitude    REAL NOT NULL,
    longitude   REAL NOT NULL,
    depth_m     REAL NOT NULL,
    feature_type TEXT DEFAULT 'plain'
);

-- ── GIS map bookmarks (Phase 2 extended) ─────────────────────
CREATE TABLE IF NOT EXISTS gis_map_bookmarks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER DEFAULT 0,
    name        TEXT NOT NULL,
    latitude    REAL,
    longitude   REAL,
    zoom        REAL DEFAULT 5,
    bearing     REAL DEFAULT 0,
    pitch       REAL DEFAULT 0,
    active_layers TEXT DEFAULT '[]',
    basemap     TEXT DEFAULT 'ocean',
    created_at  TEXT,
    notes       TEXT
);

-- ── AIS vessel positions ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS ais_vessels (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    mmsi        TEXT UNIQUE,
    name        TEXT,
    vessel_type TEXT DEFAULT 'cargo',
    latitude    REAL,
    longitude   REAL,
    speed_kn    REAL DEFAULT 0,
    heading_deg REAL DEFAULT 0,
    destination TEXT,
    last_update TEXT
);

-- ── Time animation frames cache ───────────────────────────────
CREATE TABLE IF NOT EXISTS time_frames (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    layer_name  TEXT,
    frame_date  TEXT,
    data_json   TEXT,
    created_at  TEXT
);

-- ── Measurement sessions ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS measurement_sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER DEFAULT 0,
    points_json TEXT,
    total_km    REAL,
    label       TEXT,
    created_at  TEXT
);

-- ── Drawn features (polygon / line / point) ──────────────────
CREATE TABLE IF NOT EXISTS drawn_features (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER DEFAULT 0,
    feature_type TEXT DEFAULT 'polygon',
    geojson     TEXT,
    label       TEXT,
    color       TEXT DEFAULT '#22e5ff',
    created_at  TEXT
);

-- ── Location search cache ────────────────────────────────────
CREATE TABLE IF NOT EXISTS location_search_cache (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    query       TEXT UNIQUE,
    result_json TEXT,
    cached_at   TEXT
);
"""

# ──────────────────────────────────────────────────────────────
# SEED DATA
# ──────────────────────────────────────────────────────────────

PORTS_SEED = [
    # India
    ("Mumbai", "India", "major_commercial", 18.9220, 72.8347, 5800000, 71.0, 1),
    ("Chennai", "India", "container", 13.0827, 80.2707, 2000000, 48.0, 1),
    ("Kolkata", "India", "river", 22.5726, 88.3639, 750000, 18.0, 1),
    ("Visakhapatnam", "India", "industrial", 17.6868, 83.2185, 1000000, 65.0, 1),
    ("Kochi", "India", "container", 9.9312, 76.2673, 450000, 12.0, 0),
    ("Kandla", "India", "dry_bulk", 23.0330, 70.2170, 1200000, 132.0, 1),
    ("Paradip", "India", "bulk", 20.3170, 86.6114, 500000, 100.0, 0),
    ("Nhava Sheva", "India", "container", 18.9497, 72.9476, 5500000, 80.0, 1),
    ("Tuticorin", "India", "bulk", 8.7642, 78.1348, 600000, 28.0, 0),
    ("Mormugao", "India", "ore", 15.4089, 73.7997, 450000, 40.0, 0),
    # Sri Lanka
    ("Colombo", "Sri Lanka", "trans_shipment", 6.9271, 79.8612, 7200000, 87.0, 1),
    ("Hambantota", "Sri Lanka", "emerging", 6.1241, 81.1185, 500000, 8.0, 0),
    # Bangladesh
    ("Chittagong", "Bangladesh", "container", 22.3569, 91.7832, 3000000, 92.0, 1),
    ("Mongla", "Bangladesh", "bulk", 22.4868, 89.5988, 200000, 5.0, 0),
    # Pakistan
    ("Karachi", "Pakistan", "major_commercial", 24.8607, 67.0011, 1800000, 35.0, 1),
    ("Gwadar", "Pakistan", "emerging", 25.1216, 62.3254, 300000, 3.0, 0),
    # Myanmar
    ("Yangon", "Myanmar", "river", 16.8661, 96.1951, 750000, 10.0, 0),
    ("Thilawa", "Myanmar", "sez", 16.6897, 96.2592, 400000, 5.0, 0),
    # Oman
    ("Muscat", "Oman", "commercial", 23.6141, 58.5922, 1200000, 15.0, 0),
    ("Salalah", "Oman", "container", 16.9390, 54.0088, 4500000, 19.0, 1),
    # UAE
    ("Dubai (Jebel Ali)", "UAE", "mega_hub", 24.9857, 55.0272, 14500000, 80.0, 1),
    ("Abu Dhabi", "UAE", "oil_terminal", 24.4539, 54.3773, 3000000, 50.0, 1),
    # Singapore
    ("Singapore", "Singapore", "mega_hub", 1.2644, 103.8222, 37500000, 630.0, 1),
    # Yemen
    ("Aden", "Yemen", "commercial", 12.7855, 45.0187, 500000, 7.0, 0),
    # Kenya
    ("Mombasa", "Kenya", "regional_hub", -4.0435, 39.6682, 1200000, 30.0, 1),
    # Tanzania
    ("Dar es Salaam", "Tanzania", "commercial", -6.8161, 39.2803, 800000, 14.0, 0),
    # Mauritius
    ("Port Louis", "Mauritius", "trans_shipment", -20.1609, 57.5012, 750000, 5.0, 0),
    # Madagascar
    ("Toamasina", "Madagascar", "commercial", -18.1492, 49.3958, 200000, 3.0, 0),
    # Mozambique
    ("Maputo", "Mozambique", "bulk", -25.9692, 32.5732, 400000, 16.0, 0),
    # Malaysia
    ("Port Klang", "Malaysia", "container", 3.0200, 101.3910, 13200000, 130.0, 1),
    # Thailand
    ("Laem Chabang", "Thailand", "container", 13.0878, 100.8851, 7600000, 56.0, 1),
    # Indonesia
    ("Tanjung Priok", "Indonesia", "container", -6.0970, 106.8845, 8500000, 60.0, 1),
]

SHIPPING_ROUTES_SEED = [
    {
        "name": "Strait of Malacca → Arabian Sea (Main VLCC)",
        "type": "tanker",
        "traffic": "very_high",
        "color": "#e74c3c",
        "coords": [[103.8, 1.3],[95.0, 5.0],[88.0, 8.0],[80.0, 10.0],[72.0, 15.0],[60.0, 19.0],[55.0, 22.0],[45.0, 12.0]],
    },
    {
        "name": "India West Coast → Suez Canal",
        "type": "container",
        "traffic": "very_high",
        "color": "#e74c3c",
        "coords": [[72.8, 18.9],[66.0, 20.0],[58.0, 22.0],[50.0, 18.0],[45.0, 13.0],[43.0, 12.0]],
    },
    {
        "name": "Bay of Bengal → Singapore",
        "type": "container",
        "traffic": "high",
        "color": "#f39c12",
        "coords": [[88.4, 22.4],[92.0, 18.0],[96.0, 14.0],[100.0, 7.0],[103.8, 1.3]],
    },
    {
        "name": "Mumbai → East Africa",
        "type": "container",
        "traffic": "medium",
        "color": "#f1c40f",
        "coords": [[72.8, 18.9],[68.0, 15.0],[62.0, 10.0],[55.0, 5.0],[48.0, -1.0],[40.0, -4.0]],
    },
    {
        "name": "Colombo Hub Spokes",
        "type": "feeder",
        "traffic": "high",
        "color": "#2ecc71",
        "coords": [[79.9, 6.9],[76.3, 9.9],[80.3, 13.1],[83.2, 17.7]],
    },
    {
        "name": "India East Coast → SE Asia",
        "type": "container",
        "traffic": "medium",
        "color": "#f39c12",
        "coords": [[80.3, 13.1],[83.0, 15.0],[86.6, 20.3],[92.0, 20.0],[96.0, 16.0],[100.5, 13.0],[103.8, 1.3]],
    },
    {
        "name": "Persian Gulf → Indian Subcontinent",
        "type": "tanker",
        "traffic": "very_high",
        "color": "#e74c3c",
        "coords": [[55.0, 24.0],[60.0, 22.0],[65.0, 20.0],[67.0, 24.9],[70.2, 23.0],[72.8, 18.9]],
    },
    {
        "name": "Horn of Africa → Arabian Sea",
        "type": "mixed",
        "traffic": "high",
        "color": "#e67e22",
        "coords": [[45.0, 11.5],[50.0, 14.0],[55.0, 18.0],[58.6, 23.6]],
    },
    {
        "name": "Chagos Arch. Crossroads",
        "type": "commercial",
        "traffic": "medium",
        "color": "#f1c40f",
        "coords": [[72.4, -6.3],[75.0, 0.0],[79.0, 6.9]],
    },
    {
        "name": "South Indian Ocean Transit",
        "type": "bulk",
        "traffic": "low",
        "color": "#95a5a6",
        "coords": [[57.5, -20.2],[65.0, -18.0],[72.0, -15.0],[80.0, -10.0]],
    },
]

MARINE_AREAS_SEED = [
    # MPA zones
    {
        "name": "Gulf of Mannar Biosphere Reserve",
        "type": "mpa", "country": "India", "color": "#27ae60",
        "coords": [[78.8,8.4],[79.5,8.4],[79.5,9.2],[78.8,9.2],[78.8,8.4]],
    },
    {
        "name": "Andaman & Nicobar MPA",
        "type": "mpa", "country": "India", "color": "#27ae60",
        "coords": [[92.0,6.5],[94.5,6.5],[94.5,14.0],[92.0,14.0],[92.0,6.5]],
    },
    {
        "name": "Lakshadweep Sea Protected Zone",
        "type": "mpa", "country": "India", "color": "#27ae60",
        "coords": [[71.5,8.0],[74.5,8.0],[74.5,13.5],[71.5,13.5],[71.5,8.0]],
    },
    {
        "name": "Chagos Marine Reserve",
        "type": "mpa", "country": "UK (BIOT)", "color": "#16a085",
        "coords": [[70.0,-9.0],[74.0,-9.0],[74.0,-4.0],[70.0,-4.0],[70.0,-9.0]],
    },
    {
        "name": "Maldives Marine Reserve",
        "type": "mpa", "country": "Maldives", "color": "#27ae60",
        "coords": [[72.0,1.0],[74.5,1.0],[74.5,8.0],[72.0,8.0],[72.0,1.0]],
    },
    # EEZ
    {
        "name": "India EEZ",
        "type": "eez", "country": "India", "color": "#e67e22",
        "coords": [[68.0,8.0],[68.0,22.0],[80.0,22.0],[88.0,22.0],[88.0,14.0],[80.5,8.0],[74.0,6.0],[68.0,8.0]],
    },
    {
        "name": "Sri Lanka EEZ",
        "type": "eez", "country": "Sri Lanka", "color": "#d35400",
        "coords": [[79.0,5.5],[82.5,5.5],[82.5,10.5],[79.0,10.5],[79.0,5.5]],
    },
    {
        "name": "Bangladesh EEZ",
        "type": "eez", "country": "Bangladesh", "color": "#d35400",
        "coords": [[88.0,18.5],[91.0,18.5],[92.5,20.5],[90.0,22.5],[88.0,22.5],[88.0,18.5]],
    },
    {
        "name": "Myanmar EEZ",
        "type": "eez", "country": "Myanmar", "color": "#d35400",
        "coords": [[92.5,10.0],[98.5,10.0],[100.0,16.0],[98.0,22.0],[92.5,22.0],[92.5,10.0]],
    },
    {
        "name": "Pakistan EEZ",
        "type": "eez", "country": "Pakistan", "color": "#d35400",
        "coords": [[60.0,22.0],[67.0,22.0],[68.0,24.0],[65.0,25.0],[62.0,25.5],[60.0,24.0],[60.0,22.0]],
    },
    {
        "name": "Oman EEZ",
        "type": "eez", "country": "Oman", "color": "#d35400",
        "coords": [[52.0,16.5],[58.0,20.0],[60.0,22.0],[56.0,24.0],[52.0,22.0],[52.0,16.5]],
    },
    {
        "name": "Kenya EEZ",
        "type": "eez", "country": "Kenya", "color": "#d35400",
        "coords": [[37.0,-4.5],[41.0,-4.5],[44.0,2.0],[41.0,2.0],[37.0,2.0],[37.0,-4.5]],
    },
]

BATHYMETRY_SEED = [
    # Bay of Bengal features
    (13.0, 90.0, -5000, "abyssal_plain"),
    (15.0, 90.0, -3800, "abyssal_plain"),
    (8.0,  92.0, -3500, "abyssal_plain"),
    (5.0,  90.0, -4200, "abyssal_plain"),
    (0.0,  90.0, -5500, "ninety_east_ridge"),
    (-5.0, 90.0, -2000, "ninety_east_ridge"),
    (10.0, 90.0, -1500, "ninety_east_ridge"),
    # Sunda Trench
    (5.0, 104.0, -7200, "trench"),
    (0.0, 103.0, -7000, "trench"),
    (-5.0,102.0, -6800, "trench"),
    (-8.0,106.0, -7400, "trench"),
    # Arabian Sea
    (15.0, 65.0, -3800, "abyssal_plain"),
    (18.0, 62.0, -3200, "abyssal_plain"),
    (12.0, 68.0, -2500, "abyssal_plain"),
    # Carlsberg Ridge
    (10.0, 57.0, -1500, "ridge"),
    (6.0,  60.0, -2000, "ridge"),
    (3.0,  63.0, -2500, "ridge"),
    # Continental shelf (200m)
    (20.0, 70.0, -200, "shelf"),
    (18.0, 72.0, -150, "shelf"),
    (15.0, 73.0, -180, "shelf"),
    (13.0, 80.0, -100, "shelf"),
    (10.0, 78.0, -120, "shelf"),
    # Agulhas Basin
    (-30.0, 25.0, -4800, "abyssal_plain"),
    (-25.0, 30.0, -5000, "abyssal_plain"),
    # Central Indian Ridge
    (-20.0, 65.0, -2000, "ridge"),
    (-10.0, 65.0, -1500, "ridge"),
    (0.0,   67.0, -1200, "ridge"),
]

AIS_VESSELS_SEED = [
    ("338234560", "MV Pacific Voyager", "container", 13.5, 82.0, 14.2, 180, "Colombo"),
    ("566789012", "MT Arabian Star", "tanker", 18.0, 66.0, 11.8, 270, "Muscat"),
    ("211345678", "MV Bengal Express", "container", 22.0, 89.5, 12.5, 120, "Singapore"),
    ("432100987", "MV Andaman Sky", "bulk_carrier", 11.0, 93.0, 9.3, 200, "Chennai"),
    ("567891234", "MT Gulf Trader", "tanker", 24.5, 57.0, 13.1, 90, "Jebel Ali"),
    ("123456789", "MV Sri Lanka Pride", "feeder", 7.5, 80.5, 10.0, 300, "Mumbai"),
    ("987654321", "MV Indian Jewel", "container", 15.0, 73.5, 15.5, 150, "Kochi"),
    ("445566778", "MV Ocean Master", "bulk_carrier", 8.0, 77.0, 8.7, 45, "Tuticorin"),
    ("334455667", "RV Science Quest", "research", 12.0, 85.0, 6.5, 90, "Visakhapatnam"),
    ("223344556", "MV Maldives Link", "ferry", 5.0, 74.0, 12.0, 0, "Malé"),
    ("112233445", "MV East Africa Trader", "container", -2.0, 45.0, 11.0, 315, "Mombasa"),
    ("998877665", "MT Oman Glory", "tanker", 22.0, 60.0, 12.8, 200, "Salalah"),
]


def init_gis_tables(conn: sqlite3.Connection) -> None:
    """Create all GIS tables. Idempotent."""
    conn.executescript(SCHEMA_GIS)
    conn.commit()
    _seed_gis_data(conn)


def _seed_gis_data(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # Ports
    cur.execute("SELECT COUNT(*) FROM ports")
    if cur.fetchone()[0] == 0:
        for p in PORTS_SEED:
            conn.execute("""INSERT INTO ports
                (name,country,port_type,latitude,longitude,capacity_teu,annual_traffic_mt,is_major)
                VALUES (?,?,?,?,?,?,?,?)""", p)

    # Shipping routes
    cur.execute("SELECT COUNT(*) FROM shipping_routes")
    if cur.fetchone()[0] == 0:
        for r in SHIPPING_ROUTES_SEED:
            geojson = json.dumps({
                "type": "Feature",
                "properties": {"name": r["name"], "type": r["type"], "traffic": r["traffic"]},
                "geometry": {
                    "type": "LineString",
                    "coordinates": r["coords"]
                }
            })
            conn.execute("""INSERT INTO shipping_routes
                (name,route_type,traffic_level,geojson,color) VALUES (?,?,?,?,?)""",
                (r["name"], r["type"], r["traffic"], geojson, r["color"]))

    # Marine areas
    cur.execute("SELECT COUNT(*) FROM marine_areas")
    if cur.fetchone()[0] == 0:
        for a in MARINE_AREAS_SEED:
            geojson = json.dumps({
                "type": "Feature",
                "properties": {"name": a["name"], "type": a["type"], "country": a["country"]},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [a["coords"]]
                }
            })
            conn.execute("""INSERT INTO marine_areas
                (name,area_type,country,geojson,color) VALUES (?,?,?,?,?)""",
                (a["name"], a["type"], a["country"], geojson, a["color"]))

    # Bathymetry
    cur.execute("SELECT COUNT(*) FROM bathymetry_points")
    if cur.fetchone()[0] == 0:
        for b in BATHYMETRY_SEED:
            conn.execute("INSERT INTO bathymetry_points (latitude,longitude,depth_m,feature_type) VALUES (?,?,?,?)", b)

    # AIS vessels
    cur.execute("SELECT COUNT(*) FROM ais_vessels")
    if cur.fetchone()[0] == 0:
        now = _now()
        for v in AIS_VESSELS_SEED:
            conn.execute("""INSERT OR IGNORE INTO ais_vessels
                (mmsi,name,vessel_type,latitude,longitude,speed_kn,heading_deg,destination,last_update)
                VALUES (?,?,?,?,?,?,?,?,?)""", (*v, now))

    conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────
# PORTS
# ──────────────────────────────────────────────────────────────

def get_ports(conn: sqlite3.Connection, major_only: bool = False,
              country: str = None, bbox: dict = None) -> dict:
    cur = conn.cursor()
    sql = "SELECT * FROM ports WHERE 1=1"
    params = []
    if major_only:
        sql += " AND is_major=1"
    if country:
        sql += " AND country=?"
        params.append(country)
    if bbox:
        sql += " AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?"
        params += [bbox["lat_min"], bbox["lat_max"], bbox["lon_min"], bbox["lon_max"]]
    cur.execute(sql + " ORDER BY annual_traffic_mt DESC", params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return _to_geojson_points(rows, "name", extra_props=["country","port_type","capacity_teu","annual_traffic_mt","is_major"])


def add_port(conn: sqlite3.Connection, name: str, country: str, lat: float,
             lon: float, port_type: str = "commercial") -> dict:
    cur = conn.execute("""INSERT INTO ports (name,country,port_type,latitude,longitude)
        VALUES (?,?,?,?,?)""", (name, country, port_type, lat, lon))
    conn.commit()
    return {"status": "added", "id": cur.lastrowid}


def delete_port(conn: sqlite3.Connection, port_id: int) -> dict:
    conn.execute("DELETE FROM ports WHERE id=?", (port_id,))
    conn.commit()
    return {"status": "deleted", "id": port_id}


def get_ports_stats(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ports")
    total = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM ports WHERE is_major=1")
    major = cur.fetchone()[0]
    cur.execute("SELECT country, COUNT(*) as n FROM ports GROUP BY country ORDER BY n DESC LIMIT 5")
    by_country = [{"country": r[0], "count": r[1]} for r in cur.fetchall()]
    cur.execute("SELECT SUM(annual_traffic_mt) FROM ports")
    total_traffic = cur.fetchone()[0] or 0
    return {"total": total, "major": major, "by_country": by_country,
            "total_traffic_mt": round(total_traffic, 1)}


# ──────────────────────────────────────────────────────────────
# SHIPPING ROUTES
# ──────────────────────────────────────────────────────────────

def get_shipping_routes(conn: sqlite3.Connection,
                        route_type: str = None, traffic: str = None) -> dict:
    cur = conn.cursor()
    sql = "SELECT * FROM shipping_routes WHERE 1=1"
    params = []
    if route_type:
        sql += " AND route_type=?"; params.append(route_type)
    if traffic:
        sql += " AND traffic_level=?"; params.append(traffic)
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    features = []
    for r in rows:
        try:
            feat = json.loads(r["geojson"])
            feat["properties"].update({"color": r["color"], "traffic_level": r["traffic_level"]})
            features.append(feat)
        except Exception:
            pass
    return {"type": "FeatureCollection", "features": features, "total": len(features)}


def add_shipping_route(conn: sqlite3.Connection, name: str, coords: list,
                       route_type: str = "commercial", color: str = "#f1c40f") -> dict:
    geojson = json.dumps({
        "type": "Feature",
        "properties": {"name": name, "type": route_type},
        "geometry": {"type": "LineString", "coordinates": coords}
    })
    cur = conn.execute("""INSERT INTO shipping_routes (name,route_type,geojson,color)
        VALUES (?,?,?,?)""", (name, route_type, geojson, color))
    conn.commit()
    return {"status": "added", "id": cur.lastrowid}


# ──────────────────────────────────────────────────────────────
# MARINE AREAS (MPA + EEZ)
# ──────────────────────────────────────────────────────────────

def get_marine_areas(conn: sqlite3.Connection, area_type: str = None,
                     country: str = None) -> dict:
    cur = conn.cursor()
    sql = "SELECT * FROM marine_areas WHERE 1=1"
    params = []
    if area_type:
        sql += " AND area_type=?"; params.append(area_type)
    if country:
        sql += " AND country=?"; params.append(country)
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]

    features = []
    for r in rows:
        try:
            feat = json.loads(r["geojson"])
            feat["properties"].update({"color": r["color"], "fill_opacity": r.get("fill_opacity", 0.1)})
            features.append(feat)
        except Exception:
            pass
    return {"type": "FeatureCollection", "features": features, "total": len(features)}


def add_marine_area(conn: sqlite3.Connection, name: str, coords: list,
                    area_type: str = "mpa", country: str = "", color: str = "#27ae60") -> dict:
    geojson = json.dumps({
        "type": "Feature",
        "properties": {"name": name, "type": area_type, "country": country},
        "geometry": {"type": "Polygon", "coordinates": [coords]}
    })
    cur = conn.execute("""INSERT INTO marine_areas (name,area_type,country,geojson,color)
        VALUES (?,?,?,?,?)""", (name, area_type, country, geojson, color))
    conn.commit()
    return {"status": "added", "id": cur.lastrowid}


def delete_marine_area(conn: sqlite3.Connection, area_id: int) -> dict:
    conn.execute("DELETE FROM marine_areas WHERE id=?", (area_id,))
    conn.commit()
    return {"status": "deleted", "id": area_id}


# ──────────────────────────────────────────────────────────────
# BATHYMETRY
# ──────────────────────────────────────────────────────────────

def get_bathymetry(conn: sqlite3.Connection, bbox: dict = None,
                   feature_type: str = None) -> dict:
    cur = conn.cursor()
    sql = "SELECT * FROM bathymetry_points WHERE 1=1"
    params = []
    if bbox:
        sql += " AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?"
        params += [bbox["lat_min"], bbox["lat_max"], bbox["lon_min"], bbox["lon_max"]]
    if feature_type:
        sql += " AND feature_type=?"; params.append(feature_type)
    cur.execute(sql + " ORDER BY depth_m", params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return _to_geojson_points(rows, "feature_type", extra_props=["depth_m", "feature_type"])


def get_bathymetry_stats(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    cur.execute("SELECT MIN(depth_m), MAX(depth_m), AVG(depth_m), COUNT(*) FROM bathymetry_points")
    row = cur.fetchone()
    cur.execute("SELECT feature_type, COUNT(*) FROM bathymetry_points GROUP BY feature_type")
    by_type = [{"type": r[0], "count": r[1]} for r in cur.fetchall()]
    return {
        "min_depth_m": row[0], "max_depth_m": row[1],
        "avg_depth_m": round(row[2], 1) if row[2] else 0,
        "total_points": row[3], "by_feature_type": by_type,
    }


def add_bathymetry_point(conn: sqlite3.Connection, lat: float, lon: float,
                         depth_m: float, feature_type: str = "plain") -> dict:
    cur = conn.execute("""INSERT INTO bathymetry_points (latitude,longitude,depth_m,feature_type)
        VALUES (?,?,?,?)""", (lat, lon, depth_m, feature_type))
    conn.commit()
    return {"status": "added", "id": cur.lastrowid}


# ──────────────────────────────────────────────────────────────
# AIS VESSELS
# ──────────────────────────────────────────────────────────────

def get_ais_vessels(conn: sqlite3.Connection, vessel_type: str = None,
                   bbox: dict = None) -> dict:
    cur = conn.cursor()
    sql = "SELECT * FROM ais_vessels WHERE 1=1"
    params = []
    if vessel_type:
        sql += " AND vessel_type=?"; params.append(vessel_type)
    if bbox:
        sql += " AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?"
        params += [bbox["lat_min"], bbox["lat_max"], bbox["lon_min"], bbox["lon_max"]]
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    return _to_geojson_points(rows, "name", extra_props=["mmsi","vessel_type","speed_kn","heading_deg","destination","last_update"])


def update_vessel_position(conn: sqlite3.Connection, mmsi: str,
                           lat: float, lon: float, speed: float = 0,
                           heading: float = 0) -> dict:
    conn.execute("""UPDATE ais_vessels
        SET latitude=?, longitude=?, speed_kn=?, heading_deg=?, last_update=?
        WHERE mmsi=?""", (lat, lon, speed, heading, _now(), mmsi))
    conn.commit()
    return {"status": "updated", "mmsi": mmsi}


def get_vessel_by_mmsi(conn: sqlite3.Connection, mmsi: str) -> Optional[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM ais_vessels WHERE mmsi=?", (mmsi,))
    row = cur.fetchone()
    if not row:
        return None
    return dict(zip([d[0] for d in cur.description], row))


# ──────────────────────────────────────────────────────────────
# GIS MAP BOOKMARKS
# ──────────────────────────────────────────────────────────────

def add_gis_bookmark(conn: sqlite3.Connection, user_id: int, name: str,
                     lat: float, lon: float, zoom: float = 5,
                     bearing: float = 0, pitch: float = 0,
                     active_layers: list = None, basemap: str = "ocean",
                     notes: str = "") -> dict:
    cur = conn.execute("""INSERT INTO gis_map_bookmarks
        (user_id,name,latitude,longitude,zoom,bearing,pitch,active_layers,basemap,created_at,notes)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (user_id, name, lat, lon, zoom, bearing, pitch,
         json.dumps(active_layers or []), basemap, _now(), notes))
    conn.commit()
    return {"status": "added", "id": cur.lastrowid}


def list_gis_bookmarks(conn: sqlite3.Connection, user_id: int = 0) -> list:
    cur = conn.cursor()
    cur.execute("SELECT * FROM gis_map_bookmarks WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        try:
            r["active_layers"] = json.loads(r["active_layers"])
        except Exception:
            r["active_layers"] = []
    return rows


def delete_gis_bookmark(conn: sqlite3.Connection, bm_id: int, user_id: int = 0) -> dict:
    conn.execute("DELETE FROM gis_map_bookmarks WHERE id=? AND user_id=?", (bm_id, user_id))
    conn.commit()
    return {"status": "deleted", "id": bm_id}


# ──────────────────────────────────────────────────────────────
# MEASUREMENTS
# ──────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def calculate_measurement(points: list) -> dict:
    """points: [[lat,lon], [lat,lon], ...]"""
    if len(points) < 2:
        return {"total_km": 0, "segments": [], "nm": 0}
    segments = []
    total = 0.0
    for i in range(len(points) - 1):
        d = haversine_km(points[i][0], points[i][1], points[i+1][0], points[i+1][1])
        segments.append({"from": points[i], "to": points[i+1], "km": round(d, 2)})
        total += d
    return {"total_km": round(total, 2), "total_nm": round(total / 1.852, 2), "segments": segments}


def save_measurement(conn: sqlite3.Connection, user_id: int, points: list, label: str = "") -> dict:
    result = calculate_measurement(points)
    cur = conn.execute("""INSERT INTO measurement_sessions
        (user_id,points_json,total_km,label,created_at) VALUES (?,?,?,?,?)""",
        (user_id, json.dumps(points), result["total_km"], label, _now()))
    conn.commit()
    return {"status": "saved", "id": cur.lastrowid, **result}


def list_measurements(conn: sqlite3.Connection, user_id: int) -> list:
    cur = conn.cursor()
    cur.execute("SELECT * FROM measurement_sessions WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


# ──────────────────────────────────────────────────────────────
# DRAWN FEATURES
# ──────────────────────────────────────────────────────────────

def save_drawn_feature(conn: sqlite3.Connection, user_id: int,
                       feature_type: str, geojson: dict,
                       label: str = "", color: str = "#22e5ff") -> dict:
    cur = conn.execute("""INSERT INTO drawn_features
        (user_id,feature_type,geojson,label,color,created_at) VALUES (?,?,?,?,?,?)""",
        (user_id, feature_type, json.dumps(geojson), label, color, _now()))
    conn.commit()
    return {"status": "saved", "id": cur.lastrowid}


def list_drawn_features(conn: sqlite3.Connection, user_id: int) -> list:
    cur = conn.cursor()
    cur.execute("SELECT * FROM drawn_features WHERE user_id=? ORDER BY created_at DESC", (user_id,))
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, r)) for r in cur.fetchall()]
    for r in rows:
        try:
            r["geojson"] = json.loads(r["geojson"])
        except Exception:
            pass
    return rows


def delete_drawn_feature(conn: sqlite3.Connection, feat_id: int, user_id: int = 0) -> dict:
    conn.execute("DELETE FROM drawn_features WHERE id=? AND user_id=?", (feat_id, user_id))
    conn.commit()
    return {"status": "deleted", "id": feat_id}


# ──────────────────────────────────────────────────────────────
# GRID UTILITIES
# ──────────────────────────────────────────────────────────────

def get_grid_cells(lat_min: float, lat_max: float, lon_min: float, lon_max: float,
                   res: float = 0.25) -> dict:
    """Generate 0.25° GeoJSON grid for the given bbox."""
    features = []
    lat = lat_min
    while lat < lat_max:
        lon = lon_min
        while lon < lon_max:
            cell_id = f"{lat:.2f}_{lon:.2f}"
            features.append({
                "type": "Feature",
                "properties": {
                    "cell_id": cell_id,
                    "lat_center": round(lat + res/2, 3),
                    "lon_center": round(lon + res/2, 3),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [lon, lat], [lon+res, lat],
                        [lon+res, lat+res], [lon, lat+res], [lon, lat]
                    ]]
                }
            })
            lon = round(lon + res, 6)
        lat = round(lat + res, 6)
    return {"type": "FeatureCollection", "features": features, "total": len(features)}


def snap_to_grid(lat: float, lon: float, res: float = 0.25) -> dict:
    """Snap a coordinate to nearest 0.25° grid centre."""
    lat_snapped = round(round(lat / res) * res, 3)
    lon_snapped = round(round(lon / res) * res, 3)
    return {"lat": lat_snapped, "lon": lon_snapped,
            "cell_id": f"{lat_snapped:.2f}_{lon_snapped:.2f}"}


# ──────────────────────────────────────────────────────────────
# LOCATION SEARCH
# ──────────────────────────────────────────────────────────────

# Built-in gazeteer — Indian Ocean cities and ocean points
GAZETEER = [
    {"name": "Bay of Bengal", "lat": 13.0, "lon": 85.0, "type": "ocean"},
    {"name": "Arabian Sea", "lat": 17.0, "lon": 65.0, "type": "ocean"},
    {"name": "Indian Ocean", "lat": -10.0, "lon": 70.0, "type": "ocean"},
    {"name": "Laccadive Sea", "lat": 10.0, "lon": 74.0, "type": "ocean"},
    {"name": "Andaman Sea", "lat": 11.0, "lon": 96.0, "type": "ocean"},
    {"name": "Mumbai", "lat": 19.076, "lon": 72.878, "type": "city"},
    {"name": "Chennai", "lat": 13.083, "lon": 80.271, "type": "city"},
    {"name": "Kolkata", "lat": 22.573, "lon": 88.364, "type": "city"},
    {"name": "Visakhapatnam", "lat": 17.687, "lon": 83.218, "type": "city"},
    {"name": "Kochi", "lat": 9.931, "lon": 76.267, "type": "city"},
    {"name": "Colombo", "lat": 6.927, "lon": 79.861, "type": "city"},
    {"name": "Singapore", "lat": 1.352, "lon": 103.820, "type": "city"},
    {"name": "Dhaka", "lat": 23.811, "lon": 90.412, "type": "city"},
    {"name": "Karachi", "lat": 24.861, "lon": 67.010, "type": "city"},
    {"name": "Yangon", "lat": 16.866, "lon": 96.195, "type": "city"},
    {"name": "Muscat", "lat": 23.614, "lon": 58.592, "type": "city"},
    {"name": "Aden", "lat": 12.786, "lon": 45.019, "type": "city"},
    {"name": "Mombasa", "lat": -4.044, "lon": 39.668, "type": "city"},
    {"name": "Malé", "lat": 4.175, "lon": 73.509, "type": "city"},
    {"name": "Colombo", "lat": 6.927, "lon": 79.861, "type": "city"},
    {"name": "Ninety East Ridge", "lat": 5.0, "lon": 90.0, "type": "feature"},
    {"name": "Sunda Trench", "lat": -5.0, "lon": 104.0, "type": "feature"},
    {"name": "Carlsberg Ridge", "lat": 8.0, "lon": 60.0, "type": "feature"},
    {"name": "Chagos Archipelago", "lat": -6.3, "lon": 71.9, "type": "island"},
    {"name": "Lakshadweep Islands", "lat": 10.5, "lon": 72.6, "type": "island"},
    {"name": "Andaman Islands", "lat": 12.0, "lon": 92.7, "type": "island"},
    {"name": "Maldives", "lat": 4.2, "lon": 73.5, "type": "island"},
    {"name": "Sri Lanka", "lat": 7.9, "lon": 80.8, "type": "island"},
]


def search_location(conn: sqlite3.Connection, query: str) -> list:
    """Search gazeteer + ports for a location name."""
    q = query.lower().strip()
    results = []

    # Try coordinate parsing (lat,lon)
    if "," in q:
        try:
            parts = q.split(",")
            lat = float(parts[0].strip())
            lon = float(parts[1].strip())
            results.append({"name": f"{lat:.4f}°N, {lon:.4f}°E", "lat": lat, "lon": lon, "type": "coordinate", "score": 100})
            return results[:5]
        except ValueError:
            pass

    # Gazeteer match
    for item in GAZETEER:
        if q in item["name"].lower():
            score = 100 if item["name"].lower() == q else 80 if item["name"].lower().startswith(q) else 60
            results.append({**item, "score": score})

    # Port match
    cur = conn.cursor()
    cur.execute("SELECT name, country, latitude, longitude FROM ports WHERE LOWER(name) LIKE ?", (f"%{q}%",))
    for row in cur.fetchall():
        results.append({"name": f"{row[0]}, {row[1]}", "lat": row[2], "lon": row[3], "type": "port", "score": 70})

    results.sort(key=lambda x: x["score"], reverse=True)

    # Cache result
    try:
        conn.execute("""INSERT OR REPLACE INTO location_search_cache (query, result_json, cached_at)
            VALUES (?,?,?)""", (query, json.dumps(results[:5]), _now()))
        conn.commit()
    except Exception:
        pass

    return results[:5]


# ──────────────────────────────────────────────────────────────
# SST GRID DATA (synthetic — feeds real map tiles)
# ──────────────────────────────────────────────────────────────

def generate_sst_geojson(lat_min=-10, lat_max=30, lon_min=50, lon_max=110,
                         res=1.0, hotspots=None) -> dict:
    """Generate SST point GeoJSON for heatmap layer."""
    hotspots = hotspots or []
    features = []
    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            base = 28.0 + 4.0 * math.cos((lat - 5) * math.pi / 30)
            noise = 1.5 * (math.sin(lat * 3.7 + lon * 2.1) + math.cos(lon * 4.3) * 0.5)
            warm1 = 2.5 if 10 < lat < 20 and 85 < lon < 95 else 0
            warm2 = 1.8 if 12 < lat < 22 and 55 < lon < 68 else 0
            # apply hotspot heat
            hw_add = 0.0
            for hs in hotspots:
                d2 = (lat - hs.get("latitude", 0))**2 + (lon - hs.get("longitude", 0))**2
                sev = {"EXTREME": 3.5, "HIGH": 2.5, "MODERATE": 1.5}.get(hs.get("severity", "NONE"), 0)
                hw_add = max(hw_add, sev * math.exp(-d2 / 4))
            sst = min(36.0, max(24.0, base + noise + warm1 + warm2 + hw_add))
            features.append({
                "type": "Feature",
                "properties": {"sst": round(sst, 2), "intensity": round((sst - 24) / 12, 3)},
                "geometry": {"type": "Point", "coordinates": [round(lon, 3), round(lat, 3)]}
            })
            lon = round(lon + res, 6)
        lat = round(lat + res, 6)
    return {"type": "FeatureCollection", "features": features, "total": len(features)}


def generate_wind_geojson(lat_min=-10, lat_max=30, lon_min=50, lon_max=110, res=2.0) -> dict:
    """Generate wind vector GeoJSON points."""
    features = []
    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            # SW monsoon dominant
            u = -4.0 + 2.0 * math.sin(lat * 0.3 + lon * 0.1)
            v = 3.0 + 1.5 * math.cos(lat * 0.4 - lon * 0.2)
            speed = math.sqrt(u**2 + v**2)
            direction = math.degrees(math.atan2(u, v)) % 360
            features.append({
                "type": "Feature",
                "properties": {"u": round(u, 2), "v": round(v, 2),
                               "speed": round(speed, 2), "direction": round(direction, 1)},
                "geometry": {"type": "Point", "coordinates": [round(lon, 3), round(lat, 3)]}
            })
            lon = round(lon + res, 6)
        lat = round(lat + res, 6)
    return {"type": "FeatureCollection", "features": features, "total": len(features)}


def generate_current_geojson(lat_min=-10, lat_max=30, lon_min=50, lon_max=110, res=2.0) -> dict:
    """Generate ocean current vector GeoJSON."""
    features = []
    lat = lat_min
    while lat <= lat_max:
        lon = lon_min
        while lon <= lon_max:
            # NE → SW Somali, circular gyre in BoB
            u = 0.3 * math.sin(lat * 0.2 + lon * 0.15)
            v = 0.2 * math.cos(lat * 0.3 - lon * 0.1)
            speed = math.sqrt(u**2 + v**2)
            features.append({
                "type": "Feature",
                "properties": {"u": round(u, 3), "v": round(v, 3), "speed": round(speed, 3)},
                "geometry": {"type": "Point", "coordinates": [round(lon, 3), round(lat, 3)]}
            })
            lon = round(lon + res, 6)
        lat = round(lat + res, 6)
    return {"type": "FeatureCollection", "features": features, "total": len(features)}


# ──────────────────────────────────────────────────────────────
# GeoJSON builder helpers
# ──────────────────────────────────────────────────────────────

def _to_geojson_points(rows: list, name_field: str, extra_props: list = None) -> dict:
    features = []
    for r in rows:
        props = {"name": r.get(name_field, "")}
        for p in (extra_props or []):
            if p in r:
                props[p] = r[p]
        features.append({
            "type": "Feature",
            "properties": props,
            "geometry": {
                "type": "Point",
                "coordinates": [r.get("longitude", 0), r.get("latitude", 0)]
            }
        })
    return {"type": "FeatureCollection", "features": features, "total": len(features)}


# ──────────────────────────────────────────────────────────────
# DISTANCE FROM COAST (approximate for Indian Ocean)
# ──────────────────────────────────────────────────────────────

# Simplified coastline sample points (India west+east, Sri Lanka, Bangladesh)
_COAST_POINTS = [
    (8.0, 77.5),(9.0, 78.1),(10.0, 79.5),(11.0, 79.8),(12.0, 80.2),(13.0, 80.3),
    (14.0, 80.2),(15.0, 80.1),(16.0, 80.3),(17.0, 82.3),(18.0, 83.5),(19.0, 85.0),
    (20.0, 86.5),(21.0, 86.9),(22.0, 88.3),  # East coast
    (22.0, 68.0),(21.0, 69.0),(20.0, 70.0),(18.0, 72.0),(16.0, 73.5),(14.0, 74.3),
    (12.0, 74.7),(10.0, 76.0),(8.0, 76.9),(7.0, 77.5),  # West coast
    (7.0, 79.9),(6.5, 81.2),(6.0, 80.5),(8.0, 81.0),  # Sri Lanka
    (21.5, 89.0),(22.0, 91.0),(22.5, 91.8),(23.0, 91.0),  # Bangladesh
]


def distance_from_coast_km(lat: float, lon: float) -> float:
    """Approximate minimum distance to coast in km."""
    min_d = float("inf")
    for clat, clon in _COAST_POINTS:
        d = haversine_km(lat, lon, clat, clon)
        if d < min_d:
            min_d = d
    return round(min_d, 1)


# ──────────────────────────────────────────────────────────────
# AGGREGATE GIS STATS
# ──────────────────────────────────────────────────────────────

def get_gis_stats(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    stats = {}
    counts = {
        "total_ports": "SELECT COUNT(*) FROM ports",
        "major_ports": "SELECT COUNT(*) FROM ports WHERE is_major=1",
        "shipping_routes": "SELECT COUNT(*) FROM shipping_routes",
        "marine_areas": "SELECT COUNT(*) FROM marine_areas",
        "mpa_zones": "SELECT COUNT(*) FROM marine_areas WHERE area_type='mpa'",
        "eez_zones": "SELECT COUNT(*) FROM marine_areas WHERE area_type='eez'",
        "bathymetry_points": "SELECT COUNT(*) FROM bathymetry_points",
        "ais_vessels": "SELECT COUNT(*) FROM ais_vessels",
        "gis_bookmarks": "SELECT COUNT(*) FROM gis_map_bookmarks",
        "drawn_features": "SELECT COUNT(*) FROM drawn_features",
        "measurements": "SELECT COUNT(*) FROM measurement_sessions",
    }
    for key, sql in counts.items():
        try:
            cur.execute(sql)
            stats[key] = cur.fetchone()[0]
        except Exception:
            stats[key] = 0
    return stats
