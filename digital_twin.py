"""
digital_twin.py
OceanVerse AI v4.0 — Phase 3 Digital Twin Backend
Pure-Python, no external dependencies beyond stdlib.

Provides:
  - Ocean volume data (15 depth layers, 3D grid)
  - Thermocline surface extraction
  - Particle streamline data (currents + wind)
  - ARGO float animation keyframes
  - Satellite orbit positions
  - Marine object positions (ships, buoys, gliders)
  - Bathymetry 3D terrain mesh data
  - Heatwave glow intensity field
  - Coral hotspot positions
  - Timeline frames (historical playback)
  - VR/WebXR scene descriptor
  - LOD level descriptors
  - All Flask API payloads (serializable dicts)
"""

import json
import math
import random
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────

EARTH_RADIUS = 5.0          # Three.js scene units
DEPTH_LEVELS_M = [0, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000, 1500]
DEPTH_SCALE = 0.0012        # metres → Three.js units for depth offset
REGION = dict(lat_min=-5, lat_max=28, lon_min=50, lon_max=108)
GRID_RES = 2.0              # degrees (coarser for 3D volume — smoother)

# SST colourmap anchors (fraction → RGB 0-1)
SST_CMAP = [
    (0.00, (0.04, 0.12, 0.36)),   # deep blue (cold)
    (0.25, (0.05, 0.33, 0.66)),   # ocean blue
    (0.50, (0.08, 0.74, 0.50)),   # cyan-green
    (0.75, (1.00, 0.60, 0.10)),   # orange
    (1.00, (0.71, 0.00, 0.10)),   # hot red
]

# ──────────────────────────────────────────────────────────────
# UTILITY
# ──────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()

def _lerp(a, b, t):
    return a + (b - a) * t

def _sst_color(fraction: float) -> tuple:
    """Map 0-1 fraction to RGB tuple via SST colourmap."""
    fraction = max(0.0, min(1.0, fraction))
    for i in range(len(SST_CMAP) - 1):
        f0, c0 = SST_CMAP[i]
        f1, c1 = SST_CMAP[i + 1]
        if f0 <= fraction <= f1:
            t = (fraction - f0) / (f1 - f0)
            return tuple(_lerp(c0[j], c1[j], t) for j in range(3))
    return SST_CMAP[-1][1]

def _lat_lon_to_xyz(lat: float, lon: float, radius: float = EARTH_RADIUS) -> dict:
    phi = (90 - lat) * math.pi / 180
    theta = (lon + 90) * math.pi / 180
    return {
        "x": -radius * math.sin(phi) * math.cos(theta),
        "y":  radius * math.cos(phi),
        "z":  radius * math.sin(phi) * math.sin(theta),
    }

def _sst_at(lat: float, lon: float, depth_idx: int = 0, hotspots=None) -> float:
    """Synthetic SST value (°C) at a lat/lon/depth."""
    base = 28.0 + 4.0 * math.cos((lat - 5) * math.pi / 30)
    noise = 1.5 * (math.sin(lat * 3.7 + lon * 2.1) + math.cos(lon * 4.3) * 0.5)
    depth_cool = DEPTH_LEVELS_M[min(depth_idx, len(DEPTH_LEVELS_M) - 1)] * 0.012
    hw_add = 0.0
    for hs in (hotspots or []):
        d2 = (lat - hs.get("latitude", 0)) ** 2 + (lon - hs.get("longitude", 0)) ** 2
        sev = {"EXTREME": 3.5, "HIGH": 2.5, "MODERATE": 1.5}.get(hs.get("severity", ""), 0)
        hw_add = max(hw_add, sev * math.exp(-d2 / 4))
    return max(4.0, min(36.0, base + noise - depth_cool + hw_add))


# ──────────────────────────────────────────────────────────────
# OCEAN VOLUME — 15 depth layers
# ──────────────────────────────────────────────────────────────

def get_ocean_volume(hotspots=None, depth_indices: list = None) -> dict:
    """
    Returns a 3D grid of voxel data for the ocean volume renderer.
    Each voxel: {x, y, z, r, g, b, a, sst, depth_m, lat, lon}
    Optimised: 2° grid, all 15 depth layers.
    """
    if depth_indices is None:
        depth_indices = list(range(len(DEPTH_LEVELS_M)))

    voxels = []
    lat = REGION["lat_min"]
    while lat <= REGION["lat_max"]:
        lon = REGION["lon_min"]
        while lon <= REGION["lon_max"]:
            for di in depth_indices:
                depth_m = DEPTH_LEVELS_M[di]
                sst = _sst_at(lat, lon, di, hotspots)
                frac = (sst - 24) / 12.0
                r, g, b = _sst_color(frac)
                # Depth offset: voxels sink inward (negative y on sphere surface)
                depth_offset = EARTH_RADIUS - depth_m * DEPTH_SCALE
                pos = _lat_lon_to_xyz(lat, lon, depth_offset)
                alpha = max(0.05, 0.7 - di * 0.04)
                voxels.append({
                    "x": round(pos["x"], 4),
                    "y": round(pos["y"], 4),
                    "z": round(pos["z"], 4),
                    "r": round(r, 3), "g": round(g, 3), "b": round(b, 3),
                    "a": round(alpha, 3),
                    "sst": round(sst, 2),
                    "depth_m": depth_m,
                    "depth_idx": di,
                    "lat": round(lat, 2),
                    "lon": round(lon, 2),
                })
            lon = round(lon + GRID_RES, 6)
        lat = round(lat + GRID_RES, 6)

    return {
        "type": "ocean_volume",
        "voxel_count": len(voxels),
        "depth_levels": len(depth_indices),
        "grid_resolution_deg": GRID_RES,
        "depth_levels_m": [DEPTH_LEVELS_M[i] for i in depth_indices],
        "voxels": voxels,
    }


def get_depth_layer(depth_idx: int = 0, hotspots=None) -> dict:
    """Single depth layer as a flat grid for efficient rendering."""
    depth_idx = max(0, min(depth_idx, len(DEPTH_LEVELS_M) - 1))
    depth_m = DEPTH_LEVELS_M[depth_idx]
    depth_offset = EARTH_RADIUS - depth_m * DEPTH_SCALE

    cells = []
    lat = REGION["lat_min"]
    while lat <= REGION["lat_max"]:
        lon = REGION["lon_min"]
        while lon <= REGION["lon_max"]:
            sst = _sst_at(lat, lon, depth_idx, hotspots)
            frac = (sst - 24) / 12.0
            r, g, b = _sst_color(frac)
            pos = _lat_lon_to_xyz(lat, lon, depth_offset)
            cells.append({
                "x": round(pos["x"], 4), "y": round(pos["y"], 4), "z": round(pos["z"], 4),
                "r": round(r, 3), "g": round(g, 3), "b": round(b, 3),
                "sst": round(sst, 2), "lat": round(lat, 2), "lon": round(lon, 2),
            })
            lon = round(lon + GRID_RES, 6)
        lat = round(lat + GRID_RES, 6)

    return {
        "type": "depth_layer",
        "depth_idx": depth_idx,
        "depth_m": depth_m,
        "cell_count": len(cells),
        "cells": cells,
    }


def get_thermocline_surface(hotspots=None) -> dict:
    """
    Thermocline surface: the depth at which temperature drops rapidly.
    Returns 3D surface mesh points (lat, lon, thermocline_depth_m, xyz).
    """
    points = []
    lat = REGION["lat_min"]
    while lat <= REGION["lat_max"]:
        lon = REGION["lon_min"]
        while lon <= REGION["lon_max"]:
            # Thermocline depth varies with SST: warmer surface → deeper thermocline
            sst_sfc = _sst_at(lat, lon, 0, hotspots)
            tc_depth = 40 + (sst_sfc - 24) * 5 + 10 * math.sin(lat * 0.3 + lon * 0.2)
            tc_depth = max(20, min(120, tc_depth))
            # Find closest depth level
            tc_idx = min(range(len(DEPTH_LEVELS_M)),
                        key=lambda i: abs(DEPTH_LEVELS_M[i] - tc_depth))
            depth_offset = EARTH_RADIUS - tc_depth * DEPTH_SCALE
            pos = _lat_lon_to_xyz(lat, lon, depth_offset)
            frac = (sst_sfc - 24) / 12.0
            r, g, b = _sst_color(frac)
            points.append({
                "x": round(pos["x"], 4), "y": round(pos["y"], 4), "z": round(pos["z"], 4),
                "thermocline_depth_m": round(tc_depth, 1),
                "sst_surface": round(sst_sfc, 2),
                "r": round(r, 3), "g": round(g, 3), "b": round(b, 3),
                "lat": round(lat, 2), "lon": round(lon, 2),
            })
            lon = round(lon + GRID_RES, 6)
        lat = round(lat + GRID_RES, 6)

    return {
        "type": "thermocline_surface",
        "point_count": len(points),
        "avg_depth_m": round(sum(p["thermocline_depth_m"] for p in points) / max(len(points), 1), 1),
        "points": points,
    }


# ──────────────────────────────────────────────────────────────
# PARTICLE DATA — Currents + Wind
# ──────────────────────────────────────────────────────────────

def get_particle_data(n_particles: int = 400, particle_type: str = "current") -> dict:
    """
    Returns particle system data for GPU instanced rendering.
    Each particle: {x, y, z, vx, vy, vz, speed, age, max_age, r, g, b}
    """
    random.seed(42 + hash(particle_type) % 1000)  # deterministic for testing
    particles = []

    for i in range(n_particles):
        lat = REGION["lat_min"] + random.random() * (REGION["lat_max"] - REGION["lat_min"])
        lon = REGION["lon_min"] + random.random() * (REGION["lon_max"] - REGION["lon_min"])

        if particle_type == "current":
            # Indian Ocean gyre / monsoon currents
            u = 0.3 * math.sin(lat * 0.25 + lon * 0.18) + 0.1 * random.gauss(0, 1)
            v = 0.2 * math.cos(lat * 0.3 - lon * 0.12) + 0.1 * random.gauss(0, 1)
            speed = math.sqrt(u * u + v * v)
            r, g, b = 0.13, 0.90, 1.0   # cyan
            radius = EARTH_RADIUS + 0.04
        else:  # wind
            # SW monsoon dominant
            u = -3.5 + 1.5 * math.sin(lat * 0.3 + lon * 0.15) + random.gauss(0, 0.5)
            v = 2.8 + 1.2 * math.cos(lat * 0.4 - lon * 0.2) + random.gauss(0, 0.4)
            speed = math.sqrt(u * u + v * v)
            r, g, b = 0.18, 0.90, 0.53  # green
            radius = EARTH_RADIUS + 0.45

        pos = _lat_lon_to_xyz(lat, lon, radius)
        # Velocity in spherical tangent plane (simplified)
        dlat = v * 0.001
        dlon = u * 0.001 / max(math.cos(math.radians(lat)), 0.1)
        pos2 = _lat_lon_to_xyz(lat + dlat, lon + dlon, radius)

        _max_age = random.randint(60, 120)
        particles.append({
            "x": round(pos["x"], 4), "y": round(pos["y"], 4), "z": round(pos["z"], 4),
            "vx": round(pos2["x"] - pos["x"], 6),
            "vy": round(pos2["y"] - pos["y"], 6),
            "vz": round(pos2["z"] - pos["z"], 6),
            "speed": round(speed, 3),
            "age": random.randint(0, _max_age),
            "max_age": _max_age,
            "r": round(r, 3), "g": round(g, 3), "b": round(b, 3),
            "lat": round(lat, 3), "lon": round(lon, 3),
        })

    return {
        "type": "particle_system",
        "particle_type": particle_type,
        "count": len(particles),
        "particles": particles,
        "color_hint": "#22e5ff" if particle_type == "current" else "#2ee6a8",
    }


# ──────────────────────────────────────────────────────────────
# ARGO FLOAT ANIMATION KEYFRAMES
# ──────────────────────────────────────────────────────────────

def get_argo_animation(floats: list = None, n_demo: int = 30) -> dict:
    """
    Returns animation keyframes for ARGO floats (dive → drift → rise cycle).
    Each float: {float_id, keyframes: [{t, x, y, z, depth_m}]}
    """
    if not floats:
        # Generate demo floats in the Bay of Bengal / Arabian Sea
        random.seed(99)
        floats = []
        for i in range(n_demo):
            lat = 5 + random.random() * 20
            lon = 60 + random.random() * 45
            floats.append({"float_id": f"ARGO_{5900000+i}", "latitude": lat, "longitude": lon})

    animated = []
    for f in floats[:50]:  # cap for performance
        lat0 = f.get("latitude", 10)
        lon0 = f.get("longitude", 75)
        keyframes = []
        # ARGO cycle: surface (0m) → dive to 2000m → drift → rise
        cycle_t = [0, 0.1, 0.5, 0.9, 1.0]
        depths  = [0, 200, 2000, 200, 0]
        for ci, (t, d) in enumerate(zip(cycle_t, depths)):
            # Slight horizontal drift during dive
            drift_lat = lat0 + 0.05 * math.sin(t * math.pi * 2 + lat0)
            drift_lon = lon0 + 0.08 * math.cos(t * math.pi + lon0)
            radius = EARTH_RADIUS - d * DEPTH_SCALE
            pos = _lat_lon_to_xyz(drift_lat, drift_lon, radius)
            keyframes.append({
                "t": t,
                "x": round(pos["x"], 4), "y": round(pos["y"], 4), "z": round(pos["z"], 4),
                "depth_m": d,
                "lat": round(drift_lat, 4), "lon": round(drift_lon, 4),
            })
        animated.append({
            "float_id": f.get("float_id", f"ARGO_{id(f)}"),
            "surface_lat": lat0, "surface_lon": lon0,
            "keyframes": keyframes,
        })

    return {
        "type": "argo_animation",
        "float_count": len(animated),
        "cycle_duration_hours": 240,   # 10-day ARGO cycle
        "floats": animated,
    }


# ──────────────────────────────────────────────────────────────
# SATELLITE ORBITS
# ──────────────────────────────────────────────────────────────

SATELLITES = [
    {"id": "JASON-3",   "altitude_km": 1336, "inclination": 66.0,  "period_min": 112, "color": "#22e5ff"},
    {"id": "SENTINEL-6","altitude_km": 1336, "inclination": 66.0,  "period_min": 112, "color": "#2ee6a8"},
    {"id": "SARAL",     "altitude_km": 781,  "inclination": 98.5,  "period_min": 100, "color": "#ffb020"},
    {"id": "CRYOSAT-2", "altitude_km": 717,  "inclination": 92.0,  "period_min": 99,  "color": "#9b59b6"},
    {"id": "TERRA",     "altitude_km": 705,  "inclination": 98.2,  "period_min": 99,  "color": "#e74c3c"},
    {"id": "AQUA",      "altitude_km": 705,  "inclination": 98.0,  "period_min": 99,  "color": "#3498db"},
]


def get_satellite_orbits(n_frames: int = 60) -> dict:
    """
    Returns orbit path points for each satellite (n_frames positions per orbit).
    Each orbit is pre-computed for the animation loop.
    """
    orbits = []
    for sat in SATELLITES:
        alt_scene = EARTH_RADIUS + sat["altitude_km"] * 0.0003  # scale km → scene units
        inc = math.radians(sat["inclination"])
        phase_offset = random.uniform(0, math.pi * 2)
        path = []
        for fi in range(n_frames + 1):
            t = fi / n_frames  # 0-1 over one full orbit
            angle = t * 2 * math.pi + phase_offset
            # Simplified circular orbit in inclined plane
            x = alt_scene * math.cos(angle)
            y = alt_scene * math.sin(angle) * math.sin(inc)
            z = alt_scene * math.sin(angle) * math.cos(inc)
            path.append({"t": round(t, 4), "x": round(x, 4), "y": round(y, 4), "z": round(z, 4)})
        orbits.append({
            "id": sat["id"],
            "altitude_km": sat["altitude_km"],
            "inclination_deg": sat["inclination"],
            "period_min": sat["period_min"],
            "color": sat["color"],
            "path": path,
        })
    return {"type": "satellite_orbits", "satellite_count": len(orbits), "orbits": orbits}


# ──────────────────────────────────────────────────────────────
# MARINE OBJECTS (Ships, Buoys, Gliders)
# ──────────────────────────────────────────────────────────────

MARINE_OBJECTS_DEMO = [
    # Ships
    {"id": "ship_01", "type": "ship",   "lat": 13.5, "lon": 82.0, "heading": 180, "speed_kn": 14, "name": "MV Pacific Voyager"},
    {"id": "ship_02", "type": "ship",   "lat": 18.0, "lon": 66.0, "heading": 270, "speed_kn": 12, "name": "MT Arabian Star"},
    {"id": "ship_03", "type": "ship",   "lat": 22.0, "lon": 89.5, "heading": 120, "speed_kn": 11, "name": "MV Bengal Express"},
    {"id": "ship_04", "type": "ship",   "lat": 7.5,  "lon": 80.5, "heading": 300, "speed_kn": 10, "name": "MV Sri Lanka Pride"},
    # Buoys
    {"id": "buoy_01", "type": "buoy",   "lat": 15.0, "lon": 70.0, "name": "OMNI-Buoy-A", "sst": 29.1},
    {"id": "buoy_02", "type": "buoy",   "lat": 10.0, "lon": 85.0, "name": "OMNI-Buoy-B", "sst": 30.4},
    {"id": "buoy_03", "type": "buoy",   "lat": 8.0,  "lon": 75.0, "name": "RAMA-Buoy-01", "sst": 28.8},
    {"id": "buoy_04", "type": "buoy",   "lat": 20.0, "lon": 60.0, "name": "ATLAS-Buoy-N", "sst": 27.3},
    # Gliders
    {"id": "glider_01","type":"glider", "lat": 12.0, "lon": 87.0, "name": "OceanGlider-Δ1", "depth_m": 500},
    {"id": "glider_02","type":"glider", "lat": 16.0, "lon": 63.0, "name": "OceanGlider-Δ2", "depth_m": 200},
    # Underwater cables (static)
    {"id": "cable_01","type":"cable",   "lat": 6.9,  "lon": 79.9, "name": "SEA-ME-WE 4"},
    {"id": "cable_02","type":"cable",   "lat": 1.3,  "lon": 103.8,"name": "APCN-2"},
]

_OBJECT_RADII = {"ship": EARTH_RADIUS + 0.05, "buoy": EARTH_RADIUS + 0.06,
                 "glider": EARTH_RADIUS - 0.5 * 0.0012, "cable": EARTH_RADIUS + 0.01}
_OBJECT_COLORS = {"ship": "#3498db", "buoy": "#f1c40f", "glider": "#8e44ad", "cable": "#e74c3c"}


def get_marine_objects() -> dict:
    """Returns all marine objects with 3D positions for Three.js instanced rendering."""
    objects = []
    for obj in MARINE_OBJECTS_DEMO:
        obj_type = obj.get("type", "buoy")
        radius = _OBJECT_RADII.get(obj_type, EARTH_RADIUS + 0.05)
        # Gliders are submerged
        if obj_type == "glider":
            radius = EARTH_RADIUS - obj.get("depth_m", 100) * DEPTH_SCALE
        pos = _lat_lon_to_xyz(obj["lat"], obj["lon"], radius)
        objects.append({
            **obj,
            "x": round(pos["x"], 4), "y": round(pos["y"], 4), "z": round(pos["z"], 4),
            "color": _OBJECT_COLORS.get(obj_type, "#ffffff"),
        })

    by_type = {}
    for obj in objects:
        t = obj["type"]
        by_type.setdefault(t, []).append(obj)

    return {
        "type": "marine_objects",
        "total": len(objects),
        "by_type": {k: len(v) for k, v in by_type.items()},
        "objects": objects,
    }


# ──────────────────────────────────────────────────────────────
# BATHYMETRY 3D TERRAIN
# ──────────────────────────────────────────────────────────────

BATHYMETRY_FEATURES = [
    # (lat, lon, depth_m, name, type)
    (0.0,  90.0, -5000, "90°E Ridge North",    "ridge"),
    (-5.0, 90.0, -2000, "90°E Ridge South",    "ridge"),
    (5.0,  104.0,-7200, "Sunda Trench (N)",    "trench"),
    (-5.0, 102.0,-6800, "Sunda Trench (S)",    "trench"),
    (10.0, 57.0, -1500, "Carlsberg Ridge",     "ridge"),
    (-10.0,68.0, -4500, "Chagos-Laccadive",   "plateau"),
    (13.0, 90.0, -4800, "BoB Abyssal Plain",  "abyssal"),
    (17.0, 65.0, -3800, "Arabia Abyssal",     "abyssal"),
    (20.0, 70.0, -150,  "Lakshadweep Shelf",  "shelf"),
    (13.0, 80.0, -80,   "Coromandel Shelf",   "shelf"),
    (-25.0,73.0,-5500,  "C. Indian Ridge",    "ridge"),
    (-15.0,72.0,-3500,  "Mauritius Plateau",  "plateau"),
]

_DEPTH_COLORS = {
    "ridge":   (0.10, 0.28, 0.62),
    "trench":  (0.04, 0.09, 0.35),
    "abyssal": (0.07, 0.17, 0.45),
    "shelf":   (0.26, 0.52, 0.82),
    "plateau": (0.12, 0.36, 0.58),
}


def get_bathymetry_terrain(resolution: str = "medium") -> dict:
    """
    Returns 3D terrain mesh data for the ocean floor.
    Points are mapped to sphere surface minus depth offset.
    """
    res = {"low": 4.0, "medium": 2.0, "high": 1.0}.get(resolution, 2.0)
    terrain_points = []

    # Grid-based terrain
    lat = REGION["lat_min"]
    while lat <= REGION["lat_max"]:
        lon = REGION["lon_min"]
        while lon <= REGION["lon_max"]:
            # Find nearest bathymetry feature, else use baseline depth
            depth_m = -3000 - 500 * math.sin(lat * 0.4 + lon * 0.3)  # baseline
            feat_type = "abyssal"
            for flat, flon, fdepth, fname, ftype in BATHYMETRY_FEATURES:
                d2 = (lat - flat) ** 2 + (lon - flon) ** 2
                if d2 < 16:  # within ~4 degrees
                    blend = math.exp(-d2 / 6)
                    depth_m = depth_m * (1 - blend) + fdepth * blend
                    if blend > 0.4:
                        feat_type = ftype
            # Shallow coastal
            coast_dist = min(abs(lat - 20), abs(lat - 7), abs(lon - 68), abs(lon - 100))
            if coast_dist < 3:
                depth_m = max(depth_m, -200)
                feat_type = "shelf"

            radius = EARTH_RADIUS + depth_m * DEPTH_SCALE
            pos = _lat_lon_to_xyz(lat, lon, radius)
            r, g, b = _DEPTH_COLORS.get(feat_type, (0.1, 0.2, 0.5))
            terrain_points.append({
                "x": round(pos["x"], 4), "y": round(pos["y"], 4), "z": round(pos["z"], 4),
                "depth_m": round(depth_m, 0),
                "feature_type": feat_type,
                "r": round(r, 3), "g": round(g, 3), "b": round(b, 3),
                "lat": round(lat, 2), "lon": round(lon, 2),
            })
            lon = round(lon + res, 6)
        lat = round(lat + res, 6)

    # Named features
    features = []
    for flat, flon, fdepth, fname, ftype in BATHYMETRY_FEATURES:
        radius = EARTH_RADIUS + fdepth * DEPTH_SCALE
        pos = _lat_lon_to_xyz(flat, flon, radius)
        r, g, b = _DEPTH_COLORS.get(ftype, (0.1, 0.2, 0.5))
        features.append({
            "name": fname, "type": ftype, "depth_m": fdepth,
            "lat": flat, "lon": flon,
            "x": round(pos["x"], 4), "y": round(pos["y"], 4), "z": round(pos["z"], 4),
            "r": round(r, 3), "g": round(g, 3), "b": round(b, 3),
        })

    return {
        "type": "bathymetry_terrain",
        "resolution": resolution,
        "point_count": len(terrain_points),
        "feature_count": len(features),
        "deepest_m": min(p["depth_m"] for p in terrain_points),
        "shallowest_m": max(p["depth_m"] for p in terrain_points),
        "terrain": terrain_points,
        "named_features": features,
    }


# ──────────────────────────────────────────────────────────────
# HEATWAVE GLOW FIELD
# ──────────────────────────────────────────────────────────────

def get_heatwave_glow(hotspots=None) -> dict:
    """
    Returns glow intensity field for ocean surface heatwave visualisation.
    Each point: {x, y, z, intensity, severity, r, g, b}
    """
    hotspots = hotspots or []
    points = []
    lat = REGION["lat_min"]
    while lat <= REGION["lat_max"]:
        lon = REGION["lon_min"]
        while lon <= REGION["lon_max"]:
            intensity = 0.0
            severity = "NONE"
            for hs in hotspots:
                d2 = (lat - hs.get("latitude", 0)) ** 2 + (lon - hs.get("longitude", 0)) ** 2
                sev = {"EXTREME": 1.0, "HIGH": 0.7, "MODERATE": 0.4}.get(hs.get("severity", ""), 0)
                contrib = sev * math.exp(-d2 / 9)
                if contrib > intensity:
                    intensity = contrib
                    severity = hs.get("severity", "NONE")

            if intensity > 0.05:
                r = min(1.0, 0.8 + intensity * 0.2)
                g = max(0.0, 0.15 - intensity * 0.1)
                b = 0.0
                pos = _lat_lon_to_xyz(lat, lon, EARTH_RADIUS + 0.03)
                points.append({
                    "x": round(pos["x"], 4), "y": round(pos["y"], 4), "z": round(pos["z"], 4),
                    "intensity": round(intensity, 3),
                    "severity": severity,
                    "r": round(r, 3), "g": round(g, 3), "b": round(b, 3),
                    "lat": round(lat, 2), "lon": round(lon, 2),
                })
            lon = round(lon + GRID_RES, 6)
        lat = round(lat + GRID_RES, 6)

    # Also include raw hotspot markers
    hotspot_markers = []
    for hs in hotspots:
        pos = _lat_lon_to_xyz(hs.get("latitude", 0), hs.get("longitude", 0), EARTH_RADIUS + 0.05)
        hotspot_markers.append({
            **hs,
            "x": round(pos["x"], 4), "y": round(pos["y"], 4), "z": round(pos["z"], 4),
        })

    return {
        "type": "heatwave_glow",
        "point_count": len(points),
        "hotspot_count": len(hotspot_markers),
        "points": points,
        "hotspot_markers": hotspot_markers,
    }


# ──────────────────────────────────────────────────────────────
# CORAL HOTSPOTS
# ──────────────────────────────────────────────────────────────

CORAL_SITES = [
    {"name": "Lakshadweep Atolls",   "lat": 10.5,  "lon": 72.6,  "health": "bleached",  "sst_stress": 2.1},
    {"name": "Gulf of Mannar",        "lat": 8.9,   "lon": 78.8,  "health": "moderate",  "sst_stress": 1.2},
    {"name": "Andaman Reefs",         "lat": 12.2,  "lon": 92.9,  "health": "healthy",   "sst_stress": 0.4},
    {"name": "Nicobar Reefs",         "lat": 8.5,   "lon": 93.5,  "health": "moderate",  "sst_stress": 1.0},
    {"name": "Maldives N. Atolls",    "lat": 6.5,   "lon": 72.8,  "health": "bleached",  "sst_stress": 2.5},
    {"name": "Maldives S. Atolls",    "lat": 1.5,   "lon": 73.2,  "health": "moderate",  "sst_stress": 1.4},
    {"name": "Sri Lanka (E)",         "lat": 8.2,   "lon": 81.5,  "health": "stressed",  "sst_stress": 1.7},
    {"name": "Chagos Reefs",          "lat": -6.3,  "lon": 71.9,  "health": "healthy",   "sst_stress": 0.2},
    {"name": "Seychelles Banks",      "lat": -4.6,  "lon": 55.5,  "health": "moderate",  "sst_stress": 0.9},
    {"name": "Mozambique Channel",    "lat": -18.0, "lon": 44.0,  "health": "stressed",  "sst_stress": 1.5},
]

_CORAL_COLORS = {
    "healthy":  (0.1, 0.8, 0.3),
    "moderate": (1.0, 0.75, 0.0),
    "stressed": (1.0, 0.4, 0.0),
    "bleached": (0.95, 0.15, 0.2),
}


def get_coral_hotspots() -> dict:
    """Returns coral reef hotspot positions with health status for 3D glow rendering."""
    sites = []
    for site in CORAL_SITES:
        pos = _lat_lon_to_xyz(site["lat"], site["lon"], EARTH_RADIUS + 0.025)
        r, g, b = _CORAL_COLORS.get(site["health"], (1, 1, 1))
        intensity = min(1.0, site["sst_stress"] / 2.5)
        sites.append({
            **site,
            "x": round(pos["x"], 4), "y": round(pos["y"], 4), "z": round(pos["z"], 4),
            "r": round(r, 3), "g": round(g, 3), "b": round(b, 3),
            "intensity": round(intensity, 3),
        })

    return {
        "type": "coral_hotspots",
        "site_count": len(sites),
        "bleached": sum(1 for s in sites if s["health"] == "bleached"),
        "stressed": sum(1 for s in sites if s["health"] == "stressed"),
        "healthy":  sum(1 for s in sites if s["health"] == "healthy"),
        "sites": sites,
    }


# ──────────────────────────────────────────────────────────────
# DIGITAL TWIN TIMELINE
# ──────────────────────────────────────────────────────────────

def get_timeline_frames(n_years: int = 10, hotspots=None) -> dict:
    """
    Returns timeline frames for historical playback (annual snapshots).
    Each frame: {year, mean_sst, heatwave_count, anomaly, trend}
    """
    base_year = 2015
    frames = []
    trend = 0.0
    for i in range(n_years):
        year = base_year + i
        trend += 0.04 + 0.01 * math.sin(i * 0.8)
        mean_sst = 28.2 + trend + 0.3 * math.sin(i * math.pi / 5)
        hw_count = max(0, int(2 + trend * 3 + random.gauss(0, 1)))
        anomaly = mean_sst - 28.2
        frames.append({
            "year": year,
            "mean_sst": round(mean_sst, 2),
            "heatwave_count": hw_count,
            "anomaly_c": round(anomaly, 2),
            "trend_c_per_decade": round(trend * (10 / (i + 1)), 2),
            "label": f"{year}",
        })

    # Future projections
    for i in range(3):
        year = base_year + n_years + i
        proj_sst = frames[-1]["mean_sst"] + 0.1 * (i + 1)
        frames.append({
            "year": year, "mean_sst": round(proj_sst, 2),
            "heatwave_count": int(frames[-1]["heatwave_count"] * 1.15),
            "anomaly_c": round(proj_sst - 28.2, 2),
            "trend_c_per_decade": round(0.4 + 0.1 * i, 2),
            "label": f"{year} (proj)", "projected": True,
        })

    return {
        "type": "timeline",
        "frame_count": len(frames),
        "year_start": frames[0]["year"],
        "year_end": frames[-1]["year"],
        "frames": frames,
    }


# ──────────────────────────────────────────────────────────────
# AI EXPLANATION HUD
# ──────────────────────────────────────────────────────────────

_EXPLANATION_TEMPLATES = {
    "high_sst": "SST elevated {delta:.1f}°C above climatology. Primary driver: {driver}. "
                "Confidence: {conf:.0f}%. SHAP top features: SST-lag1 ({w1:.2f}), wind ({w2:.2f}).",
    "heatwave": "Marine heatwave detected. Severity: {severity}. Area: ~{area} km². "
                "Duration: {days} days. Bleaching risk: {risk}.",
    "normal":   "SST within normal range ({sst:.1f}°C). No anomaly detected. "
                "Model confidence: {conf:.0f}%.",
}


def get_ai_explanation_hud(lat: float, lon: float, hotspots=None) -> dict:
    """Returns AI explanation popup data for a clicked grid cell."""
    sst = _sst_at(lat, lon, 0, hotspots)
    climatology = 28.2 + 2 * math.cos((lat - 5) * math.pi / 30)
    delta = sst - climatology
    conf = 75 + 15 * math.cos(lat * 0.2 + lon * 0.1)

    # Check heatwave
    in_hw = False
    severity = "NONE"
    for hs in (hotspots or []):
        d2 = (lat - hs.get("latitude", 0)) ** 2 + (lon - hs.get("longitude", 0)) ** 2
        if d2 < 4:
            in_hw = True
            severity = hs.get("severity", "MODERATE")
            break

    if in_hw:
        tpl = _EXPLANATION_TEMPLATES["heatwave"]
        text = tpl.format(severity=severity, area=random.randint(5000, 80000),
                          days=random.randint(5, 30), risk="HIGH" if severity=="EXTREME" else "MODERATE")
    elif delta > 1.5:
        drivers = ["reduced upwelling", "suppressed monsoon winds", "anomalous warm advection"]
        text = _EXPLANATION_TEMPLATES["high_sst"].format(
            delta=delta, driver=random.choice(drivers), conf=conf,
            w1=0.3 + 0.1 * abs(math.sin(lat)), w2=0.2 + 0.05 * abs(math.cos(lon)))
    else:
        text = _EXPLANATION_TEMPLATES["normal"].format(sst=sst, conf=conf)

    shap_values = {
        "sst_lag1": round(0.32 + 0.05 * math.sin(lat), 3),
        "wind_speed": round(0.21 + 0.03 * math.cos(lon), 3),
        "sla": round(0.15, 3),
        "sss": round(0.12 + 0.02 * math.sin(lon), 3),
        "mld": round(0.09, 3),
        "other": round(0.11, 3),
    }

    return {
        "type": "ai_explanation",
        "lat": lat, "lon": lon,
        "sst": round(sst, 2),
        "climatology": round(climatology, 2),
        "anomaly_c": round(delta, 2),
        "confidence_pct": round(conf, 1),
        "heatwave_status": severity,
        "explanation_text": text,
        "shap_values": shap_values,
        "model_version": "OceanTransformer v3",
    }


# ──────────────────────────────────────────────────────────────
# WAVE SURFACE
# ──────────────────────────────────────────────────────────────

def get_wave_surface(n_waves: int = 8) -> dict:
    """Returns wave parameters for animated GLSL wave shader."""
    waves = []
    for i in range(n_waves):
        angle = i * 2 * math.pi / n_waves + 0.3
        waves.append({
            "amplitude": round(0.004 + 0.002 * math.sin(i * 1.3), 4),
            "frequency": round(2.5 + i * 0.8, 3),
            "speed": round(0.5 + i * 0.15, 3),
            "dir_x": round(math.cos(angle), 4),
            "dir_z": round(math.sin(angle), 4),
            "phase": round(i * 0.9, 3),
        })
    return {
        "type": "wave_surface",
        "wave_count": n_waves,
        "waves": waves,
        "glsl_snippet": "// Use waves[i].amplitude * sin(dot(pos.xz, dir) * frequency + time * speed + phase)",
    }


# ──────────────────────────────────────────────────────────────
# SCENE DESCRIPTOR (full twin state for frontend init)
# ──────────────────────────────────────────────────────────────

def get_scene_descriptor(hotspots=None) -> dict:
    """
    Master descriptor sent on page load — tells the frontend
    what layers to build and their parameters.
    """
    return {
        "type": "scene_descriptor",
        "version": "4.0",
        "earth_radius": EARTH_RADIUS,
        "depth_levels": DEPTH_LEVELS_M,
        "depth_scale": DEPTH_SCALE,
        "region": REGION,
        "grid_resolution_deg": GRID_RES,
        "layers": {
            "ocean_volume":    {"enabled": True,  "lod": "medium"},
            "thermocline":     {"enabled": True,  "lod": "medium"},
            "current_particles":{"enabled":True,  "count": 400},
            "wind_particles":  {"enabled": True,  "count": 300},
            "wave_surface":    {"enabled": True,  "n_waves": 8},
            "argo_animation":  {"enabled": True,  "n_floats": 30},
            "satellite_orbits":{"enabled": True},
            "marine_objects":  {"enabled": True},
            "bathymetry":      {"enabled": True,  "resolution": "medium"},
            "heatwave_glow":   {"enabled": True},
            "coral_hotspots":  {"enabled": True},
        },
        "camera": {
            "initial_position": {"x": 0, "y": 0, "z": 14},
            "fly_sequence": [
                {"label": "Earth",          "lat": 0,    "lon": 0,   "dist": 14},
                {"label": "Asia",           "lat": 30,   "lon": 80,  "dist": 11},
                {"label": "India",          "lat": 20,   "lon": 78,  "dist": 9},
                {"label": "N. Indian Ocean","lat": 15,   "lon": 82,  "dist": 8},
                {"label": "Bay of Bengal",  "lat": 13,   "lon": 88,  "dist": 7.5},
                {"label": "0.25° Grid",     "lat": 12.5, "lon": 82.5,"dist": 6.9},
            ],
        },
        "vr": {"enabled": False, "xr_session_mode": "immersive-vr"},
        "performance": {
            "lod_levels": 3,
            "max_particles": 600,
            "target_fps": 60,
            "lazy_load_distance": 7.5,
        },
    }


# ──────────────────────────────────────────────────────────────
# VR / WebXR DESCRIPTOR
# ──────────────────────────────────────────────────────────────

def get_vr_descriptor() -> dict:
    return {
        "type": "vr_descriptor",
        "xr_session_mode": "immersive-vr",
        "xr_features": ["local-floor", "hand-tracking", "hit-test"],
        "controllers": {
            "left":  {"action": "depth_dive", "gesture": "trigger"},
            "right": {"action": "layer_toggle","gesture": "grip"},
        },
        "scene_scale": 0.01,  # metres per scene unit
        "notes": "WebXR hooks ready. Activate with navigator.xr.requestSession('immersive-vr').",
        "webxr_polyfill": "https://cdn.jsdelivr.net/npm/webxr-polyfill/build/webxr-polyfill.min.js",
    }


# ──────────────────────────────────────────────────────────────
# LOD DESCRIPTOR
# ──────────────────────────────────────────────────────────────

def get_lod_descriptor() -> dict:
    return {
        "type": "lod_descriptor",
        "levels": [
            {"level": 0, "label": "Ultra",  "grid_res_deg": 0.5, "particles": 600, "camera_dist_max": 7},
            {"level": 1, "label": "High",   "grid_res_deg": 1.0, "particles": 400, "camera_dist_max": 10},
            {"level": 2, "label": "Medium", "grid_res_deg": 2.0, "particles": 200, "camera_dist_max": 14},
            {"level": 3, "label": "Low",    "grid_res_deg": 4.0, "particles": 100, "camera_dist_max": 40},
        ],
        "auto_lod": True,
        "gpu_instancing": True,
    }
