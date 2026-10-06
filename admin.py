"""
admin.py
North Indian Ocean Intelligence Platform — Phase 3
Admin Dashboard Backend (Section P) + User Dashboard Backend (Section Q).
"""

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger("ocean_platform")

# ──────────────────────────────────────────────────────────────
# SIMPLE ADMIN AUTH  (token-based, no external dependencies)
# ──────────────────────────────────────────────────────────────

_ADMIN_TOKEN = hashlib.sha256(b"sih_ocean_admin_2025").hexdigest()


def verify_admin_token(token: str) -> bool:
    """Token must equal sha256("sih_ocean_admin_2025")."""
    return token == _ADMIN_TOKEN


# ──────────────────────────────────────────────────────────────
# ADMIN SQLite TABLES
# ──────────────────────────────────────────────────────────────

SCHEMA_ADMIN = """
CREATE TABLE IF NOT EXISTS admin_uploads (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    filename     TEXT,
    filepath     TEXT,
    size_bytes   INTEGER,
    dataset_type TEXT,
    uploaded_at  TEXT,
    uploaded_by  TEXT DEFAULT 'admin'
);

CREATE TABLE IF NOT EXISTS admin_actions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    action      TEXT,
    detail      TEXT,
    performed_at TEXT,
    performed_by TEXT DEFAULT 'admin'
);

CREATE TABLE IF NOT EXISTS model_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    version_name    TEXT,
    checkpoint_path TEXT,
    val_loss        REAL,
    epochs          INTEGER,
    created_at      TEXT
);

CREATE TABLE IF NOT EXISTS dataset_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_name TEXT,
    action       TEXT,
    filepath     TEXT,
    modified_at  TEXT
);

CREATE TABLE IF NOT EXISTS prediction_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_date TEXT,
    forecast_days   INTEGER,
    model_version   TEXT,
    confidence      REAL,
    created_at      TEXT
);
"""


def init_admin_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_ADMIN)
    conn.commit()
    logger.info("Admin SQLite tables created")


# ──────────────────────────────────────────────────────────────
# ADMIN ACTIONS
# ──────────────────────────────────────────────────────────────

def admin_upload_dataset(conn: sqlite3.Connection, filename: str,
                          filepath: str, dataset_type: str) -> dict:
    size = os.path.getsize(filepath) if os.path.exists(filepath) else 0
    cur  = conn.cursor()
    cur.execute("""INSERT INTO admin_uploads
        (filename, filepath, size_bytes, dataset_type, uploaded_at)
        VALUES (?,?,?,?,?)""",
        (filename, filepath, size, dataset_type,
         datetime.now(timezone.utc).isoformat()))
    cur.execute("""INSERT INTO admin_actions (action, detail, performed_at)
        VALUES ('upload', ?, ?)""",
        (f"Uploaded {filename}", datetime.now(timezone.utc).isoformat()))
    cur.execute("""INSERT INTO dataset_history (dataset_name, action, filepath, modified_at)
        VALUES (?,?,?,?)""",
        (filename, "upload", filepath, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    logger.info(f"Admin upload registered: {filename}")
    return {"status": "uploaded", "filename": filename,
            "size_bytes": size, "dataset_type": dataset_type}


def admin_delete_dataset(conn: sqlite3.Connection, filename: str,
                          filepath: str) -> dict:
    if os.path.exists(filepath):
        os.remove(filepath)
        status = "deleted"
    else:
        status = "not_found"
    cur = conn.cursor()
    cur.execute("""INSERT INTO admin_actions (action, detail, performed_at)
        VALUES ('delete', ?, ?)""",
        (f"Deleted {filename}", datetime.now(timezone.utc).isoformat()))
    cur.execute("""INSERT INTO dataset_history (dataset_name, action, filepath, modified_at)
        VALUES (?,?,?,?)""",
        (filename, "delete", filepath, datetime.now(timezone.utc).isoformat()))
    conn.commit()
    logger.info(f"Admin delete: {filename} → {status}")
    return {"status": status, "filename": filename}


def admin_refresh_database(conn: sqlite3.Connection) -> dict:
    cur = conn.cursor()
    cur.execute("""INSERT INTO admin_actions (action, detail, performed_at)
        VALUES ('refresh_db', 'Manual database refresh triggered',?)""",
        (datetime.now(timezone.utc).isoformat(),))
    conn.commit()
    logger.info("Admin: database refresh triggered")
    return {"status": "refreshed", "timestamp": datetime.now(timezone.utc).isoformat()}


def admin_retrain_model(conn: sqlite3.Connection, cfg: dict) -> dict:
    """Trigger model retraining (sets resume=False for fresh run)."""
    cur = conn.cursor()
    cur.execute("""INSERT INTO admin_actions (action, detail, performed_at)
        VALUES ('retrain', 'Model retrain triggered by admin', ?)""",
        (datetime.now(timezone.utc).isoformat(),))
    conn.commit()
    logger.info("Admin: retrain triggered")
    return {
        "status":  "retrain_queued",
        "message": "Retrain has been queued. Run `python app.py --retrain` to execute.",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def get_admin_logs(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM admin_actions ORDER BY id DESC LIMIT ?", (limit,))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_model_history(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM model_history ORDER BY id DESC")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_dataset_history(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.cursor()
    cur.execute("SELECT * FROM dataset_history ORDER BY id DESC")
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_prediction_history(conn: sqlite3.Connection, limit: int = 50) -> list[dict]:
    cur = conn.cursor()
    # Pull from Phase 3 predictions table
    try:
        cur.execute("""SELECT * FROM predictions ORDER BY id DESC LIMIT ?""", (limit,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        return []


def get_training_history(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.cursor()
    try:
        cur.execute("SELECT * FROM training_runs ORDER BY id DESC")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        return []


def get_platform_statistics(conn: sqlite3.Connection) -> dict:
    cur  = conn.cursor()
    stats = {}

    for table in ["datasets", "harmonized_dataset", "predictions",
                  "heatwave_history", "admin_uploads"]:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            stats[f"{table}_count"] = cur.fetchone()[0]
        except Exception:
            stats[f"{table}_count"] = 0

    try:
        cur.execute("SELECT AVG(rmse) FROM predictions WHERE rmse IS NOT NULL")
        stats["avg_prediction_rmse"] = cur.fetchone()[0]
    except Exception:
        stats["avg_prediction_rmse"] = None

    try:
        cur.execute("SELECT COUNT(*) FROM heatwave_history WHERE severity IN ('HIGH','EXTREME')")
        stats["high_extreme_heatwaves"] = cur.fetchone()[0]
    except Exception:
        stats["high_extreme_heatwaves"] = 0

    return stats


# ──────────────────────────────────────────────────────────────
# USER DASHBOARD  (Section Q)
# ──────────────────────────────────────────────────────────────

def get_latest_prediction(pred_ds, hotspots: list, uncertainty: dict) -> dict:
    """User-facing: most recent prediction summary."""
    import numpy as np
    if pred_ds is None:
        return {"status": "no_prediction"}
    sst_latest = pred_ds["thetao_pred"].isel(time=-1, depth=0).values
    return {
        "timestamp":          str(pred_ds["time"].values[-1]),
        "surface_temp_mean":  round(float(np.nanmean(sst_latest)), 3),
        "surface_temp_max":   round(float(np.nanmax(sst_latest)), 3),
        "heatwave_count":     len(hotspots),
        "overall_confidence": uncertainty.get("overall_confidence", 0),
        "depths_predicted":   pred_ds.sizes.get("depth", 0),
    }


def get_heatwave_alert(hotspots: list) -> dict:
    """User-facing: current heatwave alert level."""
    if not hotspots:
        return {"alert_level": "NONE", "hotspots": 0, "message": "No active marine heatwaves."}
    worst = max(hotspots, key=lambda h: h["sst_anomaly"])
    return {
        "alert_level": worst["severity"],
        "hotspots":    len(hotspots),
        "worst": {
            "lat":      worst["latitude"],
            "lon":      worst["longitude"],
            "anomaly":  worst["sst_anomaly"],
            "severity": worst["severity"],
        },
        "message": f"{len(hotspots)} heatwave zone(s) detected. Worst: {worst['severity']} at "
                   f"{worst['latitude']:.2f}N {worst['longitude']:.2f}E.",
    }


def get_ocean_summary(pred_ds, surface_ds) -> dict:
    """User-facing: broad ocean state summary."""
    import numpy as np
    summary: dict = {}
    if pred_ds is not None:
        sst_all = pred_ds["thetao_pred"].isel(depth=0).values
        summary["predicted_sst_mean"] = round(float(np.nanmean(sst_all)), 3)
        summary["predicted_sst_trend"] = round(float(
            np.nanmean(sst_all[-3:]) - np.nanmean(sst_all[:3])), 3)
    if surface_ds is not None and "sst" in surface_ds.data_vars:
        obs_sst = surface_ds["sst"].values
        summary["observed_sst_mean"] = round(float(np.nanmean(obs_sst)), 3)
    if surface_ds is not None and "wind_u" in surface_ds.data_vars:
        wu = surface_ds["wind_u"].values
        wv = surface_ds["wind_v"].values if "wind_v" in surface_ds.data_vars else wu * 0
        ws = np.sqrt(wu**2 + wv**2)
        summary["mean_wind_speed_ms"] = round(float(np.nanmean(ws)), 3)
    return summary


def build_download_netcdf_path(pred_ds, output_dir: str) -> str:
    """Return path of prediction NetCDF for user download."""
    import xarray as xr
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "user_prediction_download.nc")
    if pred_ds is not None:
        pred_ds.to_netcdf(path, engine="netcdf4")
    return path


def build_download_csv(pred_ds, output_dir: str) -> str:
    """Return path of CSV prediction export for user download."""
    import numpy as np
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "user_prediction_download.csv")
    if pred_ds is None:
        pd.DataFrame().to_csv(path, index=False)
        return path
    arr   = pred_ds["thetao_pred"].isel(depth=0).values   # (T, H, W)
    times = list(pred_ds["time"].values)
    lats  = list(pred_ds["latitude"].values)
    lons  = list(pred_ds["longitude"].values)
    rows  = []
    for ti, t in enumerate(times):
        for li, lat in enumerate(lats):
            for oi, lon in enumerate(lons):
                v = float(arr[ti, li, oi])
                if np.isfinite(v):
                    rows.append({"time": str(t), "latitude": lat,
                                 "longitude": lon, "sst_pred": round(v, 3)})
    pd.DataFrame(rows).to_csv(path, index=False)
    return path
