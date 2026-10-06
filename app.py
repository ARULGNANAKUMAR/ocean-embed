"""
app.py
North Indian Ocean Intelligence Platform — SIH Final Build
Flask backend: REST API + SocketIO live updates + template routes.
All Phase 1/2/3 orchestration logic lives in pipeline.py.
"""

import hashlib
import io
import json
import logging
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import secrets as _secrets_mod

import numpy as np
import pandas as pd
import xarray as xr
from dotenv import load_dotenv
from flask import (Flask, jsonify, render_template, request,
                    send_file, send_from_directory)
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename

import admin as adm
import ai_engine as ai
import evaluation as ev
import firebase_auth as fbauth
import pipeline
import preprocessing as pre
import visualization as viz
from pipeline import _state, run_phase1, run_phase2, run_phase3

load_dotenv()  # reads .env if present; safe no-op if absent

# ──────────────────────────────────────────────────────────────
# FLASK APP + SOCKETIO
# ──────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", template_folder="templates")

_secret_key = os.environ.get("SECRET_KEY", "").strip()
if not _secret_key:
    # No hardcoded fallback secret is committed. In dev without a .env this
    # generates a random key each restart (sessions won't survive a
    # restart, which is intentionally safer than a shared static secret).
    _secret_key = _secrets_mod.token_hex(32)
    logging.getLogger("app").warning(
        "SECRET_KEY not set in environment -- using a random ephemeral key. "
        "Set SECRET_KEY in your .env for stable sessions."
    )
app.config["SECRET_KEY"] = _secret_key
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB uploads
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# Enable Secure cookies automatically once running behind HTTPS in production.
app.config["SESSION_COOKIE_SECURE"] = os.environ.get("FLASK_ENV") == "production"

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


def _nan_safe(obj):
    """Recursively replace NaN/Inf with None for JSON safety."""
    if isinstance(obj, dict):
        return {k: _nan_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return v if math.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _nan_safe(obj.tolist())
    return obj


def jr(data, status=200):
    """JSON response helper — NaN-safe."""
    return jsonify(_nan_safe(data)), status


def emit_notification(kind: str, message: str, level: str = "info"):
    """Push a live toast notification to all connected clients."""
    try:
        socketio.emit("notification", {
            "kind": kind, "message": message, "level": level,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════
# TEMPLATE / PAGE ROUTES
# ══════════════════════════════════════════════════════════════

@app.route("/")
def page_home():
    return render_template("index.html")

@app.route("/earth")
def page_earth():
    return render_template("earth.html")

@app.route("/dashboard")
def page_dashboard():
    return render_template("dashboard.html")

@app.route("/prediction")
def page_prediction():
    return render_template("prediction.html")

@app.route("/heatwave")
def page_heatwave():
    return render_template("heatwave.html")

@app.route("/gismap")
def page_gismap():
    return render_template("gismap.html")

@app.route("/admin")
@fbauth.require_role("admin")
def page_admin():
    return render_template("admin.html")

@app.route("/voice")
def page_voice():
    return render_template("voice.html")

@app.route("/about")
def page_about():
    return render_template("about.html")


# ══════════════════════════════════════════════════════════════
# PHASE 1 API
# ══════════════════════════════════════════════════════════════

@app.get("/api/datasets")
def api_datasets():
    return jr({"datasets": _state["scan_records"], "total": len(_state["scan_records"])})

@app.get("/api/datasets/<int:dataset_id>")
def api_dataset_by_id(dataset_id):
    recs = _state["scan_records"]
    if dataset_id < 0 or dataset_id >= len(recs):
        return jr({"error": "Dataset not found"}, 404)
    rec = recs[dataset_id]
    return jr({"dataset": rec, "inspection": _state["inspections"].get(rec["filepath"], {})})

@app.get("/api/reports")
def api_reports():
    files = []
    for rdir in ["reports/phase1", "reports/phase2", "reports/phase3"]:
        if os.path.exists(rdir):
            files += [str(f) for f in sorted(Path(rdir).rglob("*")) if f.is_file()]
    return jr({"reports": files})

@app.get("/api/quality")
def api_quality():
    return jr({"quality_reports": _state["qc_results"], "total": len(_state["qc_results"])})

@app.get("/api/coverage")
def api_coverage():
    return jr({"coverage_reports": _state["cov_results"], "total": len(_state["cov_results"])})

@app.get("/api/summary")
def api_summary():
    return jr(_state["summary"])


# ══════════════════════════════════════════════════════════════
# PHASE 2 API
# ══════════════════════════════════════════════════════════════

@app.get("/api/phase2/status")
def api_phase2_status():
    return jr(_state.get("p2_status", {}))

@app.get("/api/statistics")
def api_statistics():
    return jr({"statistics": _state["p2_statistics"], "total": len(_state["p2_statistics"])})

@app.get("/api/depths")
def api_depths():
    return jr({"depths": _state["p2_depths"], "total": len(_state["p2_depths"])})

@app.get("/api/surface")
def api_surface():
    return jr({"shape": _state["p2_surface_shape"],
              "variables": ["sst","sss","sla","sla_observation_count",
                            "current_u","current_v","wind_u","wind_v"]})

@app.get("/api/target")
def api_target():
    return jr({"shape": _state["p2_target_shape"], "variables": ["thetao"]})

@app.get("/api/missing")
def api_missing():
    return jr({"missing": _state["p2_missing"], "total": len(_state["p2_missing"])})

@app.get("/api/interpolation")
def api_interpolation():
    return jr({"interpolation": _state["p2_interpolation"], "total": len(_state["p2_interpolation"])})


# ══════════════════════════════════════════════════════════════
# PHASE 3 API — CORE
# ══════════════════════════════════════════════════════════════

@app.get("/api/health")
def api_health():
    return jr({"status": "ok", "phase": 3, "model_ready": _state["p3_model"] is not None})

@app.get("/api/status")
def api_status():
    return jr({
        "phase1": "complete",
        "phase2": _state["p2_status"].get("status", "unknown"),
        "phase3": _state["p3_status"].get("status", "unknown"),
        "model": "loaded" if _state["p3_model"] is not None else "not_loaded",
    })

@app.get("/api/prediction/latest")
def api_prediction_latest():
    result = adm.get_latest_prediction(_state["p3_pred_ds"], _state["p3_hotspots"], _state["p3_uncertainty"])
    return jr(result)

@app.get("/api/prediction/date")
def api_prediction_date():
    date = request.args.get("date")
    days_ahead = int(request.args.get("days_ahead", 0))
    if days_ahead in _state["p3_future_fc"]:
        return jr(_state["p3_future_fc"][days_ahead])
    if _state["p3_pred_ds"] is None:
        return jr({"error": "No predictions available"}, 404)
    try:
        target_date = date or str(pd.Timestamp(_state["p3_times"][-1]).date())
        prof = _state["p3_pred_ds"]["thetao_pred"].sel(
            time=target_date, method="nearest").mean(dim=["latitude","longitude"]).values
        return jr({"date": target_date, "depth_profile": prof.tolist(),
                  "depths": list(_state["p3_pred_ds"]["depth"].values)})
    except Exception as e:
        return jr({"error": str(e)}, 400)

@app.get("/api/heatwave")
def api_heatwave():
    return jr({"hotspots": _state["p3_hotspots"], "total": len(_state["p3_hotspots"])})

@app.get("/api/heatwave/forecast")
def api_heatwave_forecast():
    alert = adm.get_heatwave_alert(_state["p3_hotspots"])
    return jr({"alert": alert, "impacts": _state["p3_impacts"][:5]})

@app.get("/api/depth-profile")
def api_depth_profile():
    lat  = float(request.args.get("lat", 12.5))
    lon  = float(request.args.get("lon", 82.5))
    date = request.args.get("date", "2025-01-15")
    cfg  = _state["config"] or {}
    raw_dir  = cfg.get("paths", {}).get("raw_data", "data/raw")
    argo_dir = raw_dir if raw_dir.endswith("argo") else os.path.join(raw_dir, "argo")
    if not os.path.exists(argo_dir):
        argo_dir = raw_dir
    if _state["p3_pred_ds"] is None:
        return jr({"error": "No prediction dataset loaded"}, 404)
    profile = ev.get_depth_profile(_state["p3_pred_ds"], lat, lon, date, argo_dir, cfg)
    return jr(profile)

@app.get("/api/argo/validation")
def api_argo_validation():
    return jr(_state["p3_argo_result"])

@app.get("/api/attention")
def api_attention():
    return jr({"attention_map": _state["p3_xai"].get("attention_map"),
              "explanation": _state["p3_xai"].get("explanation")})

@app.get("/api/embedding")
def api_embedding():
    if _state["p3_pred_ds"] is None:
        return jr({"error": "No prediction available"}, 404)
    arr = _state["p3_pred_ds"]["thetao_pred"].isel(depth=0, time=-1).values
    return jr({"embedding_shape": list(arr.shape), "surface_embedding": arr.tolist()})

@app.get("/api/confidence")
def api_confidence():
    return jr({"overall_confidence": _state["p3_uncertainty"].get("overall_confidence"),
              "confidence_map": _state["p3_uncertainty"].get("confidence_map")})

@app.get("/api/uncertainty")
def api_uncertainty():
    return jr({"overall_confidence": _state["p3_uncertainty"].get("overall_confidence"),
              "uncertainty_map": _state["p3_uncertainty"].get("uncertainty_map"),
              "depth_uncertainty": _state["p3_uncertainty"].get("depth_uncertainty")})

@app.post("/api/predict")
def api_predict():
    lat = float(request.args.get("lat", 12.5))
    lon = float(request.args.get("lon", 82.5))
    if _state["p3_model"] is None or _state["p3_surface"] is None:
        return jr({"error": "Model not ready"}, 503)
    pred = ai.predict_single(_state["p3_model"], _state["p3_surface"][-1], _state["p3_norm_stats"])
    lats_arr = np.array(_state["p3_lats"])
    lons_arr = np.array(_state["p3_lons"])
    li = int(np.argmin(np.abs(lats_arr - lat)))
    oi = int(np.argmin(np.abs(lons_arr - lon)))
    profile = pred[:, li, oi].tolist()
    result = {"lat": float(lats_arr[li]), "lon": float(lons_arr[oi]),
              "depth_profile": profile,
              "depths": (_state["config"] or {}).get("target_depths", list(range(15)))}
    emit_notification("prediction", f"Prediction generated for {lat:.2f}N {lon:.2f}E", "success")
    return jr(result)

@app.post("/api/voice/query")
def api_voice_query():
    query = request.args.get("query") or (request.json or {}).get("query", "")
    if _state["p3_pred_ds"] is None:
        return jr({"error": "No prediction loaded", "query": query})
    result = ai.process_voice_query(query, _state["p3_pred_ds"], _state["p3_hotspots"], _state["config"] or {})
    return jr(result)


# ══════════════════════════════════════════════════════════════
# ADMIN API
# ══════════════════════════════════════════════════════════════

ALLOWED_UPLOAD_EXTENSIONS = {".nc", ".nc4", ".csv", ".json", ".txt"}


def _safe_upload_path(raw_dir: str, filename: str):
    """
    Sanitize an uploaded filename and confirm the resolved path stays
    inside raw_dir. Returns (dest_path, error_message_or_None).
    """
    safe_name = secure_filename(filename or "")
    if not safe_name:
        return None, "Invalid filename"
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        return None, f"File type '{ext}' not allowed. Allowed: {sorted(ALLOWED_UPLOAD_EXTENSIONS)}"
    dest = os.path.abspath(os.path.join(raw_dir, safe_name))
    raw_dir_abs = os.path.abspath(raw_dir)
    if not dest.startswith(raw_dir_abs + os.sep) and dest != raw_dir_abs:
        return None, "Invalid upload path"
    # Never allow overwriting an existing file with a same-named upload --
    # de-duplicate with a short random suffix instead.
    if os.path.exists(dest):
        base, ext2 = os.path.splitext(safe_name)
        dest = os.path.join(raw_dir_abs, f"{base}_{_secrets_mod.token_hex(4)}{ext2}")
    return dest, None


@app.post("/api/admin/upload")
@fbauth.require_role("admin")
def api_admin_upload():
    if "file" not in request.files:
        return jr({"error": "No file provided"}, 400)
    file = request.files["file"]
    dataset_type = request.args.get("dataset_type", "UNKNOWN")
    cfg = _state["config"] or {}
    raw_dir = cfg.get("paths", {}).get("raw_data", "data/raw")
    os.makedirs(raw_dir, exist_ok=True)
    dest, err = _safe_upload_path(raw_dir, file.filename)
    if err:
        return jr({"error": err}, 400)
    file.save(dest)
    conn = sqlite3.connect(_state["db_path"])
    adm.init_admin_tables(conn)
    fbauth.log_security_event(conn, "dataset_upload", flask_session.get("fb_uid", ""),
                               flask_session.get("fb_email", ""), os.path.basename(dest))
    result = adm.admin_upload_dataset(conn, os.path.basename(dest), dest, dataset_type)
    conn.close()
    emit_notification("upload", f"Dataset uploaded: {os.path.basename(dest)}", "success")
    return jr(result)

@app.post("/api/admin/retrain")
@fbauth.require_role("admin")
def api_admin_retrain():
    conn = sqlite3.connect(_state["db_path"])
    adm.init_admin_tables(conn)
    fbauth.log_security_event(conn, "training_start", flask_session.get("fb_uid", ""),
                               flask_session.get("fb_email", ""))
    result = adm.admin_retrain_model(conn, _state["config"] or {})
    conn.close()
    emit_notification("training", "Model retrain triggered", "info")
    return jr(result)

@app.get("/api/admin/models")
@fbauth.require_role("admin")
def api_admin_models():
    conn = sqlite3.connect(_state["db_path"])
    history = adm.get_model_history(conn)
    conn.close()
    return jr({"models": history})

@app.get("/api/admin/history")
@fbauth.require_role("admin")
def api_admin_history():
    conn = sqlite3.connect(_state["db_path"])
    logs = adm.get_admin_logs(conn)
    preds = adm.get_prediction_history(conn)
    trains = adm.get_training_history(conn)
    stats = adm.get_platform_statistics(conn)
    conn.close()
    return jr({"logs": logs, "predictions": preds, "training": trains, "statistics": stats})


# ══════════════════════════════════════════════════════════════
# DIGITAL TWIN + MAP API
# ══════════════════════════════════════════════════════════════

@app.get("/api/digital-twin")
def api_digital_twin():
    time_idx = int(request.args.get("time_idx", -1))
    if _state["p3_pred_ds"] is None:
        return jr({"error": "No prediction available"}, 404)
    data = ev.get_digital_twin_data(_state["p3_pred_ds"], time_idx)
    return jr(data)

@app.get("/api/map/layer")
def api_map_layer():
    layer     = request.args.get("layer", "sst")
    time_idx  = int(request.args.get("time_idx", -1))
    depth_idx = int(request.args.get("depth_idx", 0))
    if _state["p3_pred_ds"] is None:
        return jr({"error": "No prediction available"}, 404)
    data = ev.get_map_layer(_state["p3_surface_ds"] or xr.Dataset(),
                            _state["p3_pred_ds"], layer, time_idx, depth_idx)
    return jr(data)

@app.get("/api/multivar")
def api_multivar():
    return jr(_state["p3_multivar"])


# ══════════════════════════════════════════════════════════════
# USER API
# ══════════════════════════════════════════════════════════════

@app.get("/api/user/prediction")
def api_user_prediction():
    return jr(adm.get_latest_prediction(_state["p3_pred_ds"], _state["p3_hotspots"], _state["p3_uncertainty"]))

@app.get("/api/user/heatwave-alert")
def api_user_heatwave_alert():
    return jr(adm.get_heatwave_alert(_state["p3_hotspots"]))

@app.get("/api/user/ocean-summary")
def api_user_ocean_summary():
    return jr(adm.get_ocean_summary(_state["p3_pred_ds"], _state["p3_surface_ds"]))

@app.get("/api/user/download/netcdf")
def api_user_download_netcdf():
    path = adm.build_download_netcdf_path(_state["p3_pred_ds"], "predictions")
    if os.path.exists(path):
        return send_file(path, as_attachment=True, download_name="ocean_prediction.nc")
    return jr({"error": "NetCDF not ready"}, 404)

@app.get("/api/user/download/csv")
def api_user_download_csv():
    path = adm.build_download_csv(_state["p3_pred_ds"], "predictions")
    return send_file(path, as_attachment=True, download_name="ocean_prediction.csv")


# ══════════════════════════════════════════════════════════════
# ARGO FLOATS API  (Section 9)
# ══════════════════════════════════════════════════════════════

@app.get("/api/argo/floats")
def api_argo_floats():
    """List all ARGO floats with summary info for map markers."""
    cfg = _state["config"] or {}
    raw_dir = cfg.get("paths", {}).get("raw_data", "data/raw")
    argo_dir = raw_dir if raw_dir.endswith("argo") else os.path.join(raw_dir, "argo")
    if not os.path.exists(argo_dir):
        argo_dir = raw_dir

    floats = []
    for fp in Path(argo_dir).glob("*.nc"):
        try:
            ds_a = xr.open_dataset(str(fp))
            lat_v = "LATITUDE" if "LATITUDE" in ds_a else "latitude"
            lon_v = "LONGITUDE" if "LONGITUDE" in ds_a else "longitude"
            n_prof = int(ds_a.sizes.get("N_PROF", ds_a.sizes.get("profile", 0)))
            for i in range(min(n_prof, 50)):
                floats.append({
                    "float_id": f"{fp.stem}_{i}",
                    "filename": fp.name,
                    "latitude": float(ds_a[lat_v].values[i]) if lat_v in ds_a else None,
                    "longitude": float(ds_a[lon_v].values[i]) if lon_v in ds_a else None,
                })
            ds_a.close()
        except Exception:
            pass
    return jr({"floats": floats, "total": len(floats)})

@app.get("/api/argo/float/<float_id>")
def api_argo_float_detail(float_id):
    """Return profile detail for a specific ARGO float."""
    cfg = _state["config"] or {}
    raw_dir = cfg.get("paths", {}).get("raw_data", "data/raw")
    argo_dir = raw_dir if raw_dir.endswith("argo") else os.path.join(raw_dir, "argo")
    if not os.path.exists(argo_dir):
        argo_dir = raw_dir

    try:
        fname, idx_str = float_id.rsplit("_", 1)
        idx = int(idx_str)
    except Exception:
        return jr({"error": "Invalid float_id"}, 400)

    fp = None
    for f in Path(argo_dir).glob("*.nc"):
        if f.stem == fname:
            fp = f; break
    if fp is None:
        return jr({"error": "Float file not found"}, 404)

    try:
        ds_a = xr.open_dataset(str(fp))
        lat_v = "LATITUDE" if "LATITUDE" in ds_a else "latitude"
        lon_v = "LONGITUDE" if "LONGITUDE" in ds_a else "longitude"
        temp_v = "TEMP" if "TEMP" in ds_a else "temperature"
        pres_v = "PRES" if "PRES" in ds_a else "pressure"
        time_v = "JULD" if "JULD" in ds_a else "time"

        result = {
            "float_id": float_id,
            "latitude": float(ds_a[lat_v].values[idx]),
            "longitude": float(ds_a[lon_v].values[idx]),
            "date": str(pd.Timestamp(ds_a[time_v].values[idx])) if time_v in ds_a else None,
            "temperature_profile": [float(v) for v in ds_a[temp_v].values[idx]] if temp_v in ds_a else [],
            "depth_profile": [float(v) for v in ds_a[pres_v].values[idx]] if pres_v in ds_a else [],
        }
        ds_a.close()
        return jr(result)
    except Exception as e:
        return jr({"error": str(e)}, 500)


# ══════════════════════════════════════════════════════════════
# OCEAN GRID INFO API  (Section 5)
# ══════════════════════════════════════════════════════════════

@app.get("/api/grid-cell")
def api_grid_cell():
    """Return all info for a clicked 0.25° grid cell."""
    lat = float(request.args.get("lat", 12.5))
    lon = float(request.args.get("lon", 82.5))
    if _state["p3_surface_ds"] is None or _state["p3_pred_ds"] is None:
        return jr({"error": "Data not loaded"}, 404)

    surf = _state["p3_surface_ds"]
    pred = _state["p3_pred_ds"]
    lats_arr = surf["latitude"].values
    lons_arr = surf["longitude"].values
    li = int(np.argmin(np.abs(lats_arr - lat)))
    oi = int(np.argmin(np.abs(lons_arr - lon)))

    def _last(var):
        if var in surf.data_vars:
            v = float(surf[var].isel(time=-1, latitude=li, longitude=oi).values)
            return v if math.isfinite(v) else None
        return None

    conf = _state["p3_uncertainty"].get("confidence_map")
    conf_val = None
    if conf:
        arr_c = np.array(conf)
        if arr_c.ndim == 3:
            arr_c = arr_c.mean(axis=0)
        if li < arr_c.shape[0] and oi < arr_c.shape[1]:
            conf_val = round(float(arr_c[li, oi]), 3)

    hw_status = "NONE"
    for hs in _state["p3_hotspots"]:
        if abs(hs["latitude"] - lat) < 0.15 and abs(hs["longitude"] - lon) < 0.15:
            hw_status = hs["severity"]
            break

    return jr({
        "latitude": float(lats_arr[li]), "longitude": float(lons_arr[oi]),
        "sst": _last("sst"), "sss": _last("sss"), "sla": _last("sla"),
        "wind_u": _last("wind_u"), "wind_v": _last("wind_v"),
        "current_u": _last("current_u"), "current_v": _last("current_v"),
        "prediction_confidence": conf_val,
        "heatwave_status": hw_status,
    })


# ══════════════════════════════════════════════════════════════
# SOCKETIO EVENTS  (Section 24 — Live Notifications)
# ══════════════════════════════════════════════════════════════

@socketio.on("connect")
def on_connect():
    emit("notification", {
        "kind": "system", "message": "Connected to Ocean Intelligence Platform",
        "level": "info", "timestamp": datetime.now(timezone.utc).isoformat(),
    })

@socketio.on("disconnect")
def on_disconnect():
    pass

@socketio.on("request_status")
def on_request_status():
    emit("status_update", {
        "phase1": "complete",
        "phase2": _state["p2_status"].get("status", "unknown"),
        "phase3": _state["p3_status"].get("status", "unknown"),
    })


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

def run_all_phases():
    pre.setup_logging()
    cfg = pre.load_config("config.json")
    run_phase1(cfg, start_api=False)
    run_phase2(cfg, start_api=False)
    run_phase3(cfg, start_api=False)
    return cfg


if __name__ == "__main__":
    cfg = run_all_phases()
    host = cfg.get("api", {}).get("host", "0.0.0.0")
    port = cfg.get("api", {}).get("port", 8000)
    print(f"\n  Flask + SocketIO server -> http://{host}:{port}\n")
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)


# ══════════════════════════════════════════════════════════════
# PHASE 1 — PAGE ROUTES  (Section 29)
# ══════════════════════════════════════════════════════════════

import ocean_platform as plat_module
from flask import session as flask_session

# Lazy DB helper that always uses the running state db_path
def _db():
    path = _state.get("db_path") or "data/database/ocean_platform.db"
    if path != ":memory:":
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    plat_module.init_phase1_tables(conn)
    return conn

def _current_user():
    token = flask_session.get("token")
    if not token:
        return None
    conn = _db()
    u = plat_module.validate_session(conn, token)
    conn.close()
    return u

@app.route("/settings")
@fbauth.require_auth
def page_settings():
    return render_template("settings.html")

@app.route("/profile")
@fbauth.require_auth
def page_profile():
    return render_template("profile.html")

@app.route("/downloads")
@fbauth.require_auth
def page_downloads():
    return render_template("downloads.html")

@app.route("/database")
@fbauth.require_role("admin")
def page_database():
    return render_template("database.html")

@app.route("/training")
@fbauth.require_role("admin")
def page_training():
    return render_template("training.html")

@app.route("/activity")
@fbauth.require_auth
def page_activity():
    return render_template("activity.html")

@app.route("/api-management")
@fbauth.require_role("admin")
def page_api_management():
    return render_template("api_management.html")

@app.route("/model-registry")
@fbauth.require_role("admin")
def page_model_registry():
    return render_template("model_registry.html")

@app.route("/reports")
@fbauth.require_auth
def page_reports():
    return render_template("reports.html")

@app.route("/login")
def page_login():
    return render_template("login.html")


# ══════════════════════════════════════════════════════════════
# FIREBASE AUTH API  (primary auth boundary)
# ══════════════════════════════════════════════════════════════

@app.get("/api/firebase-config")
def api_firebase_config():
    """Public Firebase Web SDK config -- these values are identifiers, not secrets."""
    return jr(fbauth.get_public_firebase_config())


@app.post("/api/auth/firebase-session")
def api_firebase_session():
    """
    Exchange a Firebase ID token (obtained client-side via the Firebase Web
    SDK) for a server-signed Flask session. The ID token is verified here,
    server-side, via the Firebase Admin SDK -- the client's claims about its
    own email/uid/role are never trusted directly.
    """
    data = request.json or {}
    id_token = data.get("idToken", "")
    conn = _db()
    try:
        decoded = fbauth.verify_id_token(id_token)
    except fbauth.TokenError as e:
        fbauth.log_security_event(conn, "login_token_rejected", detail=str(e))
        conn.close()
        return jr({"error": "Your session could not be verified. Please sign in again."}, 401)

    uid = decoded.get("uid")
    email = decoded.get("email", "") or ""
    email_verified = bool(decoded.get("email_verified", False))
    display_name = decoded.get("name", "") or ""

    role = fbauth.get_or_create_role(conn, uid, email, email_verified, display_name)

    flask_session.clear()
    flask_session["fb_uid"] = uid
    flask_session["fb_email"] = email
    flask_session["fb_role"] = role
    flask_session["fb_email_verified"] = email_verified
    flask_session["fb_display_name"] = display_name
    flask_session.permanent = True

    fbauth.log_security_event(conn, "login_success", uid, email)
    conn.close()

    return jr({
        "status": "ok",
        "user": {
            "uid": uid, "email": email, "role": role,
            "email_verified": email_verified, "display_name": display_name,
        },
    })


@app.post("/api/auth/firebase-logout")
def api_firebase_logout():
    conn = _db()
    fbauth.log_security_event(conn, "logout", flask_session.get("fb_uid", ""),
                               flask_session.get("fb_email", ""))
    conn.close()
    flask_session.clear()
    return jr({"status": "logged_out"})


@app.get("/api/auth/firebase-me")
def api_firebase_me():
    user = fbauth.current_firebase_user()
    if not user:
        return jr({"error": "Not authenticated"}, 401)
    return jr({"user": user})


# ══════════════════════════════════════════════════════════════
# LEGACY AUTH API
# (Pre-existing username/password system. Superseded by Firebase Auth
#  above for new logins -- kept only so existing session-dependent code
#  paths in this file (favorites/bookmarks/settings fallbacks) don't break
#  mid-migration. Do not build new features against this; it is scheduled
#  for removal once every route below reads identity from
#  fbauth.current_firebase_user() instead of _current_user().)
# ══════════════════════════════════════════════════════════════

@app.post("/api/auth/login")
def api_auth_login():
    data = request.json or {}
    username = data.get("username", "").strip()
    password = data.get("password", "")
    if not username or not password:
        return jr({"error": "Username and password required"}, 400)
    conn = _db()
    user = plat_module.authenticate(conn, username, password)
    if not user:
        conn.close()
        return jr({"error": "Invalid credentials"}, 401)
    token = plat_module.create_session(conn, user["id"])
    conn.close()
    flask_session["token"] = token
    flask_session["user_id"] = user["id"]
    safe_user = {k: v for k, v in user.items() if k != "password_hash"}
    return jr({"status": "ok", "token": token, "user": safe_user})

@app.post("/api/auth/logout")
def api_auth_logout():
    token = flask_session.get("token") or request.json.get("token", "")
    if token:
        conn = _db()
        plat_module.invalidate_session(conn, token)
        conn.close()
    flask_session.clear()
    return jr({"status": "logged_out"})

@app.get("/api/auth/me")
def api_auth_me():
    user = _current_user()
    if not user:
        return jr({"error": "Not authenticated"}, 401)
    safe = {k: v for k, v in user.items() if k != "password_hash"}
    return jr({"user": safe})

@app.post("/api/auth/register")
def api_auth_register():
    data = request.json or {}
    conn = _db()
    result = plat_module.create_user(
        conn, data.get("username",""), data.get("email",""),
        data.get("password",""), role="guest",
        display_name=data.get("display_name",""))
    conn.close()
    return jr(result)


# ══════════════════════════════════════════════════════════════
# SETTINGS API
# ══════════════════════════════════════════════════════════════

def _identity_uid(default=0):
    """
    Prefer the Firebase-verified session; fall back to the legacy session
    system only for routes not yet gated by @require_auth, so nothing
    silently 500s mid-migration. Routes decorated with @fbauth.require_auth
    always have a real fb_uid by the time they run.
    """
    fb = fbauth.current_firebase_user()
    if fb:
        return fb["uid"]
    user = _current_user()
    return user["id"] if user else default


@app.get("/api/settings")
@fbauth.require_auth
def api_get_settings():
    uid = _identity_uid()
    conn = _db()
    s = plat_module.get_settings(conn, uid)
    conn.close()
    return jr({"settings": s})

@app.post("/api/settings")
@fbauth.require_auth
def api_save_settings():
    data = request.json or {}
    uid = _identity_uid()
    conn = _db()
    result = plat_module.save_settings(conn, uid, data)
    conn.close()
    return jr(result)


# ══════════════════════════════════════════════════════════════
# FAVORITES API
# ══════════════════════════════════════════════════════════════

@app.get("/api/favorites")
@fbauth.require_auth
def api_list_favorites():
    uid = _identity_uid(default=1)
    conn = _db()
    favs = plat_module.list_favorites(conn, uid)
    conn.close()
    return jr({"favorites": favs, "total": len(favs)})

@app.post("/api/favorites")
@fbauth.require_auth
def api_add_favorite():
    data = request.json or {}
    uid = _identity_uid(default=1)
    conn = _db()
    result = plat_module.add_favorite(
        conn, uid, data.get("name","Location"),
        float(data.get("lat", 0)), float(data.get("lon", 0)),
        data.get("notes",""))
    conn.close()
    return jr(result)

@app.delete("/api/favorites/<int:fav_id>")
@fbauth.require_auth
def api_delete_favorite(fav_id):
    uid = _identity_uid(default=1)
    conn = _db()
    result = plat_module.delete_favorite(conn, uid, fav_id)
    conn.close()
    return jr(result)


# ══════════════════════════════════════════════════════════════
# BOOKMARKS API
# ══════════════════════════════════════════════════════════════

@app.get("/api/bookmarks")
@fbauth.require_auth
def api_list_bookmarks():
    uid = _identity_uid(default=1)
    conn = _db()
    bms = plat_module.list_bookmarks(conn, uid)
    conn.close()
    return jr({"bookmarks": bms, "total": len(bms)})

@app.post("/api/bookmarks")
@fbauth.require_auth
def api_add_bookmark():
    data = request.json or {}
    uid = _identity_uid(default=1)
    conn = _db()
    result = plat_module.add_bookmark(
        conn, uid, data.get("name","Bookmark"),
        float(data.get("lat", 0)), float(data.get("lon", 0)),
        int(data.get("zoom", 6)), data.get("layer","sst"))
    conn.close()
    return jr(result)

@app.delete("/api/bookmarks/<int:bm_id>")
@fbauth.require_auth
def api_delete_bookmark(bm_id):
    uid = _identity_uid(default=1)
    conn = _db()
    result = plat_module.delete_bookmark(conn, uid, bm_id)
    conn.close()
    return jr(result)

@app.patch("/api/bookmarks/<int:bm_id>")
@fbauth.require_auth
def api_rename_bookmark(bm_id):
    data = request.json or {}
    uid = _identity_uid(default=1)
    conn = _db()
    result = plat_module.rename_bookmark(conn, uid, bm_id, data.get("name",""))
    conn.close()
    return jr(result)


# ══════════════════════════════════════════════════════════════
# ACTIVITY API
# ══════════════════════════════════════════════════════════════

@app.get("/api/activity")
def api_activity():
    user = _current_user()
    uid = user["id"] if user else None
    limit = int(request.args.get("limit", 100))
    conn = _db()
    events = plat_module.get_activity(conn, uid, limit)
    conn.close()
    return jr({"activity": events, "total": len(events)})


# ══════════════════════════════════════════════════════════════
# DATASET REGISTRY API
# ══════════════════════════════════════════════════════════════

@app.get("/api/registry/datasets")
def api_registry_datasets():
    conn = _db()
    datasets = plat_module.list_datasets(conn)
    stats = plat_module.get_dataset_stats(conn)
    conn.close()
    return jr({"datasets": datasets, "stats": stats, "total": len(datasets)})

@app.post("/api/registry/datasets")
@fbauth.require_role("admin")
def api_registry_register():
    """Register a manually specified dataset entry."""
    data = request.json or {}
    cfg = _state["config"] or {}
    raw_dir = cfg.get("paths", {}).get("raw_data", "data/raw")
    filepath = os.path.join(raw_dir, data.get("filename", ""))
    conn = _db()
    result = plat_module.register_dataset(
        conn, data.get("filename",""), filepath,
        data.get("source",""), data.get("variable",""),
        uploader=data.get("uploader","admin"),
        extra=data.get("meta",{}))
    conn.close()
    return jr(result)

@app.delete("/api/registry/datasets/<int:ds_id>")
@fbauth.require_role("admin")
def api_registry_delete(ds_id):
    conn = _db()
    result = plat_module.delete_dataset_registry(conn, ds_id)
    conn.close()
    return jr(result)


# ══════════════════════════════════════════════════════════════
# MODEL REGISTRY API
# ══════════════════════════════════════════════════════════════

@app.get("/api/registry/models")
def api_registry_models():
    conn = _db()
    models = plat_module.list_models(conn)
    conn.close()
    return jr({"models": models, "total": len(models)})

@app.post("/api/registry/models/activate/<int:model_id>")
@fbauth.require_role("admin")
def api_activate_model(model_id):
    conn = _db()
    result = plat_module.activate_model(conn, model_id)
    conn.close()
    emit_notification("model", f"Model {model_id} activated", "success")
    return jr(result)

@app.post("/api/registry/models/rollback/<int:model_id>")
@fbauth.require_role("admin")
def api_rollback_model(model_id):
    conn = _db()
    result = plat_module.rollback_model(conn, model_id)
    conn.close()
    return jr(result)

@app.get("/api/registry/models/compare")
def api_compare_models():
    ids_raw = request.args.get("ids","")
    try:
        ids = [int(x) for x in ids_raw.split(",") if x.strip()]
    except ValueError:
        return jr({"error": "Invalid ids"}, 400)
    conn = _db()
    result = plat_module.compare_models(conn, ids)
    conn.close()
    return jr({"models": result})


# ══════════════════════════════════════════════════════════════
# API KEY VAULT API
# ══════════════════════════════════════════════════════════════

@app.get("/api/apikeys")
@fbauth.require_role("admin")
def api_list_apikeys():
    conn = _db()
    keys = plat_module.list_api_keys(conn)
    conn.close()
    return jr({"keys": keys, "total": len(keys)})

@app.post("/api/apikeys")
@fbauth.require_role("admin")
def api_set_apikey():
    data = request.json or {}
    conn = _db()
    result = plat_module.set_api_key(
        conn, data.get("service",""), data.get("key",""), data.get("notes",""))
    conn.close()
    return jr(result)

@app.post("/api/apikeys/test")
@fbauth.require_role("admin")
def api_test_apikey():
    service = (request.json or {}).get("service","")
    conn = _db()
    result = plat_module.test_api_key(conn, service)
    conn.close()
    return jr(result)

@app.patch("/api/apikeys/<path:service>")
@fbauth.require_role("admin")
def api_toggle_apikey(service):
    data = request.json or {}
    conn = _db()
    result = plat_module.toggle_api_key(conn, service, bool(data.get("enabled", True)))
    conn.close()
    return jr(result)


# ══════════════════════════════════════════════════════════════
# SCHEDULER API
# ══════════════════════════════════════════════════════════════

@app.get("/api/scheduler")
def api_list_scheduler():
    conn = _db()
    jobs = plat_module.list_jobs(conn)
    conn.close()
    return jr({"jobs": jobs, "total": len(jobs)})

@app.post("/api/scheduler/run")
@fbauth.require_role("admin")
def api_run_job():
    job = (request.json or {}).get("job_name","")
    conn = _db()
    result = plat_module.run_job_now(conn, job)
    conn.close()
    emit_notification("scheduler", f"Job triggered: {job}", "info")
    return jr(result)

@app.patch("/api/scheduler/<path:job_name>")
@fbauth.require_role("admin")
def api_update_job(job_name):
    data = request.json or {}
    conn = _db()
    result = plat_module.update_job_interval(conn, job_name, data.get("interval","daily"))
    conn.close()
    return jr(result)


# ══════════════════════════════════════════════════════════════
# NOTIFICATIONS API
# ══════════════════════════════════════════════════════════════

@app.get("/api/notifications")
def api_list_notifications():
    unread = request.args.get("unread","false").lower() == "true"
    limit = int(request.args.get("limit", 50))
    conn = _db()
    notifs = plat_module.list_notifications(conn, unread, limit)
    conn.close()
    return jr({"notifications": notifs, "total": len(notifs)})

@app.post("/api/notifications/read")
def api_mark_read():
    conn = _db()
    result = plat_module.mark_notifications_read(conn)
    conn.close()
    return jr(result)


# ══════════════════════════════════════════════════════════════
# DATABASE HEALTH API
# ══════════════════════════════════════════════════════════════

@app.get("/api/db/health")
def api_db_health():
    db_path = _state.get("db_path") or "data/database/ocean_platform.db"
    conn = _db()
    health = plat_module.get_db_health(conn, db_path)
    conn.close()
    return jr(health)


# ══════════════════════════════════════════════════════════════
# REPORTS API
# ══════════════════════════════════════════════════════════════

@app.get("/api/reports/catalog")
def api_reports_catalog():
    conn = _db()
    reps = plat_module.list_reports(conn)
    conn.close()
    return jr({"reports": reps, "total": len(reps)})


# ══════════════════════════════════════════════════════════════
# PLATFORM STATS (admin dashboard counters)
# ══════════════════════════════════════════════════════════════

@app.get("/api/platform/stats")
def api_platform_stats():
    conn = _db()
    stats = plat_module.get_platform_stats(conn)
    conn.close()
    return jr(stats)

@app.get("/api/admin/users")
@fbauth.require_role("admin")
def api_admin_users():
    conn = _db()
    users = plat_module.list_users(conn)
    fb_users = fbauth.list_firebase_users(conn)
    conn.close()
    return jr({"users": users, "total": len(users), "firebase_users": fb_users})

@app.patch("/api/admin/users/<int:user_id>/role")
@fbauth.require_role("admin")
def api_admin_update_role(user_id):
    data = request.json or {}
    conn = _db()
    result = plat_module.update_user_role(conn, user_id, data.get("role","guest"))
    fbauth.log_security_event(conn, "role_change_legacy", flask_session.get("fb_uid",""),
                               flask_session.get("fb_email",""),
                               f"legacy user_id={user_id} -> {data.get('role','guest')}")
    conn.close()
    return jr(result)

@app.patch("/api/admin/firebase-users/<uid>/role")
@fbauth.require_role("admin")
def api_admin_update_firebase_role(uid):
    """Promote/demote a Firebase-authenticated user. Admin-only, server-verified."""
    data = request.json or {}
    new_role = data.get("role", "user")
    conn = _db()
    result = fbauth.set_user_role(conn, uid, new_role)
    fbauth.log_security_event(conn, "role_change", flask_session.get("fb_uid",""),
                               flask_session.get("fb_email",""),
                               f"target_uid={uid} -> {new_role}")
    conn.close()
    return jr(result)



# ══════════════════════════════════════════════════════════════
# PHASE 2 — GIS MAP ROUTES + APIs
# ══════════════════════════════════════════════════════════════

import gis_engine as gis

def _gis_db():
    """Open GIS database connection (same file as platform DB)."""
    path = _state.get("db_path") or "data/database/ocean_platform.db"
    if path != ":memory:":
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    gis.init_gis_tables(conn)
    return conn


# ── Page route ─────────────────────────────────────────────
@app.route("/map")
def page_map():
    return render_template("map.html")


# ── /api/map/layers ─────────────────────────────────────────
@app.get("/api/map/layers")
def api_map_layers():
    """List all available map layers with metadata."""
    layers = [
        {"id": "sst",        "label": "Sea Surface Temp",    "category": "ocean",  "default": True},
        {"id": "sss",        "label": "Sea Surface Salinity","category": "ocean",  "default": False},
        {"id": "sla",        "label": "Sea Level Anomaly",   "category": "ocean",  "default": False},
        {"id": "current",    "label": "Ocean Currents",      "category": "ocean",  "default": False},
        {"id": "wind",       "label": "Wind Speed",          "category": "atmos",  "default": False},
        {"id": "heatwave",   "label": "Heatwave Hotspots",   "category": "ocean",  "default": True},
        {"id": "bathymetry", "label": "Bathymetry",          "category": "geo",    "default": False},
        {"id": "ports",      "label": "Major Ports",         "category": "infra",  "default": True},
        {"id": "shipping",   "label": "Shipping Routes",     "category": "infra",  "default": False},
        {"id": "eez",        "label": "EEZ Boundaries",      "category": "policy", "default": False},
        {"id": "mpa",        "label": "Marine Protected Areas","category":"policy", "default": False},
        {"id": "argo",       "label": "ARGO Floats",         "category": "obs",    "default": False},
        {"id": "ships",      "label": "AIS Vessels",         "category": "obs",    "default": False},
        {"id": "grid",       "label": "0.25° Grid",          "category": "tools",  "default": False},
        {"id": "draw",       "label": "Drawn Features",      "category": "tools",  "default": False},
    ]
    return jr({"layers": layers, "total": len(layers)})


# ── /api/map/sst ────────────────────────────────────────────
@app.get("/api/map/sst")
def api_map_sst():
    hotspots = _state.get("p3_hotspots") or []
    res = float(request.args.get("res", 1.0))
    data = gis.generate_sst_geojson(hotspots=hotspots, res=res)
    return jr(data)


# ── /api/map/wind ────────────────────────────────────────────
@app.get("/api/map/wind")
def api_map_wind():
    res = float(request.args.get("res", 2.0))
    data = gis.generate_wind_geojson(res=res)
    return jr(data)


# ── /api/map/currents ────────────────────────────────────────
@app.get("/api/map/currents")
def api_map_currents():
    res = float(request.args.get("res", 2.0))
    data = gis.generate_current_geojson(res=res)
    return jr(data)


# ── /api/map/ports ───────────────────────────────────────────
@app.get("/api/map/ports")
def api_map_ports():
    major = request.args.get("major", "false").lower() == "true"
    country = request.args.get("country")
    bbox = _parse_bbox()
    conn = _gis_db()
    data = gis.get_ports(conn, major_only=major, country=country, bbox=bbox)
    stats = gis.get_ports_stats(conn)
    conn.close()
    return jr({**data, "stats": stats})


@app.post("/api/map/ports")
def api_map_ports_add():
    d = request.json or {}
    conn = _gis_db()
    result = gis.add_port(conn, d.get("name",""), d.get("country",""),
                          float(d.get("lat",0)), float(d.get("lon",0)),
                          d.get("port_type","commercial"))
    conn.close()
    return jr(result)


@app.delete("/api/map/ports/<int:port_id>")
def api_map_ports_delete(port_id):
    conn = _gis_db()
    result = gis.delete_port(conn, port_id)
    conn.close()
    return jr(result)


# ── /api/map/routes ──────────────────────────────────────────
@app.get("/api/map/routes")
def api_map_routes():
    route_type = request.args.get("type")
    traffic = request.args.get("traffic")
    conn = _gis_db()
    data = gis.get_shipping_routes(conn, route_type=route_type, traffic=traffic)
    conn.close()
    return jr(data)


@app.post("/api/map/routes")
def api_map_routes_add():
    d = request.json or {}
    conn = _gis_db()
    result = gis.add_shipping_route(conn, d.get("name",""), d.get("coords",[]),
                                    d.get("route_type","commercial"), d.get("color","#f1c40f"))
    conn.close()
    return jr(result)


# ── /api/map/areas ───────────────────────────────────────────
@app.get("/api/map/areas")
def api_map_areas():
    area_type = request.args.get("type")
    country = request.args.get("country")
    conn = _gis_db()
    data = gis.get_marine_areas(conn, area_type=area_type, country=country)
    conn.close()
    return jr(data)


@app.post("/api/map/areas")
def api_map_areas_add():
    d = request.json or {}
    conn = _gis_db()
    result = gis.add_marine_area(conn, d.get("name",""), d.get("coords",[]),
                                 d.get("area_type","mpa"), d.get("country",""), d.get("color","#27ae60"))
    conn.close()
    return jr(result)


@app.delete("/api/map/areas/<int:area_id>")
def api_map_areas_delete(area_id):
    conn = _gis_db()
    result = gis.delete_marine_area(conn, area_id)
    conn.close()
    return jr(result)


# ── /api/map/bathymetry ──────────────────────────────────────
@app.get("/api/map/bathymetry")
def api_map_bathymetry():
    bbox = _parse_bbox()
    feature_type = request.args.get("feature_type")
    conn = _gis_db()
    data = gis.get_bathymetry(conn, bbox=bbox, feature_type=feature_type)
    stats = gis.get_bathymetry_stats(conn)
    conn.close()
    return jr({**data, "stats": stats})


@app.post("/api/map/bathymetry")
def api_map_bathymetry_add():
    d = request.json or {}
    conn = _gis_db()
    result = gis.add_bathymetry_point(conn, float(d.get("lat",0)), float(d.get("lon",0)),
                                      float(d.get("depth_m",0)), d.get("feature_type","plain"))
    conn.close()
    return jr(result)


# ── /api/map/argo ────────────────────────────────────────────
@app.get("/api/map/argo")
def api_map_argo():
    """Return ARGO floats as GeoJSON."""
    floats = _state.get("p3_argo_result", {}).get("floats") or []
    if not floats:
        # Build demo ARGO GeoJSON
        import math as _math
        demo = []
        for i in range(60):
            angle = (i / 60) * 2 * _math.pi
            lat = 12.0 + 8 * _math.sin(angle) + (i % 5) * 0.7
            lon = 80.0 + 10 * _math.cos(angle * 0.7) + (i % 3) * 0.9
            demo.append({"type": "Feature",
                "properties": {"float_id": f"ARGO_{5900000+i}", "sst": round(28+_math.sin(i)*2,2),
                               "salinity": round(34.5+_math.cos(i)*0.5,2), "depth_m": 2000},
                "geometry": {"type": "Point", "coordinates": [round(lon,3), round(lat,3)]}})
        return jr({"type": "FeatureCollection", "features": demo, "total": len(demo)})
    features = [{"type":"Feature","properties":{"float_id":f.get("float_id",""),
                 "sst":f.get("sst"),"salinity":f.get("salinity")},
                 "geometry":{"type":"Point","coordinates":[f.get("longitude",0),f.get("latitude",0)]}}
                for f in floats]
    return jr({"type": "FeatureCollection", "features": features, "total": len(features)})


# ── /api/map/ships ───────────────────────────────────────────
@app.get("/api/map/ships")
def api_map_ships():
    vessel_type = request.args.get("type")
    bbox = _parse_bbox()
    conn = _gis_db()
    data = gis.get_ais_vessels(conn, vessel_type=vessel_type, bbox=bbox)
    conn.close()
    return jr(data)


@app.patch("/api/map/ships/<mmsi>")
def api_map_ships_update(mmsi):
    d = request.json or {}
    conn = _gis_db()
    result = gis.update_vessel_position(conn, mmsi, float(d.get("lat",0)),
                                        float(d.get("lon",0)), float(d.get("speed",0)),
                                        float(d.get("heading",0)))
    conn.close()
    return jr(result)


# ── /api/map/grid ────────────────────────────────────────────
@app.get("/api/map/grid")
def api_map_grid():
    lat_min = float(request.args.get("lat_min", -5))
    lat_max = float(request.args.get("lat_max", 25))
    lon_min = float(request.args.get("lon_min", 60))
    lon_max = float(request.args.get("lon_max", 100))
    res     = float(request.args.get("res", 0.25))
    # Cap to avoid huge responses
    lat_range = abs(lat_max - lat_min)
    lon_range = abs(lon_max - lon_min)
    if lat_range * lon_range / (res * res) > 2000:
        res = max(res, 1.0)
    data = gis.get_grid_cells(lat_min, lat_max, lon_min, lon_max, res)
    return jr(data)


@app.get("/api/map/grid/snap")
def api_map_grid_snap():
    lat = float(request.args.get("lat", 0))
    lon = float(request.args.get("lon", 0))
    res = float(request.args.get("res", 0.25))
    return jr(gis.snap_to_grid(lat, lon, res))


# ── /api/map/location ────────────────────────────────────────
@app.get("/api/map/location")
def api_map_location():
    q = request.args.get("q", "").strip()
    if not q:
        return jr({"error": "Query required"}, 400)
    conn = _gis_db()
    results = gis.search_location(conn, q)
    conn.close()
    return jr({"results": results, "total": len(results), "query": q})


# ── /api/map/bookmarks ───────────────────────────────────────
@app.get("/api/map/bookmarks")
def api_map_bookmarks_list():
    user_id = int(request.args.get("user_id", 0))
    conn = _gis_db()
    bms = gis.list_gis_bookmarks(conn, user_id)
    conn.close()
    return jr({"bookmarks": bms, "total": len(bms)})


@app.post("/api/map/bookmarks")
def api_map_bookmarks_add():
    d = request.json or {}
    conn = _gis_db()
    result = gis.add_gis_bookmark(
        conn, int(d.get("user_id", 0)), d.get("name","Bookmark"),
        float(d.get("lat", 0)), float(d.get("lon", 0)),
        float(d.get("zoom", 5)), float(d.get("bearing", 0)),
        float(d.get("pitch", 0)), d.get("active_layers", []),
        d.get("basemap", "ocean"), d.get("notes", ""))
    conn.close()
    return jr(result)


@app.delete("/api/map/bookmarks/<int:bm_id>")
def api_map_bookmarks_delete(bm_id):
    user_id = int(request.args.get("user_id", 0))
    conn = _gis_db()
    result = gis.delete_gis_bookmark(conn, bm_id, user_id)
    conn.close()
    return jr(result)


# ── /api/map/measure ─────────────────────────────────────────
@app.post("/api/map/measure")
def api_map_measure():
    d = request.json or {}
    points = d.get("points", [])
    if len(points) < 2:
        return jr({"error": "At least 2 points required"}, 400)
    result = gis.calculate_measurement(points)
    if d.get("save"):
        conn = _gis_db()
        gis.save_measurement(conn, int(d.get("user_id", 0)), points, d.get("label",""))
        conn.close()
    return jr(result)


@app.get("/api/map/measure")
def api_map_measure_list():
    user_id = int(request.args.get("user_id", 0))
    conn = _gis_db()
    ms = gis.list_measurements(conn, user_id)
    conn.close()
    return jr({"measurements": ms, "total": len(ms)})


# ── /api/map/draw ────────────────────────────────────────────
@app.post("/api/map/draw")
def api_map_draw_save():
    d = request.json or {}
    conn = _gis_db()
    result = gis.save_drawn_feature(conn, int(d.get("user_id", 0)),
                                    d.get("feature_type","polygon"), d.get("geojson",{}),
                                    d.get("label",""), d.get("color","#22e5ff"))
    conn.close()
    return jr(result)


@app.get("/api/map/draw")
def api_map_draw_list():
    user_id = int(request.args.get("user_id", 0))
    conn = _gis_db()
    feats = gis.list_drawn_features(conn, user_id)
    conn.close()
    return jr({"features": feats, "total": len(feats)})


@app.delete("/api/map/draw/<int:feat_id>")
def api_map_draw_delete(feat_id):
    user_id = int(request.args.get("user_id", 0))
    conn = _gis_db()
    result = gis.delete_drawn_feature(conn, feat_id, user_id)
    conn.close()
    return jr(result)


# ── /api/map/inspect ────────────────────────────────────────
@app.get("/api/map/inspect")
def api_map_inspect():
    """Full coordinate inspector — all layers at a lat/lon."""
    lat = float(request.args.get("lat", 12.5))
    lon = float(request.args.get("lon", 82.5))

    # Grid snap
    grid = gis.snap_to_grid(lat, lon)

    # Distance from coast
    dist_coast = gis.distance_from_coast_km(lat, lon)

    # Nearest ARGO float (from state)
    nearest_argo = None
    min_argo_d = float("inf")
    for fl in (_state.get("p3_argo_result", {}).get("floats") or []):
        d = gis.haversine_km(lat, lon, fl.get("latitude",0), fl.get("longitude",0))
        if d < min_argo_d:
            min_argo_d = d
            nearest_argo = {"float_id": fl.get("float_id"), "distance_km": round(d,1)}

    # SST (synthetic if no model)
    base = 28.0 + 4.0 * math.cos((lat - 5) * math.pi / 30)
    noise = 1.5 * (math.sin(lat * 3.7 + lon * 2.1) + math.cos(lon * 4.3) * 0.5)
    sst_synthetic = round(min(36.0, max(24.0, base + noise)), 2)

    # Actual grid cell from state if available
    grid_cell_data = {}
    if _state.get("p3_surface_ds") and _state.get("p3_pred_ds"):
        try:
            surf = _state["p3_surface_ds"]
            lats_arr = surf["latitude"].values
            lons_arr = surf["longitude"].values
            li = int(np.argmin(np.abs(lats_arr - lat)))
            oi = int(np.argmin(np.abs(lons_arr - lon)))
            def _v(var):
                if var in surf.data_vars:
                    v = float(surf[var].isel(time=-1, latitude=li, longitude=oi).values)
                    return v if math.isfinite(v) else None
                return None
            grid_cell_data = {"sst": _v("sst"), "sss": _v("sss"), "sla": _v("sla")}
        except Exception:
            pass

    # Heatwave at point
    hw_status = "NONE"
    for hs in (_state.get("p3_hotspots") or []):
        if abs(hs.get("latitude",0) - lat) < 0.2 and abs(hs.get("longitude",0) - lon) < 0.2:
            hw_status = hs.get("severity","NONE")
            break

    return jr({
        "latitude": lat, "longitude": lon,
        "grid": grid,
        "distance_from_coast_km": dist_coast,
        "nearest_argo": nearest_argo,
        "sst": grid_cell_data.get("sst") or sst_synthetic,
        "sss": grid_cell_data.get("sss"),
        "sla": grid_cell_data.get("sla"),
        "heatwave_status": hw_status,
    })


# ── /api/map/stats ───────────────────────────────────────────
@app.get("/api/map/stats")
def api_map_stats():
    conn = _gis_db()
    stats = gis.get_gis_stats(conn)
    conn.close()
    return jr(stats)


# ── /api/map/export ─────────────────────────────────────────
@app.get("/api/map/export/geojson")
def api_map_export_geojson():
    """Export all GIS features as a single GeoJSON FeatureCollection."""
    export_type = request.args.get("type", "ports")
    conn = _gis_db()
    if export_type == "ports":
        data = gis.get_ports(conn)
    elif export_type == "routes":
        data = gis.get_shipping_routes(conn)
    elif export_type == "areas":
        data = gis.get_marine_areas(conn)
    elif export_type == "bathymetry":
        data = gis.get_bathymetry(conn)
    elif export_type == "ships":
        data = gis.get_ais_vessels(conn)
    elif export_type == "sst":
        data = gis.generate_sst_geojson()
    else:
        data = {"type": "FeatureCollection", "features": [], "error": "Unknown type"}
    conn.close()
    response = app.response_class(
        response=json.dumps(data, indent=2),
        status=200,
        mimetype="application/geo+json",
        headers={"Content-Disposition": f"attachment; filename=oceanverse_{export_type}.geojson"}
    )
    return response


# ── Helper ───────────────────────────────────────────────────
def _parse_bbox():
    try:
        return {
            "lat_min": float(request.args.get("lat_min", -90)),
            "lat_max": float(request.args.get("lat_max", 90)),
            "lon_min": float(request.args.get("lon_min", -180)),
            "lon_max": float(request.args.get("lon_max", 180)),
        }
    except (ValueError, TypeError):
        return None



# ══════════════════════════════════════════════════════════════
# PHASE 3 — DIGITAL TWIN APIs  (Sections 24 + extras)
# ══════════════════════════════════════════════════════════════

import digital_twin as dt

@app.route("/twin")
def page_twin():
    return render_template("twin.html")

# ── /api/digital-twin (extended) ─────────────────────────────
@app.get("/api/twin/scene")
def api_twin_scene():
    hotspots = _state.get("p3_hotspots") or []
    return jr(dt.get_scene_descriptor(hotspots))

# ── /api/ocean-volume ────────────────────────────────────────
@app.get("/api/ocean-volume")
def api_ocean_volume():
    hotspots = _state.get("p3_hotspots") or []
    raw = request.args.get("depth_indices")
    if raw:
        try:
            depth_indices = [int(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            depth_indices = None
    else:
        depth_indices = None
    data = dt.get_ocean_volume(hotspots=hotspots, depth_indices=depth_indices)
    return jr(data)

# ── /api/depth-layer ─────────────────────────────────────────
@app.get("/api/depth-layer")
def api_depth_layer():
    depth_idx = int(request.args.get("depth_idx", 0))
    hotspots  = _state.get("p3_hotspots") or []
    data = dt.get_depth_layer(depth_idx=depth_idx, hotspots=hotspots)
    return jr(data)

# ── /api/thermocline ─────────────────────────────────────────
@app.get("/api/thermocline")
def api_thermocline():
    hotspots = _state.get("p3_hotspots") or []
    return jr(dt.get_thermocline_surface(hotspots=hotspots))

# ── /api/particle-data ───────────────────────────────────────
@app.get("/api/particle-data")
def api_particle_data():
    ptype = request.args.get("type", "current")
    n     = int(request.args.get("n", 400))
    n     = min(max(n, 10), 800)
    data  = dt.get_particle_data(n_particles=n, particle_type=ptype)
    return jr(data)

# ── /api/argo-animation ──────────────────────────────────────
@app.get("/api/argo-animation")
def api_argo_animation():
    argo_result = _state.get("p3_argo_result") or {}
    floats = argo_result.get("floats") or []
    data = dt.get_argo_animation(floats=floats if floats else None)
    return jr(data)

# ── /api/satellite-orbits ────────────────────────────────────
@app.get("/api/satellite-orbits")
def api_satellite_orbits():
    n_frames = int(request.args.get("n_frames", 60))
    return jr(dt.get_satellite_orbits(n_frames=n_frames))

# ── /api/marine-objects ──────────────────────────────────────
@app.get("/api/marine-objects")
def api_marine_objects():
    return jr(dt.get_marine_objects())

# ── /api/bathymetry-terrain ──────────────────────────────────
@app.get("/api/bathymetry-terrain")
def api_bathymetry_terrain():
    res = request.args.get("resolution", "medium")
    return jr(dt.get_bathymetry_terrain(resolution=res))

# ── /api/heatwave-glow ───────────────────────────────────────
@app.get("/api/heatwave-glow")
def api_heatwave_glow():
    hotspots = _state.get("p3_hotspots") or []
    return jr(dt.get_heatwave_glow(hotspots=hotspots))

# ── /api/coral-hotspots ──────────────────────────────────────
@app.get("/api/coral-hotspots")
def api_coral_hotspots():
    return jr(dt.get_coral_hotspots())

# ── /api/wave-surface ────────────────────────────────────────
@app.get("/api/wave-surface")
def api_wave_surface():
    n = int(request.args.get("n_waves", 8))
    return jr(dt.get_wave_surface(n_waves=n))

# ── /api/twin/timeline ───────────────────────────────────────
@app.get("/api/twin/timeline")
def api_twin_timeline():
    hotspots = _state.get("p3_hotspots") or []
    n_years  = int(request.args.get("n_years", 10))
    return jr(dt.get_timeline_frames(n_years=n_years, hotspots=hotspots))

# ── /api/twin/explain ────────────────────────────────────────
@app.get("/api/twin/explain")
def api_twin_explain():
    lat = float(request.args.get("lat", 12.5))
    lon = float(request.args.get("lon", 82.5))
    hotspots = _state.get("p3_hotspots") or []
    return jr(dt.get_ai_explanation_hud(lat, lon, hotspots))

# ── /api/twin/vr ─────────────────────────────────────────────
@app.get("/api/twin/vr")
def api_twin_vr():
    return jr(dt.get_vr_descriptor())

# ── /api/twin/lod ────────────────────────────────────────────
@app.get("/api/twin/lod")
def api_twin_lod():
    return jr(dt.get_lod_descriptor())

# ── /api/twin/stats ──────────────────────────────────────────
@app.get("/api/twin/stats")
def api_twin_stats():
    hotspots = _state.get("p3_hotspots") or []
    coral = dt.get_coral_hotspots()
    marine = dt.get_marine_objects()
    return jr({
        "depth_levels": len(dt.DEPTH_LEVELS_M),
        "satellites": len(dt.SATELLITES),
        "coral_sites": coral["site_count"],
        "bleached_reefs": coral["bleached"],
        "marine_objects": marine["total"],
        "hotspots": len(hotspots),
        "region": dt.REGION,
        "earth_radius": dt.EARTH_RADIUS,
    })

