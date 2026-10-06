"""
argo_validation.py
OceanVerse AI v4 — Real ARGO Validation Pipeline
SIH Problem Statement 26066 — OceanEmbed

Implements:
  - ARGO file discovery (NetCDF + CSV)
  - Quality control (coordinates, timestamps, temperatures, QC flags)
  - Spatiotemporal matching with configurable tolerances
  - Full metrics: RMSE, MAE, Bias, Correlation, R², SDE
  - Depth-wise and region-wise breakdowns
  - Explicit validation states (DATA_NOT_AVAILABLE / MATCHING_FAILED / VALIDATED)
  - Report generation

Region (North Indian Ocean):
  Lat: 5°N–30°N  |  Lon: 45°E–105°E
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("ocean_platform")

# ──────────────────────────────────────────────────────────────
# CONSTANTS  (all configurable via cfg)
# ──────────────────────────────────────────────────────────────

DEFAULT_REGION = {
    "lat_min": 5.0, "lat_max": 30.0,
    "lon_min": 45.0, "lon_max": 105.0,
}

# Standard SIH target depths (metres)
TARGET_DEPTHS_M = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# Physical temperature limits for the Indian Ocean (realistic range)
TEMP_MIN_C = -2.0
TEMP_MAX_C = 35.0

# Default matching tolerances (overridable in cfg)
MAX_TIME_DELTA_DAYS = 3
MAX_DISTANCE_KM     = 50.0
MAX_DEPTH_DELTA_M   = 25.0

# Sub-regions for regional breakdown
SUB_REGIONS = {
    "Arabian Sea":        {"lat_min": 5,  "lat_max": 25, "lon_min": 45, "lon_max": 77},
    "Bay of Bengal":      {"lat_min": 5,  "lat_max": 25, "lon_min": 77, "lon_max": 100},
    "North Indian Ocean": {"lat_min": 5,  "lat_max": 30, "lon_min": 45, "lon_max": 105},
}

# ARGO QC: accept quality flags 1 (good) and 2 (probably good)
GOOD_QC_FLAGS = {1, 2}


# ──────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    R = 6371.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def _metrics(obs: np.ndarray, pred: np.ndarray) -> dict:
    """Compute all validation metrics from paired arrays."""
    if len(obs) == 0:
        return {"n": 0, "rmse": None, "mae": None, "bias": None,
                "correlation": None, "r2": None, "sde": None}
    diff   = pred - obs
    rmse   = float(np.sqrt(np.mean(diff ** 2)))
    mae    = float(np.mean(np.abs(diff)))
    bias   = float(np.mean(diff))
    sde    = float(np.std(diff))
    if len(obs) > 2:
        corr = float(np.corrcoef(obs, pred)[0, 1])
        ss_res = np.sum(diff ** 2)
        ss_tot = np.sum((obs - obs.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
    else:
        corr = float("nan")
        r2   = float("nan")
    return {"n": int(len(obs)), "rmse": round(rmse, 4), "mae": round(mae, 4),
            "bias": round(bias, 4), "correlation": round(corr, 4),
            "r2": round(r2, 4), "sde": round(sde, 4)}


# ──────────────────────────────────────────────────────────────
# ARGO QUALITY CONTROL
# ──────────────────────────────────────────────────────────────

def qc_argo_profile(lat: float, lon: float, juld,
                    temps: np.ndarray, pres: np.ndarray,
                    temp_qc: np.ndarray, pos_qc: float,
                    region: dict) -> tuple:
    """
    Apply QC checks to one ARGO profile.

    Returns (valid: bool, reason: str, temps_clean, pres_clean)
    """
    # --- Coordinate check ---
    if not (np.isfinite(lat) and np.isfinite(lon)):
        return False, "non_finite_coordinates", None, None
    if not (region["lat_min"] <= lat <= region["lat_max"] and
            region["lon_min"] <= lon <= region["lon_max"]):
        return False, "out_of_region", None, None
    if float(pos_qc) not in GOOD_QC_FLAGS:
        return False, "bad_position_qc", None, None

    # --- Timestamp check ---
    try:
        t = pd.Timestamp(juld)
        if pd.isnull(t):
            return False, "invalid_timestamp", None, None
    except Exception:
        return False, "invalid_timestamp", None, None

    # --- Temperature / pressure ---
    if temps is None or pres is None:
        return False, "missing_temp_pres", None, None

    temps_arr = np.asarray(temps, dtype="float64")
    pres_arr  = np.asarray(pres,  dtype="float64")
    tqc_arr   = np.asarray(temp_qc, dtype="float32") if temp_qc is not None else np.ones_like(temps_arr)

    # Mask: finite, in physical range, good QC flag
    mask = (
        np.isfinite(temps_arr) &
        (temps_arr >= TEMP_MIN_C) & (temps_arr <= TEMP_MAX_C) &
        np.array([float(q) in GOOD_QC_FLAGS for q in tqc_arr.flat], dtype=bool) &
        np.isfinite(pres_arr) & (pres_arr >= 0) & (pres_arr < 7000)
    )

    temps_clean = temps_arr[mask]
    pres_clean  = pres_arr[mask]

    if len(temps_clean) < 3:
        return False, "insufficient_valid_levels", None, None

    return True, "ok", temps_clean, pres_clean


# ──────────────────────────────────────────────────────────────
# ARGO LOADER
# ──────────────────────────────────────────────────────────────

def load_argo_profiles(argo_dir: str, region: dict) -> tuple:
    """
    Discover and load all ARGO files (.nc and .csv).

    Returns:
        profiles   — list of dicts with keys: lat, lon, time, temps, pres
        qc_stats   — QC statistics dict
    """
    argo_path = Path(argo_dir)
    nc_files  = list(argo_path.glob("*.nc"))
    csv_files = list(argo_path.glob("*.csv"))
    all_files = nc_files + csv_files

    if not all_files:
        return [], {"status": "NO_FILES", "total": 0, "valid": 0, "rejected": 0,
                    "reject_reasons": {}}

    qc_stats = {
        "total": 0, "valid": 0, "rejected": 0,
        "reject_reasons": {},
        "files_read": len(all_files),
        "files_failed": 0,
    }
    profiles = []

    for fp in all_files:
        try:
            if str(fp).endswith(".nc"):
                _load_nc_file(fp, region, profiles, qc_stats)
            else:
                _load_csv_file(fp, region, profiles, qc_stats)
        except Exception as e:
            logger.warning(f"ARGO file error {fp}: {e}")
            qc_stats["files_failed"] += 1

    qc_stats["depth_coverage_m"] = TARGET_DEPTHS_M
    qc_stats["temporal_coverage"] = (
        [str(min(p["time"] for p in profiles)),
         str(max(p["time"] for p in profiles))]
        if profiles else None
    )
    qc_stats["spatial_coverage"] = (
        {"lat_min": float(min(p["lat"] for p in profiles)),
         "lat_max": float(max(p["lat"] for p in profiles)),
         "lon_min": float(min(p["lon"] for p in profiles)),
         "lon_max": float(max(p["lon"] for p in profiles))}
        if profiles else None
    )
    qc_stats["profiles_used_for_validation"] = qc_stats["valid"]

    return profiles, qc_stats


def _load_nc_file(fp: Path, region: dict, profiles: list, qc_stats: dict):
    """Load profiles from one ARGO-format NetCDF."""
    ds = xr.open_dataset(str(fp), engine="netcdf4")

    lat_v  = "LATITUDE"  if "LATITUDE"   in ds else "latitude"
    lon_v  = "LONGITUDE" if "LONGITUDE"  in ds else "longitude"
    time_v = "JULD"      if "JULD"       in ds else "time"
    temp_v = "TEMP"      if "TEMP"       in ds else "temperature"
    pres_v = "PRES"      if "PRES"       in ds else "pressure"
    tqc_v  = "TEMP_QC"   if "TEMP_QC"   in ds else None
    pqc_v  = "POSITION_QC" if "POSITION_QC" in ds else None

    lats   = ds[lat_v].values
    lons   = ds[lon_v].values
    times  = ds[time_v].values
    temps  = ds[temp_v].values
    press  = ds[pres_v].values
    tqc    = ds[tqc_v].values if tqc_v else None
    posqc  = ds[pqc_v].values if pqc_v else np.ones(len(lats))

    n_prof = len(lats)
    for ip in range(n_prof):
        qc_stats["total"] += 1
        t_ip   = temps[ip] if temps.ndim == 2 else temps
        p_ip   = press[ip] if press.ndim == 2 else press
        tq_ip  = tqc[ip]   if (tqc is not None and tqc.ndim == 2) else (tqc or np.ones_like(t_ip))
        pos_ip = float(posqc[ip]) if posqc is not None else 1.0

        ok, reason, t_clean, p_clean = qc_argo_profile(
            float(lats[ip]), float(lons[ip]), times[ip],
            t_ip, p_ip, tq_ip, pos_ip, region
        )
        if ok:
            profiles.append({
                "lat":   float(lats[ip]),
                "lon":   float(lons[ip]),
                "time":  pd.Timestamp(times[ip]),
                "temps": t_clean,
                "pres":  p_clean,
                "source": str(fp.name),
            })
            qc_stats["valid"] += 1
        else:
            qc_stats["rejected"] += 1
            qc_stats["reject_reasons"][reason] = qc_stats["reject_reasons"].get(reason, 0) + 1

    ds.close()


def _load_csv_file(fp: Path, region: dict, profiles: list, qc_stats: dict):
    """Load profiles from CSV. Expects columns: lat,lon,time,depth,temperature."""
    df = pd.read_csv(str(fp))
    required = {"lat", "lon", "time", "depth", "temperature"}
    if not required.issubset(df.columns):
        logger.warning(f"CSV {fp} missing columns {required - set(df.columns)}")
        return

    for group_key, grp in df.groupby(["lat", "lon", "time"]):
        qc_stats["total"] += 1
        lat, lon, t = float(grp["lat"].iloc[0]), float(grp["lon"].iloc[0]), grp["time"].iloc[0]
        pres  = grp["depth"].values.astype("float64")
        temps = grp["temperature"].values.astype("float64")
        tqc   = np.ones_like(temps)

        ok, reason, t_clean, p_clean = qc_argo_profile(
            lat, lon, t, temps, pres, tqc, 1.0, region
        )
        if ok:
            profiles.append({
                "lat": lat, "lon": lon, "time": pd.Timestamp(t),
                "temps": t_clean, "pres": p_clean, "source": str(fp.name),
            })
            qc_stats["valid"] += 1
        else:
            qc_stats["rejected"] += 1
            qc_stats["reject_reasons"][reason] = qc_stats["reject_reasons"].get(reason, 0) + 1


# ──────────────────────────────────────────────────────────────
# SPATIOTEMPORAL MATCHING
# ──────────────────────────────────────────────────────────────

def match_profiles_to_predictions(profiles: list, pred_ds: xr.Dataset,
                                   cfg: dict) -> list:
    """
    Match each ARGO profile to the nearest model prediction in
    space and time, then interpolate to the 15 target depths.

    Returns list of matched pairs: {depth, obs_temp, pred_temp, lat, lon, time}
    """
    tol_cfg  = cfg.get("matching_tolerances", {})
    max_days = tol_cfg.get("max_time_delta_days", MAX_TIME_DELTA_DAYS)
    max_km   = tol_cfg.get("max_distance_km",    MAX_DISTANCE_KM)

    pred_lats  = pred_ds["latitude"].values
    pred_lons  = pred_ds["longitude"].values
    pred_times = pd.DatetimeIndex(pred_ds["time"].values)
    pred_depths = np.array(pred_ds["depth"].values, dtype="float64")

    pairs = []

    for profile in profiles:
        p_lat  = profile["lat"]
        p_lon  = profile["lon"]
        p_time = profile["time"]

        # --- Find nearest time in predictions ---
        time_diffs = np.abs((pred_times - p_time).days.astype("float64"))
        t_idx = int(np.argmin(time_diffs))
        if float(time_diffs[t_idx]) > max_days:
            continue

        # --- Find nearest grid point ---
        lat_idx = int(np.argmin(np.abs(pred_lats - p_lat)))
        lon_idx = int(np.argmin(np.abs(pred_lons - p_lon)))

        dist_km = _haversine_km(p_lat, p_lon,
                                float(pred_lats[lat_idx]),
                                float(pred_lons[lon_idx]))
        if dist_km > max_km:
            continue

        # --- Extract model profile at this grid point + time ---
        try:
            pred_profile = pred_ds["thetao_pred"].isel(
                time=t_idx, latitude=lat_idx, longitude=lon_idx
            ).values.astype("float64")   # (D,)
        except Exception:
            continue

        # --- Interpolate ARGO to model standard depths ---
        argo_pres  = profile["pres"]
        argo_temps = profile["temps"]

        for di, target_depth in enumerate(pred_depths):
            # Find ARGO measurements within MAX_DEPTH_DELTA of this depth
            depth_diffs = np.abs(argo_pres - target_depth)
            nearest_idx = int(np.argmin(depth_diffs))
            if float(depth_diffs[nearest_idx]) > MAX_DEPTH_DELTA_M:
                continue
            obs_t  = float(argo_temps[nearest_idx])
            pred_t = float(pred_profile[di])

            if not (np.isfinite(obs_t) and np.isfinite(pred_t)):
                continue
            if not (TEMP_MIN_C <= obs_t <= TEMP_MAX_C):
                continue

            pairs.append({
                "depth":     float(target_depth),
                "obs_temp":  obs_t,
                "pred_temp": pred_t,
                "lat":       p_lat,
                "lon":       p_lon,
                "time":      str(p_time.date()),
                "dist_km":   round(dist_km, 2),
                "time_delta_days": float(time_diffs[t_idx]),
            })

    return pairs


# ──────────────────────────────────────────────────────────────
# FULL VALIDATION PIPELINE
# ──────────────────────────────────────────────────────────────

def run_argo_validation(pred_ds: xr.Dataset, argo_dir: str, cfg: dict) -> dict:
    """
    Main entry point: full ARGO validation pipeline.

    Returns structured result with explicit status codes:
      DATA_NOT_AVAILABLE  — no ARGO files found
      MATCHING_FAILED     — files found but no pairs could be matched
      VALIDATED           — validation completed with real metrics
    """
    logger.info("Starting ARGO validation pipeline...")

    region = cfg.get("region", {}).get("full", DEFAULT_REGION)

    # --- Step 1: Load + QC ARGO profiles ---
    profiles, qc_stats = load_argo_profiles(argo_dir, region)

    if not profiles:
        return {
            "status": "DATA_NOT_AVAILABLE",
            "status_detail": "No valid ARGO profiles found in the project region after QC",
            "profiles_matched": 0,
            "qc_stats": qc_stats,
            "rmse": None, "mae": None, "bias": None,
            "correlation": None, "r2": None,
            "depth_metrics": {},
            "region_metrics": {},
        }

    # --- Step 2: Spatiotemporal matching ---
    pairs = match_profiles_to_predictions(profiles, pred_ds, cfg)

    if not pairs:
        return {
            "status": "MATCHING_FAILED",
            "status_detail": ("ARGO profiles exist but no spatiotemporal matches found. "
                              "Check tolerances or date ranges."),
            "profiles_matched": 0,
            "qc_stats": qc_stats,
            "rmse": None, "mae": None, "bias": None,
            "correlation": None, "r2": None,
            "depth_metrics": {},
            "region_metrics": {},
        }

    # --- Step 3: Overall metrics ---
    obs_arr  = np.array([p["obs_temp"]  for p in pairs])
    pred_arr = np.array([p["pred_temp"] for p in pairs])
    overall  = _metrics(obs_arr, pred_arr)

    # --- Step 4: Depth-wise metrics ---
    depth_metrics = {}
    for d in TARGET_DEPTHS_M:
        d_pairs = [p for p in pairs if p["depth"] == d]
        if d_pairs:
            o = np.array([p["obs_temp"]  for p in d_pairs])
            pr = np.array([p["pred_temp"] for p in d_pairs])
            depth_metrics[d] = _metrics(o, pr)

    # --- Step 5: Region-wise metrics ---
    region_metrics = {}
    for rname, rbounds in SUB_REGIONS.items():
        r_pairs = [
            p for p in pairs
            if (rbounds["lat_min"] <= p["lat"] <= rbounds["lat_max"] and
                rbounds["lon_min"] <= p["lon"] <= rbounds["lon_max"])
        ]
        if r_pairs:
            o  = np.array([p["obs_temp"]  for p in r_pairs])
            pr = np.array([p["pred_temp"] for p in r_pairs])
            region_metrics[rname] = _metrics(o, pr)

    logger.info(f"ARGO validation complete: {len(pairs)} matched pairs, "
                f"RMSE={overall['rmse']}, Corr={overall['correlation']}")

    return {
        "status":          "VALIDATED",
        "status_detail":   f"Validated against {qc_stats['valid']} ARGO profiles, {len(pairs)} depth-matched pairs",
        "profiles_found":  qc_stats["total"],
        "profiles_valid":  qc_stats["valid"],
        "profiles_matched": qc_stats["valid"],
        "n_obs":           len(pairs),
        "qc_stats":        qc_stats,
        # Overall metrics
        "rmse":            overall["rmse"],
        "mae":             overall["mae"],
        "bias":            overall["bias"],
        "correlation":     overall["correlation"],
        "r2":              overall["r2"],
        "sde":             overall["sde"],
        # Breakdowns
        "depth_metrics":   depth_metrics,
        "region_metrics":  region_metrics,
        # Raw pairs for export
        "_pairs":          pairs,
    }


# ──────────────────────────────────────────────────────────────
# REPORT WRITERS
# ──────────────────────────────────────────────────────────────

def write_argo_reports(result: dict, report_dir: str) -> None:
    """Write all ARGO validation reports to disk."""
    os.makedirs(report_dir, exist_ok=True)

    # --- JSON summary ---
    summary = {k: v for k, v in result.items() if k != "_pairs"}
    # Convert depth keys to strings for JSON
    if "depth_metrics" in summary:
        summary["depth_metrics"] = {str(k): v for k, v in summary["depth_metrics"].items()}
    summary_path = os.path.join(report_dir, "argo_validation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"ARGO summary → {summary_path}")

    # --- Depth-wise CSV ---
    rows = []
    for d, m in result.get("depth_metrics", {}).items():
        rows.append({"depth_m": d, **m})
    if rows:
        csv_path = os.path.join(report_dir, "argo_validation_depthwise.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        logger.info(f"Depth-wise CSV → {csv_path}")

    # --- Pairs CSV ---
    pairs = result.get("_pairs", [])
    if pairs:
        pairs_path = os.path.join(report_dir, "argo_matched_pairs.csv")
        pd.DataFrame(pairs).drop(columns=["_pairs"], errors="ignore").to_csv(pairs_path, index=False)

    # --- Markdown report ---
    status = result.get("status", "UNKNOWN")
    lines  = [
        "# ARGO Validation Report — OceanVerse AI v4",
        "",
        f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
        f"**Status:** {status}",
        f"**Detail:** {result.get('status_detail','')}",
        "",
        "---",
        "",
    ]

    if status == "VALIDATED":
        lines += [
            "## QC Statistics",
            f"- Total profiles found: {result.get('profiles_found', 0)}",
            f"- Valid after QC: {result.get('profiles_valid', 0)}",
            f"- Matched pairs: {result.get('n_obs', 0)}",
        ]
        reject = result.get("qc_stats", {}).get("reject_reasons", {})
        if reject:
            lines.append("- Rejection reasons:")
            for reason, count in reject.items():
                lines.append(f"  - {reason}: {count}")
        lines += [
            "",
            "## Overall Metrics",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| RMSE   | {result.get('rmse')} °C |",
            f"| MAE    | {result.get('mae')} °C |",
            f"| Bias   | {result.get('bias')} °C |",
            f"| Correlation | {result.get('correlation')} |",
            f"| R²     | {result.get('r2')} |",
            f"| Std Dev Error | {result.get('sde')} °C |",
            "",
            "## Depth-wise Metrics",
            "| Depth (m) | RMSE | MAE | Bias | Corr | R² | N |",
            "|-----------|------|-----|------|------|----|---|",
        ]
        for d, m in sorted(result.get("depth_metrics", {}).items()):
            lines.append(
                f"| {d} | {m.get('rmse')} | {m.get('mae')} | "
                f"{m.get('bias')} | {m.get('correlation')} | "
                f"{m.get('r2')} | {m.get('n')} |"
            )
        lines += [
            "",
            "## Regional Metrics",
            "| Region | RMSE | MAE | Bias | Corr | N |",
            "|--------|------|-----|------|------|---|",
        ]
        for rname, m in result.get("region_metrics", {}).items():
            lines.append(
                f"| {rname} | {m.get('rmse')} | {m.get('mae')} | "
                f"{m.get('bias')} | {m.get('correlation')} | {m.get('n')} |"
            )
    else:
        lines += [
            "## Data Availability",
            f"- ARGO files found: {result.get('qc_stats', {}).get('files_read', 0)}",
            f"- Profiles after QC: {result.get('profiles_valid', 0)}",
            "",
            "> **Note:** Validation cannot proceed without matching ARGO observations.",
            "> Obtain ARGO data from https://argo.ucsd.edu/data/data-from-gdacs/ or",
            "> Copernicus Marine Service and place .nc files in data/raw/argo/",
        ]

    lines += ["", "---", "*OceanVerse AI v4 — SIH 26066 OceanEmbed*"]
    md_path = os.path.join(report_dir, "argo_validation_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Markdown report → {md_path}")
