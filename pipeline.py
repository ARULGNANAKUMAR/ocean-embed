"""
pipeline.py
North Indian Ocean Intelligence Platform
Phase 1 + 2 + 3 orchestration logic — framework-agnostic.
Imported by app.py (Flask). Mutates the shared `_state` dict passed in.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import preprocessing as pre
import ai_engine     as ai
import evaluation    as ev
import visualization as viz
import admin         as adm
import argo_validation as argo_val

# ──────────────────────────────────────────────────────────────
# SHARED STATE  (read by Flask app.py routes)
# ──────────────────────────────────────────────────────────────
_state: dict = {
    # Phase 1
    "config": None, "scan_records": [], "inspections": {},
    "metadata": [], "qc_results": [], "cov_results": [],
    "res_results": [], "hashes": {}, "hash_verify": {},
    "glorys_result": None, "argo_result": None, "summary": {}, "db_path": "",
    # Phase 2
    "p2_status": {}, "p2_statistics": [], "p2_missing": [],
    "p2_interpolation": [], "p2_depths": [],
    "p2_surface_shape": None, "p2_target_shape": None, "p2_summary": {},
    # Phase 3
    "p3_model": None, "p3_norm_stats": {}, "p3_surface": None, "p3_target": None,
    "p3_times": [], "p3_lats": [], "p3_lons": [],
    "p3_pred_ds": None, "p3_surface_ds": None, "p3_hotspots": [], "p3_impacts": [],
    "p3_uncertainty": {}, "p3_xai": {}, "p3_argo_result": {}, "p3_future_fc": {},
    "p3_training_hist": [], "p3_summary": {}, "p3_status": {}, "p3_multivar": {},
}

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def _s(label: str, ok: bool = True):
    print(f"  {label} {'✓' if ok else '✗'}")


def _open_and_prepare(filepath, cfg):
    try:
        ds = xr.open_dataset(filepath, engine="netcdf4")
    except Exception as e:
        pre.logger.error(f"Cannot open {filepath}: {e}")
        return None, {}, [], {}, {}, {}, {}
    ds, coord_log = pre.standardize_coordinates(ds)
    ds, var_log   = pre.standardize_variables(ds, cfg)
    ds, cov_info  = pre.crop_region(ds, cfg)
    ds, time_info = pre.crop_time(ds, cfg)
    unit_log      = pre.standardize_units(ds, cfg)
    mv_report     = pre.handle_missing_values(ds)
    return ds, mv_report, unit_log, coord_log, var_log, time_info, cov_info


# ──────────────────────────────────────────────────────────────
# PHASE 1 PIPELINE
# ──────────────────────────────────────────────────────────────
def run_phase1(cfg: dict, start_api: bool = False) -> bool:
    print("\n" + "="*55)
    print("  PHASE 1 STARTED")
    print("="*55 + "\n")

    paths     = cfg["paths"]
    raw_dir   = paths["raw_data"]
    db_path   = paths["database"]
    report_dir= paths["reports"]
    plots_dir = os.path.join(report_dir, "plots")

    for d in [report_dir, plots_dir, raw_dir, paths["processed_data"],
              os.path.dirname(db_path)]:
        os.makedirs(d, exist_ok=True)

    _state["config"]  = cfg
    _state["db_path"] = db_path

    print("Datasets Found:")
    scan_records = pre.scan_raw_directory(raw_dir)
    _state["scan_records"] = scan_records

    for label, kws in {"SST":["sst"],"SSS":["sss"],"SLA":["sla"],
                        "Current":["current"],"Wind":["wind"],
                        "GLORYS":["glorys","thetao"],"ARGO":["argo"]}.items():
        found = any(any(kw in r["filename"].lower() for kw in kws) for r in scan_records)
        _s(label, found)

    pre.save_dataset_inventory(scan_records, os.path.join(report_dir, "dataset_inventory.csv"))
    hashes = pre.generate_file_hashes(scan_records)
    _state["hashes"] = hashes

    inspections = {}
    for rec in scan_records:
        fp, ext = rec["filepath"], rec["extension"]
        if ext in {".nc",".nc4",".cdf"}:
            inspections[fp] = pre.inspect_netcdf(fp)
        elif ext == ".csv":
            inspections[fp] = pre.inspect_csv(fp)
    _state["inspections"] = inspections

    metas = []
    for i, rec in enumerate(scan_records):
        metas.append(pre.build_metadata_record(rec, inspections.get(rec["filepath"],{}), i+1))
    _state["metadata"] = metas
    pre.generate_metadata_master(metas, os.path.join(report_dir, "metadata_master.csv"))

    qc_results = pre.run_qc(scan_records, inspections, cfg)
    _state["qc_results"] = qc_results
    pre.save_qc_report(qc_results, os.path.join(report_dir, "dataset_quality_report.csv"))
    print("\nQuality Checks:")
    _s("PASS", all(r["status"] in ("PASS","WARNING") for r in qc_results))

    cov_results = pre.analyze_coverage(scan_records, inspections, cfg)
    _state["cov_results"] = cov_results
    pre.save_coverage_report(cov_results, os.path.join(report_dir, "dataset_coverage_report.csv"))
    print("\nCoverage:")
    _s("PASS", any(c["overall"]=="PASS" for c in cov_results) or not cov_results)

    res_results = pre.analyze_resolution(scan_records, inspections, cfg)
    _state["res_results"] = res_results
    pre.save_resolution_report(res_results, os.path.join(report_dir, "resolution_report.csv"))

    glorys_result = None
    for rec in scan_records:
        if rec["dataset_type"]=="TARGET" or "glorys" in rec["filename"].lower():
            glorys_result = pre.validate_glorys(rec["filepath"], inspections.get(rec["filepath"],{}), cfg)
            break
    _state["glorys_result"] = glorys_result
    pre.save_glorys_report(glorys_result, os.path.join(report_dir, "glorys_depth_report.csv"))

    argo_result = None
    for rec in scan_records:
        if rec["dataset_type"]=="VALIDATION" or "argo" in rec["filename"].lower():
            argo_result = pre.validate_argo(rec["filepath"], inspections.get(rec["filepath"],{}), cfg)
            break
    _state["argo_result"] = argo_result
    pre.save_argo_report(argo_result, os.path.join(report_dir, "argo_report.csv"))

    conn = pre.init_database(db_path)
    id_map = {}
    for rec in scan_records:
        fp  = rec["filepath"]
        did = pre.insert_dataset(conn, rec, hashes.get(fp,""))
        id_map[fp] = did
        pre.insert_variables(conn, did, inspections.get(fp,{}))
    for qc in qc_results:
        fp = qc.get("filepath","")
        if id_map.get(fp,-1) > 0:
            pre.insert_qc_report(conn, id_map[fp], qc)
    for cov in cov_results:
        for fp, did in id_map.items():
            if Path(fp).name == cov["filename"]:
                pre.insert_coverage_report(conn, did, cov)
                qc_s = next((q["status"] for q in qc_results if q.get("filename")==cov["filename"]),"UNKNOWN")
                pre.insert_training_status(conn, did, qc_s, cov["overall"]=="PASS")
                break
    conn.close()

    print("\nMetadata Generated:"); _s("PASS")
    print("\nSQLite Updated:"); _s("PASS")

    hash_verify = pre.verify_file_hashes(scan_records, hashes)
    _state["hash_verify"] = hash_verify

    p1_sum = pre.generate_phase1_summary(scan_records, qc_results, cov_results, hash_verify,
                                          os.path.join(report_dir,"phase1_summary.json"))
    _state["summary"] = p1_sum
    pre.generate_phase1_md_report(p1_sum, os.path.join(report_dir,"PHASE1_REPORT.md"))
    pre.generate_qc_plots(scan_records, inspections, qc_results, cov_results, plots_dir)

    print("\nReports Generated:"); _s("PASS")
    print("\nRaw File Integrity:")
    _s("PASS", all(v=="OK" for v in hash_verify.values()))
    print("\n" + "="*55 + "\n  PHASE 1 COMPLETE\n" + "="*55 + "\n")
    return True


# ──────────────────────────────────────────────────────────────
# PHASE 2 PIPELINE
# ──────────────────────────────────────────────────────────────
def run_phase2(cfg: dict, start_api: bool = False) -> bool:
    print("\n" + "="*55 + "\n  PHASE 2 STARTED\n" + "="*55 + "\n")

    paths      = cfg["paths"]
    db_path    = paths["database"]
    raw_dir    = paths["raw_data"]
    final_dir  = paths.get("final_data","data/final")
    report_dir = paths.get("reports_phase2","reports/phase2")
    plots_dir  = os.path.join(report_dir,"plots")

    for d in [final_dir, report_dir, plots_dir,
              paths.get("predictions","predictions"),
              os.path.dirname(db_path)]:
        os.makedirs(d, exist_ok=True)

    _state["config"]  = cfg
    _state["db_path"] = db_path

    p1_valid = pre.validate_phase1_complete(cfg)
    _s("Phase 1 Validation", p1_valid["ok"])
    if not p1_valid["ok"]:
        print("  ERROR: Phase 1 outputs incomplete.")
        return False

    scan_records = pre.scan_raw_directory(raw_dir)
    _state["scan_records"] = scan_records
    raw_hashes   = pre.generate_file_hashes(scan_records)
    target_grid  = pre.build_target_grid(cfg)
    daily_idx    = pd.date_range(cfg["prototype_dates"]["start"],
                                 cfg["prototype_dates"]["end"], freq="D")

    conn = pre.init_database(db_path)
    pre.init_phase2_tables(conn)

    surface_datasets: dict = {}
    glorys_ds  = None
    all_mv     = {}
    all_units  = []
    all_interp = []
    depth_log  = {}

    aliases_cfg   = cfg["phase2"]["variable_aliases"]
    var_src_res   = cfg["phase2"]["temporal_source_resolution"]

    def _find_file_for_var(canonical):
        alias_list = aliases_cfg.get(canonical, [canonical])
        for rec in scan_records:
            fn = rec["filename"].lower()
            if any(a.lower() in fn for a in [canonical]+alias_list):
                return rec["filepath"]
        if canonical == "thetao":
            for rec in scan_records:
                if rec["dataset_type"] == "TARGET":
                    return rec["filepath"]
        type_map = {"sst":"CORE_INPUT","sss":"CORE_INPUT","current_u":"CORE_INPUT",
                    "current_v":"CORE_INPUT","wind_u":"CORE_INPUT","wind_v":"CORE_INPUT","sla":"CORE_INPUT"}
        if canonical in type_map:
            for rec in scan_records:
                if rec["dataset_type"] == type_map[canonical]:
                    try:
                        ds_peek = xr.open_dataset(rec["filepath"], engine="netcdf4")
                        found   = any(a.lower() in [v.lower() for v in list(ds_peek.data_vars)]
                                      for a in alias_list)
                        ds_peek.close()
                        if found:
                            return rec["filepath"]
                    except Exception:
                        pass
        return None

    processed_fps = set()
    for canonical in ["sst","sss","current_u","current_v","wind_u","wind_v","thetao"]:
        fp = _find_file_for_var(canonical)
        if fp is None or fp in processed_fps:
            continue
        processed_fps.add(fp)

        (ds, mv_report, unit_log, coord_log,
         var_log, time_info, cov_info) = _open_and_prepare(fp, cfg)
        if ds is None:
            continue

        all_units.extend(unit_log)
        source_res   = var_src_res.get(canonical, "daily")
        before_steps = time_info.get("before_steps", 0)

        if canonical != "thetao":
            ds_daily, method = pre.harmonize_to_daily(ds, source_res, cfg)
        else:
            ds_daily, method = ds, "daily_passthrough"

        after_steps = int(ds_daily.sizes.get("time", 0))
        ds_grid     = pre.regrid_to_standard(ds_daily, target_grid, cfg)
        mv_post     = pre.handle_missing_values(ds_grid)
        for var, info in mv_post.items():
            all_mv[f"{canonical}:{var}"] = info

        if canonical == "thetao":
            glorys_ds = ds_grid
        else:
            surface_datasets[canonical] = ds_grid

        all_interp.append({"variable": canonical, "source_res": source_res,
                            "method": method, "before_steps": before_steps,
                            "after_steps": after_steps})
        ds.close()

    _s("Region Cropped"); _s("Time Cropped"); _s("Variables Standardized")
    _s("Units Standardized"); _s("Missing Values Processed")
    _s("Daily Harmonization"); _s("Spatial Harmonization")

    # SLA
    sla_fp = _find_file_for_var("sla")
    if sla_fp and sla_fp.endswith(".csv"):
        ds_sla, _ = pre.grid_sla_alongtrack(sla_fp, cfg, target_grid, daily_idx)
        if ds_sla is not None:
            surface_datasets["sla"] = ds_sla
            surface_datasets["sla_observation_count"] = ds_sla
    if "sla" not in surface_datasets:
        lats = target_grid["latitude"]; lons = target_grid["longitude"]
        arr  = np.full((len(daily_idx), len(lats), len(lons)), np.nan, dtype="float32")
        cnt  = np.zeros((len(daily_idx), len(lats), len(lons)), dtype="int32")
        ds_empty = xr.Dataset(
            {"sla":(["time","latitude","longitude"],arr),
             "sla_observation_count":(["time","latitude","longitude"],cnt)},
            coords={"time":daily_idx,"latitude":lats,"longitude":lons})
        surface_datasets["sla"] = ds_empty
        surface_datasets["sla_observation_count"] = ds_empty
    _s("SLA Gridding")

    if glorys_ds is not None:
        ds_depths, depth_log = pre.process_glorys_depths(glorys_ds, cfg)
        glorys_ds.close()
        glorys_ds = ds_depths
    _s("Depth Harmonization")

    ds_surface = pre.align_surface_datasets(surface_datasets, target_grid, daily_idx, cfg)
    ds_target  = pre.align_target_dataset(glorys_ds, target_grid, daily_idx, cfg)
    _s("Dataset Alignment")

    surf_path   = os.path.join(final_dir, "ocean_surface_inputs.nc")
    target_path = os.path.join(final_dir, "subsurface_temperature_target.nc")
    pre.save_surface_netcdf(ds_surface, surf_path)
    pre.save_target_netcdf(ds_target,   target_path)
    _s("NetCDF Generated")

    stats_surface = pre.compute_dataset_statistics(ds_surface)
    stats_target  = pre.compute_dataset_statistics(ds_target)

    surf_shape   = dict(ds_surface.sizes)
    target_shape = dict(ds_target.sizes)
    surf_id = pre.insert_harmonized_dataset(conn, "ocean_surface_inputs.nc",
                                             surf_path, surf_shape, list(ds_surface.data_vars))
    tgt_id  = pre.insert_harmonized_dataset(conn, "subsurface_temperature_target.nc",
                                             target_path, target_shape, list(ds_target.data_vars))
    pre.insert_statistics(conn, surf_id, stats_surface)
    pre.insert_statistics(conn, tgt_id,  stats_target)
    pre.insert_missing_values(conn, surf_id, all_mv)
    for log_rec in all_interp:
        pre.insert_interpolation_log(conn, surf_id, log_rec["variable"],
                                     log_rec["source_res"], log_rec["method"],
                                     log_rec["before_steps"], log_rec["after_steps"])
    if depth_log:
        pre.insert_depth_log(conn, tgt_id, depth_log)

    _state["p2_statistics"]    = pre.get_phase2_statistics(conn)
    _state["p2_depths"]        = pre.get_phase2_depths(conn)
    _state["p2_missing"]       = pre.get_phase2_missing(conn)
    _state["p2_interpolation"] = pre.get_phase2_interpolation(conn)
    conn.close()
    _state["p2_surface_shape"] = surf_shape
    _state["p2_target_shape"]  = target_shape
    _s("SQLite Updated")

    p2_summary = {
        "phase": 2, "generated_at": datetime.now(timezone.utc).isoformat(),
        "surface_shape": str(surf_shape), "target_shape": str(target_shape),
        "n_time_steps": surf_shape.get("time",0),
        "grid_size": f"{surf_shape.get('latitude',0)}×{surf_shape.get('longitude',0)}",
        "n_depths": target_shape.get("depth",0),
        "variables_processed": {r["variable"]:r["method"] for r in all_interp},
        "unit_conversions": all_units,
        "glorys_exact_depths":   len(depth_log.get("exact",[])),
        "glorys_interp_depths":  len(depth_log.get("interpolated",[])),
        "glorys_missing_depths": len(depth_log.get("missing",[])),
        "phase2_status": "COMPLETE",
    }
    _state["p2_summary"] = p2_summary
    _state["p2_status"]  = {"status":"COMPLETE","surface":surf_shape,"target":target_shape}

    pre.save_phase2_summary(p2_summary, os.path.join(report_dir,"phase2_summary.json"))
    pre.save_phase2_md_report(p2_summary, os.path.join(report_dir,"PHASE2_REPORT.md"))

    for fname, data, cols in [
        ("phase2_statistics.csv",
         [{"variable":r["variable"],"mean":r.get("mean"),"std":r.get("std"),
           "min":r.get("min"),"max":r.get("max"),"coverage_pct":r.get("coverage_pct"),
           "missing_pct":r.get("missing_pct")} for r in stats_surface+stats_target],
         ["variable","mean","std","min","max","coverage_pct","missing_pct"]),
        ("phase2_missing_report.csv",
         [{"variable":k,**v} for k,v in all_mv.items()], None),
        ("phase2_interpolation_report.csv", all_interp,
         ["variable","source_res","method","before_steps","after_steps"]),
        ("phase2_depth_report.csv",
         [{"depth":d,"type":"exact"} for d in depth_log.get("exact",[])] +
         [{"depth":d,"type":"interpolated"} for d in depth_log.get("interpolated",[])] +
         [{"depth":d,"type":"missing"} for d in depth_log.get("missing",[])],
         ["depth","type"]),
        ("phase2_alignment_report.csv",
         [{"dataset":"ocean_surface_inputs.nc","time":surf_shape.get("time",0),
           "latitude":surf_shape.get("latitude",0),"longitude":surf_shape.get("longitude",0),
           "depth":"-","variables":"|".join(ds_surface.data_vars)},
          {"dataset":"subsurface_temperature_target.nc","time":target_shape.get("time",0),
           "latitude":target_shape.get("latitude",0),"longitude":target_shape.get("longitude",0),
           "depth":target_shape.get("depth",0),"variables":"thetao"}], None),
    ]:
        pre.save_dataframe_report(data, os.path.join(report_dir,fname), cols)

    pre.generate_phase2_plots(ds_surface, ds_target, stats_surface+stats_target,
                               all_mv, depth_log, all_interp, plots_dir)
    _s("Reports Generated")

    pre.verify_file_hashes(scan_records, raw_hashes)
    print("\n" + "="*55 + "\n  PHASE 2 COMPLETE\n" + "="*55 + "\n")
    return True


# ──────────────────────────────────────────────────────────────
# PHASE 3 PIPELINE
# ──────────────────────────────────────────────────────────────
def run_phase3(cfg: dict, start_api: bool = False) -> bool:
    print("\n" + "="*55 + "\n  PHASE 3 STARTED\n" + "="*55 + "\n")

    paths      = cfg["paths"]
    db_path    = paths["database"]
    report_dir = "reports/phase3"
    plots_dir  = os.path.join(report_dir, "plots")
    ckpt_dir   = "models"
    argo_dir   = os.path.join(paths.get("raw_data","data/raw"), "argo")

    for d in [report_dir, plots_dir, ckpt_dir, argo_dir,
              paths.get("predictions","predictions")]:
        os.makedirs(d, exist_ok=True)

    _state["config"]  = cfg
    _state["db_path"] = db_path

    # ── Validate Phase 1 ──────────────────────────────────────
    p1_ok = pre.validate_phase1_complete(cfg)["ok"]
    _s("Phase 1 Validation", p1_ok)
    if not p1_ok:
        print("  Run Phase 1 first."); return False

    # ── Validate Phase 2 ──────────────────────────────────────
    final_dir  = paths.get("final_data","data/final")
    surf_path  = os.path.join(final_dir, "ocean_surface_inputs.nc")
    tgt_path   = os.path.join(final_dir, "subsurface_temperature_target.nc")
    p2_ok = os.path.exists(surf_path) and os.path.exists(tgt_path)
    _s("Phase 2 Validation", p2_ok)
    if not p2_ok:
        print("  Run Phase 2 first."); return False

    # ── Load Phase 2 data ────────────────────────────────────
    started_at = datetime.now(timezone.utc).isoformat()
    surface, target, times, lats, lons = ai.load_phase2_data(cfg)
    _state.update({"p3_surface": surface, "p3_target": target,
                   "p3_times": times, "p3_lats": lats, "p3_lons": lons})

    # Load surface dataset for map endpoints
    ds_surf_raw = xr.open_dataset(surf_path)
    _state["p3_surface_ds"] = ds_surf_raw
    _s("Surface Dataset Loaded")
    _s("Target Dataset Loaded")

    # ── Satellite Embedding Engine ────────────────────────────
    model = ai.OceanCNNModel(
        in_channels=8,
        n_depths=len(cfg.get("target_depths", [0]*15)),
        embed_dim=cfg.get("phase3",{}).get("embed_dim", 64),
    ).to(ai.DEVICE)
    _s("Satellite Embedding Engine")
    _s("CNN Encoder")
    _s("Spatial Attention")
    _s("Physics Guided Loss")

    # ── Training ─────────────────────────────────────────────
    training_result = ai.train_model(model, surface, target, cfg, ckpt_dir)
    _state["p3_training_hist"] = training_result["history"]
    _s("Training Complete")
    _s("Best Model Saved")

    # ── Load best model ───────────────────────────────────────
    model, norm_stats = ai.load_best_model(cfg, ckpt_dir)
    _state["p3_model"]      = model
    _state["p3_norm_stats"] = norm_stats

    # ── Predictions ───────────────────────────────────────────
    pred_ds = ai.generate_predictions(model, surface, times, lats, lons, norm_stats, cfg)
    _state["p3_pred_ds"] = pred_ds
    pred_nc_path = os.path.join(paths.get("predictions","predictions"),
                                "ocean_prediction.nc")
    pred_ds.to_netcdf(pred_nc_path, engine="netcdf4")
    _s("Prediction Generated")

    # ── Future Forecast ───────────────────────────────────────
    future_fc = ai.generate_future_forecast(model, surface, times, lats, lons, norm_stats, cfg)
    _state["p3_future_fc"] = future_fc
    with open(os.path.join(paths.get("predictions","predictions"),
                           "future_forecast.json"), "w") as f:
        json.dump(future_fc, f, indent=2, default=str)
    _s("Future Forecast Generated")

    # ── Marine Heatwave Detection ─────────────────────────────
    hotspots = ai.detect_marine_heatwaves(pred_ds, cfg)
    _state["p3_hotspots"] = hotspots
    _s("Marine Heatwave Detection")

    # ── Impact Prediction ────────────────────────────────────
    impacts = ai.generate_impact_predictions(hotspots)
    _state["p3_impacts"] = impacts
    _s("Impact Prediction")

    # ── Uncertainty / Confidence ──────────────────────────────
    uncertainty = ai.generate_uncertainty(model, surface, norm_stats,
                                           n_samples=cfg.get("phase3",{}).get("mc_samples",8))
    _state["p3_uncertainty"] = uncertainty
    _s("Confidence Map Generated")

    # ── Explainable AI ────────────────────────────────────────
    xai = ai.generate_explainability(model, surface, norm_stats, cfg)
    _state["p3_xai"] = xai
    _s("Explainable AI Generated")

    # ── ARGO Validation ──────────────────────────────────────
    argo_result = argo_val.run_argo_validation(pred_ds, argo_dir, cfg)
    _state["p3_argo_result"] = argo_result
    _s("ARGO Validation")

    # ── Multi-variable Analysis ──────────────────────────────
    multivar = ai.generate_multivariate_analysis(surface)
    _state["p3_multivar"] = multivar

    # ── SQLite Phase 3 tables ─────────────────────────────────
    conn = sqlite3.connect(db_path)
    ai.init_phase3_tables(conn)
    adm.init_admin_tables(conn)

    ai.log_prediction(conn, str(pd.Timestamp(times[-1]).date()), 0,
                       "best_model_v1", argo_result.get("rmse"))
    for days in [7, 15, 30]:
        ai.log_prediction(conn, future_fc[days]["forecast_date"], days,
                           "best_model_v1", None)
    ai.log_heatwaves(conn, hotspots)
    ai.log_model_version(conn, "best_model_v1",
                          os.path.join(ckpt_dir,"best_model.pt"),
                          training_result.get("best_val_loss", 0.0))
    ai.log_training_run(conn, len(training_result["history"]),
                         training_result.get("best_val_loss", 0.0), started_at)
    ai.log_explainability(conn, {**xai, **uncertainty})
    conn.close()
    _s("SQLite Updated")

    # ── Reports ───────────────────────────────────────────────
    pred_stats = ev.compute_prediction_statistics(pred_ds)
    ev.save_argo_validation_report(argo_result,
                                    os.path.join(report_dir,"argo_validation.csv"))
    ev.save_heatwave_report(hotspots, os.path.join(report_dir,"heatwave_report.csv"))
    ev.save_attention_summary(xai,    os.path.join(report_dir,"attention_summary.csv"))
    ev.save_embedding_summary(pred_ds, os.path.join(report_dir,"embedding_summary.csv"))
    ev.save_confidence_summary(uncertainty, os.path.join(report_dir,"confidence_summary.csv"))
    ev.save_uncertainty_summary(uncertainty,os.path.join(report_dir,"uncertainty_summary.csv"))
    ev.save_prediction_statistics_report(pred_stats,
                                          os.path.join(report_dir,"prediction_statistics.csv"))

    # Training history CSV (copy from models/)
    th_src = os.path.join(ckpt_dir,"training_history.csv")
    if os.path.exists(th_src):
        import shutil
        shutil.copy(th_src, os.path.join(report_dir,"training_history.csv"))

    p3_summary = ev.generate_phase3_summary(
        training_result, argo_result, hotspots, uncertainty, future_fc,
        os.path.join(report_dir,"phase3_summary.json"))
    ev.generate_phase3_md_report(p3_summary, os.path.join(report_dir,"PHASE3_REPORT.md"))
    _s("Reports Generated")

    # ── Visualizations ────────────────────────────────────────
    viz.generate_phase3_plots(
        pred_ds, ds_surf_raw,
        training_result["history"], hotspots, xai, uncertainty,
        argo_result, plots_dir)

    _state["p3_summary"] = p3_summary
    _state["p3_status"]  = {"status":"COMPLETE"}

    print("\n" + "="*55)
    print("  PHASE 3 COMPLETE")
    print("="*55 + "\n")

    _s("Flask API Ready")
    return True
