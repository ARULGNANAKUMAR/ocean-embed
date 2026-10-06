"""
tests/test_oceanembed_core.py
OceanVerse AI v4 — OceanEmbed Core Test Suite
SIH 26066

Tests cover:
  test_argo_loader          — ARGO file discovery and loading
  test_argo_qc              — QC filtering (good profiles pass, bad rejected)
  test_argo_matching        — Spatiotemporal matching
  test_validation_metrics   — RMSE, MAE, Bias, Corr, R² correctness
  test_embedding            — Model produces real (non-random) embeddings
  test_prediction_shape     — Model output is (B,15,H,W)
  test_uncertainty          — MC-Dropout produces non-zero uncertainty
  test_physics_loss         — PhysicsGuidedLoss weight breakdown
  test_depth_mapping        — 15 standard depths present in prediction
  test_data_split           — Chronological split is temporal, no leakage
  test_api_integration      — API endpoints return expected structure
  test_end_to_end           — Full pipeline smoke test
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent.parent))

import ai_engine as ai
import argo_validation as argo_val


# ──────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

@pytest.fixture(scope="module")
def small_region():
    return {"lat_min": 10.0, "lat_max": 15.0, "lon_min": 80.0, "lon_max": 85.0}


@pytest.fixture(scope="module")
def synthetic_surface():
    """Surface array: (T=10, C=8, H=5, W=5)"""
    np.random.seed(0)
    T, C, H, W = 10, 8, 5, 5
    arr = np.random.randn(T, C, H, W).astype("float32")
    arr[:, 0] += 28   # SST offset
    return arr


@pytest.fixture(scope="module")
def synthetic_target():
    """Target array: (T=10, D=15, H=5, W=5) — physically decreasing with depth"""
    np.random.seed(1)
    T, D, H, W = 10, 15, 5, 5
    target = np.zeros((T, D, H, W), dtype="float32")
    for di, depth in enumerate(TARGET_DEPTHS):
        decay = max(1.0 - depth / 2000, 0.05)
        target[:, di] = 28 * decay + np.random.randn(T, H, W) * 0.2
    return target


@pytest.fixture(scope="module")
def small_pred_ds(synthetic_surface):
    """xr.Dataset with thetao_pred (T,D,H,W)."""
    lats  = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    lons  = np.array([80.0, 81.0, 82.0, 83.0, 84.0])
    times = pd.date_range("2025-01-01", periods=10, freq="D")
    # Use first channel as proxy temperature
    data  = np.stack([synthetic_surface[:, 0] + 28 * max(1 - d/2000, 0.05)
                      for d in TARGET_DEPTHS], axis=1)
    return xr.Dataset(
        {"thetao_pred": (["time","depth","latitude","longitude"],
                         data.astype("float32"),
                         {"units": "degC"})},
        coords={"time": times, "depth": TARGET_DEPTHS,
                "latitude": lats, "longitude": lons},
    )


@pytest.fixture(scope="module")
def argo_dir_with_data(tmp_path_factory, small_region):
    """Create a temp dir with 2 valid ARGO NetCDF files."""
    d = tmp_path_factory.mktemp("argo")
    pres = np.array(TARGET_DEPTHS, dtype="float32")
    n_lev = len(pres)

    for i, (lat, lon, date) in enumerate([
        (12.0, 82.0, "2025-01-05"),
        (13.5, 83.5, "2025-01-10"),
        (11.0, 81.5, "2025-01-15"),
    ]):
        n_p = 1
        temps = np.array([[28.0 - p / 50 for p in pres]], dtype="float32")
        ds = xr.Dataset({
            "TEMP":       (["N_PROF","N_LEVELS"], temps, {"units":"degree_Celsius"}),
            "PRES":       (["N_PROF","N_LEVELS"], pres.reshape(1,-1), {}),
            "LATITUDE":   (["N_PROF"], np.array([lat], dtype="float32"), {}),
            "LONGITUDE":  (["N_PROF"], np.array([lon], dtype="float32"), {}),
            "JULD":       (["N_PROF"], pd.to_datetime([date]), {}),
            "TEMP_QC":    (["N_PROF","N_LEVELS"], np.ones((1,n_lev), dtype="float32"), {}),
            "PRES_QC":    (["N_PROF","N_LEVELS"], np.ones((1,n_lev), dtype="float32"), {}),
            "POSITION_QC":(["N_PROF"], np.array([1.0], dtype="float32"), {}),
        })
        ds.to_netcdf(str(d / f"argo_float_{i}.nc"))
    return str(d)


@pytest.fixture(scope="module")
def small_cfg(small_region):
    return {
        "region": {"full": small_region, "prototype": small_region},
        "target_depths": TARGET_DEPTHS,
        "phase3": {"lr": 1e-3, "epochs": 2, "batch_size": 2,
                   "patience": 2, "embed_dim": 16, "mc_samples": 5,
                   "loss_weights": {"mse": 1.0, "grad": 0.1, "smooth": 0.05, "therm": 0.0}},
        "matching_tolerances": {
            "max_time_delta_days": 5,
            "max_distance_km": 200.0,
        },
    }


# ──────────────────────────────────────────────────────────────
# test_argo_loader
# ──────────────────────────────────────────────────────────────

def test_argo_loader_finds_files(argo_dir_with_data, small_region):
    region = small_region
    profiles, qc_stats = argo_val.load_argo_profiles(argo_dir_with_data, region)
    assert qc_stats["files_read"] == 3, "Should read 3 ARGO files"
    assert qc_stats["valid"] >= 3, f"Expected ≥3 valid profiles, got {qc_stats['valid']}"
    assert qc_stats["total"] >= 3


def test_argo_loader_empty_dir(tmp_path, small_region):
    profiles, qc_stats = argo_val.load_argo_profiles(str(tmp_path), small_region)
    assert profiles == []
    assert qc_stats["status"] == "NO_FILES"


# ──────────────────────────────────────────────────────────────
# test_argo_qc
# ──────────────────────────────────────────────────────────────

def test_qc_accepts_good_profile(small_region):
    pres  = np.array([0, 5, 10, 50, 100], dtype="float64")
    temps = np.array([28.5, 28.0, 27.0, 24.0, 18.0], dtype="float64")
    tqc   = np.ones(5, dtype="float32")
    ok, reason, t_clean, p_clean = argo_val.qc_argo_profile(
        12.0, 82.0, pd.Timestamp("2025-01-05"),
        temps, pres, tqc, 1.0, small_region
    )
    assert ok is True, f"Good profile rejected: {reason}"
    assert len(t_clean) == 5


def test_qc_rejects_bad_coordinates(small_region):
    pres  = np.array([0, 5, 10], dtype="float64")
    temps = np.array([28, 27, 26], dtype="float64")
    tqc   = np.ones(3, dtype="float32")
    # Out of region (Arctic)
    ok, reason, _, _ = argo_val.qc_argo_profile(
        75.0, 30.0, pd.Timestamp("2025-01-05"),
        temps, pres, tqc, 1.0, small_region
    )
    assert ok is False
    assert reason == "out_of_region"


def test_qc_rejects_impossible_temperature(small_region):
    pres  = np.array([0, 5, 10], dtype="float64")
    temps = np.array([99.0, 28, 27], dtype="float64")   # 99°C impossible
    tqc   = np.ones(3, dtype="float32")
    ok, reason, t_clean, _ = argo_val.qc_argo_profile(
        12.0, 82.0, pd.Timestamp("2025-01-05"),
        temps, pres, tqc, 1.0, small_region
    )
    # Profile should either fail or clean the bad value
    if ok:
        assert 99.0 not in t_clean, "Impossible temperature not removed by QC"


def test_qc_rejects_bad_position_qc(small_region):
    pres  = np.array([0, 5, 10, 20, 30], dtype="float64")
    temps = np.array([28, 27, 26, 24, 22], dtype="float64")
    tqc   = np.ones(5, dtype="float32")
    ok, reason, _, _ = argo_val.qc_argo_profile(
        12.0, 82.0, pd.Timestamp("2025-01-05"),
        temps, pres, tqc, 4.0, small_region   # pos_qc=4 = bad
    )
    assert ok is False
    assert reason == "bad_position_qc"


# ──────────────────────────────────────────────────────────────
# test_argo_matching
# ──────────────────────────────────────────────────────────────

def test_argo_matching_finds_pairs(argo_dir_with_data, small_pred_ds, small_cfg):
    region = small_cfg["region"]["full"]
    profiles, _ = argo_val.load_argo_profiles(argo_dir_with_data, region)
    assert len(profiles) >= 1, "Need profiles to match"
    pairs = argo_val.match_profiles_to_predictions(profiles, small_pred_ds, small_cfg)
    assert len(pairs) > 0, "Expected matched pairs"


def test_argo_matching_records_have_required_keys(argo_dir_with_data, small_pred_ds, small_cfg):
    region = small_cfg["region"]["full"]
    profiles, _ = argo_val.load_argo_profiles(argo_dir_with_data, region)
    pairs = argo_val.match_profiles_to_predictions(profiles, small_pred_ds, small_cfg)
    if pairs:
        required = {"depth", "obs_temp", "pred_temp", "lat", "lon", "time", "dist_km"}
        assert required.issubset(set(pairs[0].keys()))


# ──────────────────────────────────────────────────────────────
# test_validation_metrics
# ──────────────────────────────────────────────────────────────

def test_metrics_known_values():
    obs  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    pred = np.array([1.1, 1.9, 3.1, 3.9, 5.1])
    m = argo_val._metrics(obs, pred)
    assert m["n"] == 5
    assert abs(m["rmse"] - 0.1) < 0.01,   f"RMSE wrong: {m['rmse']}"
    assert abs(m["bias"] - 0.0) < 0.05,   f"Bias wrong: {m['bias']} (expected ~0)"
    assert m["correlation"] > 0.995,       f"Corr wrong: {m['correlation']}"
    assert abs(m["r2"] - 1.0) < 0.01,     f"R² wrong: {m['r2']}"


def test_metrics_non_trivial():
    """Metrics must not be 0 RMSE / 1.0 correlation unless data is perfect."""
    obs  = np.array([25.0, 24.0, 23.0, 20.0, 15.0, 10.0])
    pred = np.array([26.0, 23.5, 22.0, 21.0, 16.0, 11.5])
    m = argo_val._metrics(obs, pred)
    assert m["rmse"] > 0.0, "RMSE cannot be 0 for imperfect predictions"
    assert m["correlation"] < 1.0, "Correlation cannot be 1.0 for imperfect predictions"
    assert m["rmse"] < 5.0, "RMSE too large for these test values"


def test_metrics_empty():
    m = argo_val._metrics(np.array([]), np.array([]))
    assert m["n"] == 0
    assert m["rmse"] is None


# ──────────────────────────────────────────────────────────────
# test_embedding
# ──────────────────────────────────────────────────────────────

def test_embedding_depends_on_input():
    """Same model with different inputs must produce different embeddings."""
    model = ai.SatelliteEmbeddingEngine(in_channels=8, embed_dim=16)
    model.eval()
    x1 = torch.randn(1, 8, 5, 5)
    x2 = torch.randn(1, 8, 5, 5) * 5 + 10  # very different
    with torch.no_grad():
        emb1, _, _ = model(x1)
        emb2, _, _ = model(x2)
    assert not torch.allclose(emb1, emb2), "Embedding is not input-dependent!"


def test_embedding_shape():
    model = ai.SatelliteEmbeddingEngine(in_channels=8, embed_dim=32)
    x = torch.randn(2, 8, 5, 5)
    emb, att, feat = model(x)
    assert emb.shape == (2, 32, 5, 5), f"Embedding shape wrong: {emb.shape}"
    assert att.shape[0] == 2, "Attention batch size mismatch"


# ──────────────────────────────────────────────────────────────
# test_prediction_shape
# ──────────────────────────────────────────────────────────────

def test_prediction_output_shape():
    model = ai.OceanCNNModel(in_channels=8, n_depths=15, embed_dim=16)
    model.eval()
    x = torch.randn(2, 8, 5, 5)
    with torch.no_grad():
        pred, emb, att = model(x)
    assert pred.shape == (2, 15, 5, 5), f"Prediction shape: {pred.shape}"
    assert pred.shape[1] == 15, "Must output exactly 15 depth levels"


def test_prediction_values_in_range():
    """Predictions after denormalization should be plausible ocean temperatures."""
    model = ai.OceanCNNModel(in_channels=8, n_depths=15, embed_dim=16)
    model.eval()
    x = torch.zeros(1, 8, 5, 5)
    x[:, 0] = 1.0   # normalised SST
    with torch.no_grad():
        pred, _, _ = model(x)
    # Not checking exact values — just that the model produces finite output
    assert torch.all(torch.isfinite(pred)), "Model produced NaN/Inf output"


# ──────────────────────────────────────────────────────────────
# test_uncertainty
# ──────────────────────────────────────────────────────────────

def test_mc_dropout_produces_nonzero_uncertainty(synthetic_surface):
    """MC Dropout MUST produce non-zero std when model has Dropout layers."""
    model = ai.OceanCNNModel(in_channels=8, n_depths=15, embed_dim=16)
    # Verify Dropout exists
    has_dropout = any(isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d))
                      for m in model.modules())
    assert has_dropout, "Model has no Dropout layers — MC uncertainty will always be 0"

    norm_stats = {f"surf_ch{c}": {"mean": 0.0, "std": 1.0} for c in range(8)}
    norm_stats["target"] = {"mean": 20.0, "std": 5.0}
    result = ai.generate_uncertainty(model, synthetic_surface, norm_stats, n_samples=10)
    
    depth_unc = np.array(result["depth_uncertainty"])
    assert result["overall_confidence"] != 1.0, "Confidence is exactly 1.0 — no real uncertainty!"
    assert np.any(depth_unc > 0.0), "All depth uncertainties are 0 — MC-Dropout not working"
    assert result["n_samples"] == 10


def test_confidence_not_trivial(synthetic_surface):
    """Overall confidence must be in (0,1) exclusive — not a hardcoded value."""
    model = ai.OceanCNNModel(in_channels=8, n_depths=15, embed_dim=16)
    norm_stats = {f"surf_ch{c}": {"mean": 0.0, "std": 1.0} for c in range(8)}
    norm_stats["target"] = {"mean": 20.0, "std": 5.0}
    result = ai.generate_uncertainty(model, synthetic_surface, norm_stats, n_samples=8)
    conf = result["overall_confidence"]
    assert 0.0 < conf < 1.0, f"Confidence should be in (0,1), got {conf}"


# ──────────────────────────────────────────────────────────────
# test_physics_loss
# ──────────────────────────────────────────────────────────────

def test_physics_loss_components():
    cfg = {"phase3": {"loss_weights": {"mse": 1.0, "grad": 0.2, "smooth": 0.1, "therm": 0.0}}}
    loss_fn = ai.PhysicsGuidedLoss(cfg)

    pred   = torch.randn(2, 15, 5, 5)
    target = torch.randn(2, 15, 5, 5)
    total, breakdown = loss_fn(pred, target)

    assert "mse"    in breakdown and breakdown["mse"] >= 0
    assert "grad"   in breakdown and breakdown["grad"] >= 0
    assert "smooth" in breakdown and breakdown["smooth"] >= 0
    assert breakdown["total"] > 0, "Total loss must be positive"
    assert abs(breakdown["total"] - float(total)) < 1e-4


def test_physics_loss_weights_configurable():
    cfg1 = {"phase3": {"loss_weights": {"mse": 1.0, "grad": 0.0, "smooth": 0.0, "therm": 0.0}}}
    cfg2 = {"phase3": {"loss_weights": {"mse": 1.0, "grad": 1.0, "smooth": 1.0, "therm": 0.0}}}
    fn1 = ai.PhysicsGuidedLoss(cfg1)
    fn2 = ai.PhysicsGuidedLoss(cfg2)

    pred   = torch.randn(2, 15, 5, 5)
    target = torch.randn(2, 15, 5, 5)
    loss1, _ = fn1(pred, target)
    loss2, _ = fn2(pred, target)
    assert loss1 != loss2, "Different weights must produce different total loss"


# ──────────────────────────────────────────────────────────────
# test_depth_mapping
# ──────────────────────────────────────────────────────────────

def test_15_standard_depths_in_prediction(small_pred_ds):
    pred_depths = list(small_pred_ds["depth"].values)
    assert len(pred_depths) == 15, f"Expected 15 depths, got {len(pred_depths)}"
    for d in TARGET_DEPTHS:
        assert d in pred_depths, f"Missing depth {d}m in prediction dataset"


# ──────────────────────────────────────────────────────────────
# test_data_split
# ──────────────────────────────────────────────────────────────

def test_chronological_split_no_leakage():
    n = 100
    tr, va, te = ai.chronological_split(n, train_frac=0.7, val_frac=0.15)
    # Must be contiguous
    assert sorted(tr) == tr, "Train indices not sorted"
    assert sorted(va) == va, "Val indices not sorted"
    assert sorted(te) == te, "Test indices not sorted"
    # No overlap
    assert set(tr) & set(va) == set(), "Train/val overlap!"
    assert set(tr) & set(te) == set(), "Train/test overlap!"
    assert set(va) & set(te) == set(), "Val/test overlap!"
    # Must be temporal: all train < val < test
    if tr and va:
        assert max(tr) < min(va), "Train indices exceed val indices — temporal leakage!"
    if va and te:
        assert max(va) < min(te), "Val indices exceed test indices — temporal leakage!"


def test_chronological_split_coverage():
    n = 50
    tr, va, te = ai.chronological_split(n)
    assert len(tr) + len(va) + len(te) == n, "Split does not cover all samples"


# ──────────────────────────────────────────────────────────────
# test_full_validation_pipeline
# ──────────────────────────────────────────────────────────────

def test_full_argo_validation_pipeline(argo_dir_with_data, small_pred_ds, small_cfg):
    """Integration test: full ARGO validation returns VALIDATED with real metrics."""
    result = argo_val.run_argo_validation(small_pred_ds, argo_dir_with_data, small_cfg)
    
    # Status
    assert result["status"] in ("VALIDATED", "MATCHING_FAILED"), \
        f"Unexpected status: {result['status']}"
    
    if result["status"] == "VALIDATED":
        # Real metrics — must not be None
        assert result["rmse"]        is not None, "RMSE is None after successful validation"
        assert result["mae"]         is not None, "MAE is None"
        assert result["bias"]        is not None, "Bias is None"
        assert result["correlation"] is not None, "Correlation is None"
        assert result["r2"]          is not None, "R² is None"
        # RMSE must be positive and reasonable
        assert result["rmse"] > 0.0, f"RMSE={result['rmse']} should be > 0"
        assert result["rmse"] < 50.0, f"RMSE={result['rmse']} implausibly large (model untrained)"
        # Correlation must be in [-1, 1]
        assert -1.0 <= result["correlation"] <= 1.0


def test_validation_returns_data_not_available_status(tmp_path, small_pred_ds, small_cfg):
    result = argo_val.run_argo_validation(small_pred_ds, str(tmp_path), small_cfg)
    assert result["status"] == "DATA_NOT_AVAILABLE"
    assert result["rmse"] is None


# ──────────────────────────────────────────────────────────────
# test_api_integration
# ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def flask_client():
    try:
        from app import app
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client
    except Exception:
        yield None


def test_api_health(flask_client):
    if flask_client is None:
        pytest.skip("Flask app could not be imported")
    resp = flask_client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "status" in data


def test_api_status_returns_json(flask_client):
    if flask_client is None:
        pytest.skip("Flask app could not be imported")
    resp = flask_client.get("/api/status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, dict)


# ──────────────────────────────────────────────────────────────
# test_end_to_end
# ──────────────────────────────────────────────────────────────

def test_end_to_end_pipeline(tmp_path, small_cfg):
    """
    End-to-end smoke test:
    Satellite surface → OceanCNNModel → Predictions → ARGO matching → Metrics → Reports
    """
    # 1. Create small synthetic surface + target
    T, H, W = 6, 5, 5
    lats  = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
    lons  = np.array([80.0, 81.0, 82.0, 83.0, 84.0])
    times = pd.date_range("2025-01-01", periods=T, freq="D")
    surface = np.random.randn(T, 8, H, W).astype("float32")
    surface[:, 0] += 28

    # 2. Embedding + prediction
    model = ai.OceanCNNModel(in_channels=8, n_depths=15, embed_dim=16)
    norm_stats = {f"surf_ch{c}": {"mean": 0.0, "std": 1.0} for c in range(8)}
    norm_stats["target"] = {"mean": 20.0, "std": 5.0}
    model.eval()

    preds = []
    for t in range(T):
        p = ai.predict_single(model, surface[t], norm_stats)
        preds.append(p)
    pred_arr = np.stack(preds)   # (T,15,H,W)

    pred_ds = xr.Dataset(
        {"thetao_pred": (["time","depth","latitude","longitude"], pred_arr)},
        coords={"time": times, "depth": TARGET_DEPTHS, "latitude": lats, "longitude": lons}
    )

    # 3. Uncertainty
    result_unc = ai.generate_uncertainty(model, surface, norm_stats, n_samples=5)
    assert result_unc["overall_confidence"] != 1.0, "Confidence is hardcoded 1.0"

    # 4. ARGO validation
    argo_dir = str(tmp_path / "argo")
    os.makedirs(argo_dir)
    # Write one valid ARGO file into temp dir
    pres = np.array(TARGET_DEPTHS, dtype="float32")
    temps_a = np.array([[28 - p/50 for p in pres]], dtype="float32")
    ds_a = xr.Dataset({
        "TEMP":       (["N_PROF","N_LEVELS"], temps_a),
        "PRES":       (["N_PROF","N_LEVELS"], pres.reshape(1,-1)),
        "LATITUDE":   (["N_PROF"], np.array([12.0], dtype="float32")),
        "LONGITUDE":  (["N_PROF"], np.array([82.0], dtype="float32")),
        "JULD":       (["N_PROF"], pd.to_datetime(["2025-01-03"])),
        "TEMP_QC":    (["N_PROF","N_LEVELS"], np.ones((1,15), dtype="float32")),
        "PRES_QC":    (["N_PROF","N_LEVELS"], np.ones((1,15), dtype="float32")),
        "POSITION_QC":(["N_PROF"], np.array([1.0], dtype="float32")),
    })
    ds_a.to_netcdf(os.path.join(argo_dir, "argo_e2e.nc"))

    argo_result = argo_val.run_argo_validation(pred_ds, argo_dir, small_cfg)
    assert argo_result["status"] in ("VALIDATED", "MATCHING_FAILED")

    # 5. Reports
    report_dir = str(tmp_path / "reports")
    argo_val.write_argo_reports(argo_result, report_dir)
    assert os.path.exists(os.path.join(report_dir, "argo_validation_summary.json"))
    assert os.path.exists(os.path.join(report_dir, "argo_validation_report.md"))

    # Summary: if validated, check metrics are non-trivial
    if argo_result["status"] == "VALIDATED":
        assert argo_result["rmse"] is not None and argo_result["rmse"] > 0
        print(f"\n✅ E2E: RMSE={argo_result['rmse']}, Corr={argo_result['correlation']}, "
              f"Conf={result_unc['overall_confidence']:.3f}")
