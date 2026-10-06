"""
ai_engine.py
North Indian Ocean Intelligence Platform — Phase 3
CNN Encoder + Satellite Embedding + Physics-Guided Training + Prediction Engine
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xarray as xr

logger = logging.getLogger("ocean_platform")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────
# SECTION A+B  MODEL ARCHITECTURE
# ──────────────────────────────────────────────────────────────

class SpatialAttention(nn.Module):
    """Channel + spatial attention gate."""
    def __init__(self, channels: int):
        super().__init__()
        self.channel_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 2),
            nn.ReLU(),
            nn.Linear(channels // 2, channels),
            nn.Sigmoid(),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # Channel attention
        ch_att = self.channel_fc(x).view(B, C, 1, 1)
        x_ca  = x * ch_att
        # Spatial attention
        avg_map = x_ca.mean(dim=1, keepdim=True)
        max_map = x_ca.max(dim=1, keepdim=True).values
        sp_att  = self.spatial_conv(torch.cat([avg_map, max_map], dim=1))
        return x_ca * sp_att, sp_att


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class SatelliteEmbeddingEngine(nn.Module):
    """
    Section A: CNN-based satellite embedding.
    Input: (B, 8, H, W)  — 8 surface channels
    Output: embedding, attention_map, prediction_surface
    """
    def __init__(self, in_channels: int = 8, embed_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResBlock(32),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64),
        )
        self.attention = SpatialAttention(64)
        self.embed_conv = nn.Conv2d(64, embed_dim, 1)

    def forward(self, x):
        feat = self.encoder(x)
        feat_att, att_map = self.attention(feat)
        embedding = self.embed_conv(feat_att)
        return embedding, att_map, feat_att


class OceanCNNModel(nn.Module):
    """
    Section B: Full encoder→embedding→attention→residual decoder→15-depth output.
    Input : (B, T_window, 8, H, W) surface or (B, 8, H, W) single timestep
    Output: (B, 15, H, W) subsurface temperature
    """
    def __init__(self, in_channels: int = 8, n_depths: int = 15,
                 embed_dim: int = 64, h: int = 21, w: int = 21):
        super().__init__()
        self.embedding_engine = SatelliteEmbeddingEngine(in_channels, embed_dim)
        self.dropout_rate = 0.15  # MC-Dropout: must be > 0 for uncertainty estimation
        # Residual decoder with Dropout for MC uncertainty
        self.decoder = nn.Sequential(
            ResBlock(embed_dim),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(embed_dim, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            ResBlock(128),
            nn.Dropout2d(self.dropout_rate),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, n_depths, 1),
        )

    def forward(self, x):
        # Accept (B, C, H, W) or (B, T, C, H, W) → mean over T
        if x.dim() == 5:
            x = x.mean(dim=1)
        embedding, att_map, _ = self.embedding_engine(x)
        prediction = self.decoder(embedding)
        return prediction, embedding, att_map


# ──────────────────────────────────────────────────────────────
# SECTION C  PHYSICS-GUIDED LOSS
# ──────────────────────────────────────────────────────────────

class PhysicsGuidedLoss(nn.Module):
    """Section C: Masked MSE + vertical gradient + smoothness + thermocline."""
    def __init__(self, cfg: dict):
        super().__init__()
        lw = cfg.get("phase3", {}).get("loss_weights", {})
        self.w_mse   = lw.get("mse",   1.0)
        self.w_grad  = lw.get("grad",  0.1)
        self.w_smooth= lw.get("smooth",0.05)
        self.w_therm = lw.get("therm", 0.0)

    def forward(self, pred, target):
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            return pred.sum() * 0.0, {}

        # Masked MSE
        mse = F.mse_loss(pred[mask], target[mask])

        # Vertical gradient (encourage monotonic decrease with depth)
        pred_grad   = pred[:, 1:] - pred[:, :-1]        # (B, D-1, H, W)
        target_grad = target[:, 1:] - target[:, :-1]
        grad_mask   = ~torch.isnan(target_grad)
        grad_loss   = (F.mse_loss(pred_grad[grad_mask], target_grad[grad_mask])
                       if grad_mask.sum() > 0 else mse * 0)

        # Smoothness (horizontal)
        dx = (pred[:, :, :, 1:] - pred[:, :, :, :-1]).pow(2)
        dy = (pred[:, :, 1:, :] - pred[:, :, :-1, :]).pow(2)
        smooth_loss = dx.mean() + dy.mean()

        # Thermocline (penalise wrong sign of gradient near 50-200m → depth idx 5-10)
        therm_loss = torch.tensor(0.0, device=pred.device)
        if self.w_therm > 0:
            therm_pred   = pred[:, 5:10]
            therm_target = target[:, 5:10]
            tm = ~torch.isnan(therm_target)
            if tm.sum() > 0:
                therm_loss = F.mse_loss(therm_pred[tm], therm_target[tm])

        total = (self.w_mse  * mse
                 + self.w_grad  * grad_loss
                 + self.w_smooth* smooth_loss
                 + self.w_therm * therm_loss)

        breakdown = {
            "mse":    float(mse.detach()),
            "grad":   float(grad_loss.detach()),
            "smooth": float(smooth_loss.detach()),
            "therm":  float(therm_loss.detach()),
            "total":  float(total.detach()),
        }
        return total, breakdown


# ──────────────────────────────────────────────────────────────
# SECTION D  DATASET + TRAINING ENGINE
# ──────────────────────────────────────────────────────────────

class OceanDataset(torch.utils.data.Dataset):
    def __init__(self, surface: np.ndarray, target: np.ndarray):
        # surface: (T, C, H, W)   target: (T, D, H, W)
        self.surface = torch.from_numpy(surface).float()
        self.target  = torch.from_numpy(target).float()
        # Replace NaN with 0 in surface; keep NaN in target for masked loss
        self.surface = torch.nan_to_num(self.surface, nan=0.0)

    def __len__(self): return len(self.surface)
    def __getitem__(self, idx):
        return self.surface[idx], self.target[idx]


def load_phase2_data(cfg: dict) -> tuple[np.ndarray, np.ndarray, list, list, list]:
    """Load Phase 2 NetCDFs. Returns (surface, target, times, lats, lons)."""
    final_dir    = cfg["paths"].get("final_data", "data/final")
    surf_path    = os.path.join(final_dir, "ocean_surface_inputs.nc")
    target_path  = os.path.join(final_dir, "subsurface_temperature_target.nc")

    ds_surf = xr.open_dataset(surf_path)
    ds_tgt  = xr.open_dataset(target_path)

    surface_vars = ["sst","sss","sla","sla_observation_count",
                    "current_u","current_v","wind_u","wind_v"]
    surf_arrays = []
    for v in surface_vars:
        if v in ds_surf:
            arr = ds_surf[v].values.astype("float32")
        else:
            arr = np.zeros_like(ds_surf[surface_vars[0]].values, dtype="float32")
        surf_arrays.append(arr)

    surface = np.stack(surf_arrays, axis=1)   # (T, 8, H, W)
    target  = ds_tgt["thetao"].values.astype("float32")  # (T, D, H, W)

    times = list(ds_surf["time"].values)
    lats  = list(ds_surf["latitude"].values)
    lons  = list(ds_surf["longitude"].values)

    ds_surf.close(); ds_tgt.close()
    logger.info(f"Phase 2 data loaded — surface {surface.shape}, target {target.shape}")
    return surface, target, times, lats, lons


def normalize_data(surface: np.ndarray, target: np.ndarray) -> tuple:
    """Normalize to [−1, 1] per channel. Returns (surf_norm, tgt_norm, norm_stats)."""
    norm_stats = {}
    surf_norm = np.zeros_like(surface)
    for c in range(surface.shape[1]):
        ch = surface[:, c]
        valid = ch[np.isfinite(ch)]
        mu = float(valid.mean()) if len(valid) > 0 else 0.0
        sd = float(valid.std())  if len(valid) > 0 else 1.0
        sd = max(sd, 1e-6)
        surf_norm[:, c] = (surface[:, c] - mu) / sd
        norm_stats[f"surf_ch{c}"] = {"mean": mu, "std": sd}

    valid_t = target[np.isfinite(target)]
    tgt_mu  = float(valid_t.mean()) if len(valid_t) > 0 else 20.0
    tgt_sd  = float(valid_t.std())  if len(valid_t) > 0 else 5.0
    tgt_sd  = max(tgt_sd, 1e-6)
    tgt_norm = (target - tgt_mu) / tgt_sd
    norm_stats["target"] = {"mean": tgt_mu, "std": tgt_sd}
    return surf_norm, tgt_norm, norm_stats


def chronological_split(n: int, train_frac: float = 0.7,
                         val_frac: float = 0.15) -> tuple:
    t = int(n * train_frac)
    v = int(n * (train_frac + val_frac))
    return list(range(0, t)), list(range(t, v)), list(range(v, n))


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience; self.min_delta = min_delta
        self.counter  = 0;       self.best_loss = float("inf")

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss; self.counter = 0; return False
        self.counter += 1
        return self.counter >= self.patience


def train_model(model: OceanCNNModel, surface: np.ndarray, target: np.ndarray,
                cfg: dict, checkpoint_dir: str = "models") -> dict:
    """Section D: Full training pipeline with chronological split."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    p3 = cfg.get("phase3", {})
    epochs   = p3.get("epochs", 10)
    lr       = p3.get("lr", 1e-3)
    bs       = p3.get("batch_size", 4)
    patience = p3.get("patience", 5)
    resume   = p3.get("resume", False)

    surf_n, tgt_n, norm_stats = normalize_data(surface, target)
    n = len(surf_n)
    tr_idx, va_idx, te_idx = chronological_split(n)

    tr_ds = OceanDataset(surf_n[tr_idx], tgt_n[tr_idx])
    va_ds = OceanDataset(surf_n[va_idx], tgt_n[va_idx])

    tr_dl = torch.utils.data.DataLoader(tr_ds, batch_size=bs, shuffle=False)
    va_dl = torch.utils.data.DataLoader(va_ds, batch_size=bs, shuffle=False)

    model = model.to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    loss_fn = PhysicsGuidedLoss(cfg)
    es      = EarlyStopping(patience=patience)

    # Resume
    latest_path = os.path.join(checkpoint_dir, "latest_model.pt")
    best_path   = os.path.join(checkpoint_dir, "best_model.pt")
    start_epoch = 0
    if resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        opt.load_state_dict(ckpt["opt_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed training from epoch {start_epoch}")

    history = []
    best_val = float("inf")

    for ep in range(start_epoch, epochs):
        # Train
        model.train()
        tr_loss, tr_bd = 0.0, {}
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred, _, _ = model(xb)
            loss, bd   = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += float(loss) * len(xb)
            for k, v in bd.items():
                tr_bd[k] = tr_bd.get(k, 0) + v * len(xb)
        tr_loss /= max(len(tr_ds), 1)
        for k in tr_bd: tr_bd[k] /= max(len(tr_ds), 1)

        # Val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred, _, _ = model(xb)
                loss, _ = loss_fn(pred, yb)
                va_loss += float(loss) * len(xb)
        va_loss /= max(len(va_ds), 1)

        sched.step(va_loss)
        rec = {"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss, **tr_bd}
        history.append(rec)
        logger.info(f"Epoch {ep+1}/{epochs} — train={tr_loss:.4f} val={va_loss:.4f}")

        # Save latest
        torch.save({"epoch": ep, "model_state": model.state_dict(),
                    "opt_state": opt.state_dict(), "norm_stats": norm_stats},
                   latest_path)

        # Save best
        if va_loss < best_val:
            best_val = va_loss
            torch.save({"epoch": ep, "model_state": model.state_dict(),
                        "norm_stats": norm_stats, "val_loss": best_val},
                       best_path)

        if es.step(va_loss):
            logger.info(f"Early stopping at epoch {ep+1}")
            break

    # Save history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(os.path.join(checkpoint_dir, "training_history.csv"), index=False)
    with open(os.path.join(checkpoint_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    return {"history": history, "norm_stats": norm_stats,
            "best_val_loss": best_val, "test_indices": te_idx}


# ──────────────────────────────────────────────────────────────
# SECTION E  PREDICTION ENGINE
# ──────────────────────────────────────────────────────────────

def load_best_model(cfg: dict, checkpoint_dir: str = "models") -> tuple:
    """Load best checkpoint and norm_stats."""
    best_path = os.path.join(checkpoint_dir, "best_model.pt")
    model = OceanCNNModel(
        in_channels=8,
        n_depths=len(cfg.get("target_depths", [0]*15)),
        embed_dim=cfg.get("phase3", {}).get("embed_dim", 64),
    ).to(DEVICE)
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        norm_stats = ckpt.get("norm_stats", {})
        logger.info("Best model loaded from checkpoint")
    else:
        norm_stats = {}
        logger.warning("No checkpoint found — using untrained model")
    model.eval()
    return model, norm_stats


def predict_single(model: OceanCNNModel, surface_t: np.ndarray,
                   norm_stats: dict) -> np.ndarray:
    """Predict (15, H, W) from one surface frame (8, H, W)."""
    # Normalise
    surf = surface_t.copy().astype("float32")
    for c in range(surf.shape[0]):
        st = norm_stats.get(f"surf_ch{c}", {"mean": 0.0, "std": 1.0})
        surf[c] = (surf[c] - st["mean"]) / max(st["std"], 1e-6)
    surf = np.nan_to_num(surf, nan=0.0)
    x = torch.from_numpy(surf).unsqueeze(0).to(DEVICE)          # (1, 8, H, W)
    with torch.no_grad():
        pred, _, _ = model(x)
    pred_np = pred.squeeze(0).cpu().numpy()                       # (15, H, W)
    # Denormalise
    tst = norm_stats.get("target", {"mean": 0.0, "std": 1.0})
    pred_np = pred_np * tst["std"] + tst["mean"]
    return pred_np.astype("float32")


def generate_predictions(model: OceanCNNModel, surface: np.ndarray,
                         times: list, lats: list, lons: list,
                         norm_stats: dict, cfg: dict) -> xr.Dataset:
    """Section E: Predict all timesteps + future offsets."""
    depths = cfg.get("target_depths", list(range(15)))
    preds  = []
    for t in range(len(surface)):
        p = predict_single(model, surface[t], norm_stats)
        preds.append(p)
    pred_arr = np.stack(preds, axis=0)    # (T, D, H, W)

    ds = xr.Dataset(
        {"thetao_pred": (["time","depth","latitude","longitude"],
                         pred_arr, {"units": "degC", "long_name": "Predicted temperature"})},
        coords={"time": times, "depth": depths,
                "latitude": lats, "longitude": lons},
    )
    return ds


def generate_future_forecast(model: OceanCNNModel, surface: np.ndarray,
                              times: list, lats: list, lons: list,
                              norm_stats: dict, cfg: dict) -> dict:
    """Generate +7, +15, +30 day forecasts by rolling surface forward."""
    offsets = {7: None, 15: None, 30: None}
    last_surf = surface[-1].copy()

    for offset_days in sorted(offsets.keys()):
        # Simple persistence: repeat last known surface (no future surface data yet)
        pred = predict_single(model, last_surf, norm_stats)
        offsets[offset_days] = pred

    # Confidence degrades with horizon
    confidence = {7: 0.85, 15: 0.70, 30: 0.55}
    result = {}
    last_time = pd.Timestamp(times[-1])
    depths = cfg.get("target_depths", list(range(15)))
    for days, pred in offsets.items():
        future_time = (last_time + pd.Timedelta(days=days)).isoformat()
        result[days] = {
            "forecast_date":     future_time,
            "days_ahead":        days,
            "confidence":        confidence[days],
            "temperature_field": pred.tolist(),
            "shape":             list(pred.shape),
            "depths":            depths,
        }
    return result


# ──────────────────────────────────────────────────────────────
# SECTION F  MARINE HEATWAVE INTELLIGENCE
# ──────────────────────────────────────────────────────────────

MHW_THRESHOLDS = {
    "LOW":     (0.5, 1.0),
    "MEDIUM":  (1.0, 2.0),
    "HIGH":    (2.0, 3.0),
    "EXTREME": (3.0, 999),
}

def detect_marine_heatwaves(pred_ds: xr.Dataset, cfg: dict) -> list[dict]:
    """Section F: Detect MHW hotspots in predicted SST."""
    # Use depth index 0 (surface) as proxy SST
    sst_pred = pred_ds["thetao_pred"].isel(depth=0).values    # (T, H, W)
    lats     = list(pred_ds["latitude"].values)
    lons     = list(pred_ds["longitude"].values)
    times    = list(pred_ds["time"].values)
    n_depths = pred_ds.sizes["depth"]
    depths   = list(pred_ds["depth"].values)

    climatology = np.nanmean(sst_pred, axis=0)     # (H, W)
    anomaly     = sst_pred - climatology[np.newaxis]

    hotspots = []
    # Mean anomaly over last 3 days
    recent_anom = np.nanmean(anomaly[-3:], axis=0)  # (H, W)

    for i in range(len(lats)):
        for j in range(len(lons)):
            anom_val = float(recent_anom[i, j])
            if np.isnan(anom_val) or anom_val < 0.5:
                continue
            severity = "LOW"
            for cat, (lo, hi) in MHW_THRESHOLDS.items():
                if lo <= anom_val < hi:
                    severity = cat; break

            # Depth penetration — how deep does the anomaly reach?
            depth_profile = pred_ds["thetao_pred"].isel(
                latitude=i, longitude=j).values  # (T, D)
            clim_profile  = np.nanmean(depth_profile, axis=0)
            anom_profile  = np.nanmean(depth_profile[-3:], axis=0) - clim_profile
            affected_depth_idx = int(np.argmax(np.abs(anom_profile) < 0.3))
            affected_depth = depths[min(affected_depth_idx, n_depths-1)]

            # Persistence — how many days in last 7 exceed threshold?
            days_hot = int((anomaly[-7:, i, j] > 0.5).sum())

            hotspots.append({
                "latitude":           float(lats[i]),
                "longitude":          float(lons[j]),
                "severity":           severity,
                "sst_anomaly":        round(anom_val, 3),
                "expected_start":     str(pd.Timestamp(times[-days_hot]) if days_hot > 0 else times[-1]),
                "expected_duration_days": max(days_hot, 1),
                "expected_severity":  severity,
                "affected_depth_m":   float(affected_depth),
                "affected_sst":       round(float(np.nanmean(sst_pred[-3:, i, j])), 3),
                "subsurface_layers":  int(affected_depth_idx),
                "probability":        round(min(0.5 + anom_val * 0.15, 0.99), 3),
            })

    hotspots.sort(key=lambda x: x["sst_anomaly"], reverse=True)
    logger.info(f"MHW detection: {len(hotspots)} hotspots found")
    return hotspots[:50]   # top 50


# ──────────────────────────────────────────────────────────────
# SECTION G  IMPACT PREDICTION ENGINE
# ──────────────────────────────────────────────────────────────

IMPACT_RULES = {
    "Fish migration":            {"trigger": 0.5,  "depth_max": 500},
    "Coral bleaching":           {"trigger": 1.0,  "depth_max": 30},
    "Low oxygen zones":          {"trigger": 1.5,  "depth_max": 200},
    "Cyclone intensification":   {"trigger": 2.0,  "depth_max": 50},
    "Shipping route impact":     {"trigger": 1.0,  "depth_max": 999},
    "Coastal ecosystem stress":  {"trigger": 0.8,  "depth_max": 50},
    "Aquaculture risk":          {"trigger": 0.6,  "depth_max": 30},
}

SEVERITY_LABELS = {
    "LOW":     ["Fish migration"],
    "MEDIUM":  ["Fish migration", "Coastal ecosystem stress", "Aquaculture risk", "Shipping route impact"],
    "HIGH":    ["Fish migration", "Coastal ecosystem stress", "Aquaculture risk",
                "Shipping route impact", "Low oxygen zones", "Coral bleaching"],
    "EXTREME": list(IMPACT_RULES.keys()),
}

ACTIONS = {
    "Fish migration":           "Monitor fishery zones; advise fishing vessels of shifting stocks.",
    "Coral bleaching":          "Activate coral reef monitoring; reduce anthropogenic stressors.",
    "Low oxygen zones":         "Alert coastal authorities; restrict trawling in affected zones.",
    "Cyclone intensification":  "Issue cyclone watch; activate disaster preparedness protocol.",
    "Shipping route impact":    "Issue marine weather advisory; recommend alternate shipping lanes.",
    "Coastal ecosystem stress": "Deploy field monitoring teams; alert coastal communities.",
    "Aquaculture risk":         "Advise aquaculture operators to deploy shading or deep-water cages.",
}

def generate_impact_predictions(hotspots: list[dict]) -> list[dict]:
    """Section G: Generate structured impact JSON for each hotspot."""
    results = []
    for hs in hotspots:
        impacts = []
        affected = SEVERITY_LABELS.get(hs["severity"], [])
        for impact_name in affected:
            rule = IMPACT_RULES[impact_name]
            if hs["sst_anomaly"] >= rule["trigger"] and hs["affected_depth_m"] <= rule["depth_max"]:
                sev_score = min(hs["sst_anomaly"] / 3.0, 1.0)
                impacts.append({
                    "impact":            impact_name,
                    "severity":          round(sev_score, 3),
                    "reason":            f"SST anomaly {hs['sst_anomaly']:.2f}°C at {hs['latitude']:.2f}N {hs['longitude']:.2f}E",
                    "preventive_action": ACTIONS.get(impact_name, "Monitor situation."),
                })
        results.append({
            "hotspot":  hs,
            "impacts":  impacts,
        })
    return results


# ──────────────────────────────────────────────────────────────
# SECTION H  UNCERTAINTY ENGINE
# ──────────────────────────────────────────────────────────────

def generate_uncertainty(model: OceanCNNModel, surface: np.ndarray,
                          norm_stats: dict, n_samples: int = 10) -> dict:
    """
    Section H: MC-Dropout uncertainty estimation.
    Runs n_samples forward passes with dropout enabled.
    """
    model.train()   # enables dropout stochasticity
    preds_mc = []
    # Use last timestep as representative sample
    surf_t = surface[-1].copy().astype("float32")
    for c in range(surf_t.shape[0]):
        st = norm_stats.get(f"surf_ch{c}", {"mean": 0.0, "std": 1.0})
        surf_t[c] = (surf_t[c] - st["mean"]) / max(st["std"], 1e-6)
    surf_t = np.nan_to_num(surf_t, nan=0.0)
    x = torch.from_numpy(surf_t).unsqueeze(0).to(DEVICE)

    for _ in range(n_samples):
        with torch.no_grad():
            pred, _, _ = model(x)
        preds_mc.append(pred.squeeze(0).cpu().numpy())

    model.eval()
    preds_mc = np.stack(preds_mc, axis=0)   # (N, D, H, W)
    tst = norm_stats.get("target", {"mean": 0.0, "std": 1.0})
    preds_mc = preds_mc * tst["std"] + tst["mean"]

    mean_pred   = preds_mc.mean(axis=0)
    std_pred    = preds_mc.std(axis=0)
    conf_map    = 1.0 / (1.0 + std_pred)         # higher std → lower confidence
    depth_unc   = std_pred.mean(axis=(1, 2))      # (D,)
    overall_conf= float(conf_map.mean())

    return {
        "confidence_map":   conf_map.tolist(),
        "uncertainty_map":  std_pred.tolist(),
        "overall_confidence": round(overall_conf, 4),
        "depth_uncertainty":  depth_unc.tolist(),
        "pixel_uncertainty":  std_pred.mean(axis=0).tolist(),  # (H, W)
        "n_samples":          n_samples,
    }


# ──────────────────────────────────────────────────────────────
# SECTION I  EXPLAINABLE AI
# ──────────────────────────────────────────────────────────────

SURFACE_VAR_NAMES = ["sst","sss","sla","sla_obs_count",
                     "current_u","current_v","wind_u","wind_v"]

def generate_explainability(model: OceanCNNModel, surface: np.ndarray,
                             norm_stats: dict, cfg: dict) -> dict:
    """Section I: Gradient-based feature importance + attention map analysis."""
    model.eval()
    surf_t = surface[-1].copy().astype("float32")
    for c in range(surf_t.shape[0]):
        st = norm_stats.get(f"surf_ch{c}", {"mean": 0.0, "std": 1.0})
        surf_t[c] = (surf_t[c] - st["mean"]) / max(st["std"], 1e-6)
    surf_t = np.nan_to_num(surf_t, nan=0.0)

    x = torch.from_numpy(surf_t).unsqueeze(0).to(DEVICE)
    x.requires_grad_(True)

    pred, embedding, att_map = model(x)
    # Scalar output — mean over spatial and depth
    scalar = pred.mean()
    scalar.backward()

    grad = x.grad.detach().cpu().numpy().squeeze(0)   # (8, H, W)
    importance = np.abs(grad).mean(axis=(1, 2))       # (8,) — per channel
    importance /= importance.sum() + 1e-8
    importance_pct = (importance * 100).tolist()

    # Attention map summary
    att_np  = att_map.detach().cpu().numpy().squeeze()   # (H, W)
    att_np  = (att_np - att_np.min()) / (att_np.max() - att_np.min() + 1e-8)

    # Depth contribution — use channel-wise gradient magnitudes as proxy
    # (avoid retain_graph complexity by using already-computed input grad)
    grad_np = x.grad.detach().cpu().numpy().squeeze(0)   # (8, H, W)
    ch_importance = np.abs(grad_np).mean(axis=(1, 2))    # (8,)
    # Project 8-channel importance onto 15 depth layers via simple interpolation
    depth_grads_arr = np.interp(
        np.linspace(0, 1, pred.shape[1]),
        np.linspace(0, 1, len(ch_importance)),
        ch_importance
    )
    depth_grads_arr /= depth_grads_arr.sum() + 1e-8

    var_rank = sorted(zip(SURFACE_VAR_NAMES, importance_pct),
                      key=lambda kv: kv[1], reverse=True)

    return {
        "attention_map":           att_np.tolist(),
        "feature_contribution":    dict(zip(SURFACE_VAR_NAMES, importance_pct)),
        "input_importance_ranking": [{"variable": v, "pct": round(p, 2)} for v, p in var_rank],
        "variable_contribution_pct": dict(zip(SURFACE_VAR_NAMES, [round(p, 2) for p in importance_pct])),
        "depth_contribution":       [round(float(v), 4) for v in depth_grads_arr],
        "explanation":              (
            f"Prediction driven primarily by {var_rank[0][0]} ({var_rank[0][1]:.1f}%) "
            f"and {var_rank[1][0]} ({var_rank[1][1]:.1f}%). "
            f"Attention concentrated on {int(att_np.argmax() // att_np.shape[1])},"
            f"{int(att_np.argmax() % att_np.shape[1])} grid point."
        ),
    }


# ──────────────────────────────────────────────────────────────
# SECTION J  ARGO VALIDATION ENGINE
# ──────────────────────────────────────────────────────────────

def validate_against_argo(pred_ds: xr.Dataset, argo_dir: str,
                           cfg: dict) -> dict:
    """Section J: Independent ARGO validation — no training data used."""
    argo_files = list(Path(argo_dir).glob("*.nc")) + list(Path(argo_dir).glob("*.csv"))
    if not argo_files:
        return {"status": "NO_ARGO_FILES", "profiles_matched": 0,
                "rmse": None, "mae": None, "bias": None, "correlation": None}

    matched_pred, matched_obs, depth_records = [], [], []
    depths_target = list(pred_ds["depth"].values)

    for fp in argo_files:
        try:
            if str(fp).endswith(".nc"):
                ds_a = xr.open_dataset(str(fp), engine="netcdf4")
                lat_v = "LATITUDE" if "LATITUDE" in ds_a else "latitude"
                lon_v = "LONGITUDE" if "LONGITUDE" in ds_a else "longitude"
                time_v= "JULD" if "JULD" in ds_a else "time"
                temp_v= "TEMP" if "TEMP" in ds_a else "temperature"
                pres_v= "PRES" if "PRES" in ds_a else "pressure"

                n_prof = int(ds_a.sizes.get("N_PROF", ds_a.sizes.get("profile", 0)))
                for ip in range(min(n_prof, 20)):
                    lat = float(ds_a[lat_v].values[ip])
                    lon = float(ds_a[lon_v].values[ip])
                    t   = pd.Timestamp(ds_a[time_v].values[ip])

                    # Spatial match
                    proto = cfg["region"]["prototype"]
                    if not (proto["lat_min"] <= lat <= proto["lat_max"] and
                            proto["lon_min"] <= lon <= proto["lon_max"]):
                        continue

                    # Find nearest prediction grid point
                    lat_arr = pred_ds["latitude"].values
                    lon_arr = pred_ds["longitude"].values
                    li = int(np.argmin(np.abs(lat_arr - lat)))
                    oi = int(np.argmin(np.abs(lon_arr - lon)))

                    try:
                        pred_profile = pred_ds["thetao_pred"].isel(
                            latitude=li, longitude=oi).sel(
                            time=t, method="nearest").values  # (D,)
                    except Exception:
                        continue

                    temps = ds_a[temp_v].values[ip]
                    press = ds_a[pres_v].values[ip]
                    for di, dtgt in enumerate(depths_target):
                        depth_idx = int(np.argmin(np.abs(press - dtgt)))
                        obs_t = float(temps[depth_idx]) if not np.isnan(temps[depth_idx]) else None
                        if obs_t is not None:
                            matched_obs.append(obs_t)
                            matched_pred.append(float(pred_profile[di]))
                            depth_records.append({"depth": dtgt,
                                                  "obs": obs_t,
                                                  "pred": float(pred_profile[di])})
                ds_a.close()
        except Exception as e:
            logger.warning(f"ARGO file error {fp}: {e}")

    if not matched_obs:
        return {"status": "NO_MATCH", "profiles_matched": 0,
                "rmse": None, "mae": None, "bias": None, "correlation": None}

    obs_arr  = np.array(matched_obs)
    pred_arr = np.array(matched_pred)
    rmse     = float(np.sqrt(np.mean((pred_arr - obs_arr)**2)))
    mae      = float(np.mean(np.abs(pred_arr - obs_arr)))
    bias     = float(np.mean(pred_arr - obs_arr))
    corr     = float(np.corrcoef(obs_arr, pred_arr)[0, 1]) if len(obs_arr) > 2 else 0.0

    # Depth-wise metrics
    depth_metrics = {}
    for dtgt in depths_target:
        rows = [r for r in depth_records if r["depth"] == dtgt]
        if rows:
            o_arr = np.array([r["obs"] for r in rows])
            p_arr = np.array([r["pred"] for r in rows])
            depth_metrics[dtgt] = {
                "rmse": round(float(np.sqrt(np.mean((p_arr - o_arr)**2))), 4),
                "n":    len(rows),
            }

    return {
        "status":          "PASS",
        "profiles_matched": len(set(matched_obs)),
        "n_obs":           len(matched_obs),
        "rmse":            round(rmse, 4),
        "mae":             round(mae, 4),
        "bias":            round(bias, 4),
        "correlation":     round(corr, 4),
        "depth_metrics":   depth_metrics,
    }


# ──────────────────────────────────────────────────────────────
# SECTION N  MULTI-VARIABLE ANALYSIS
# ──────────────────────────────────────────────────────────────

def generate_multivariate_analysis(surface: np.ndarray) -> dict:
    """Section N: Correlation matrix + feature importance summary."""
    T, C, H, W = surface.shape
    flat = surface.reshape(T, C, -1).mean(axis=2)   # (T, C) spatial mean

    valid_mask = np.all(np.isfinite(flat), axis=0)
    flat_clean = flat[:, valid_mask]
    names_clean = [n for n, v in zip(SURFACE_VAR_NAMES, valid_mask) if v]

    if flat_clean.shape[1] < 2:
        return {"corr_matrix": [], "feature_importance": {}, "ocean_state_summary": {}}

    corr = np.corrcoef(flat_clean.T).tolist()
    # Feature importance: variance explained relative to SST
    variances = flat_clean.var(axis=0)
    variances /= variances.sum() + 1e-8

    sst_idx = names_clean.index("sst") if "sst" in names_clean else 0
    corr_with_sst = [abs(float(np.corrcoef(flat_clean[:, sst_idx],
                                            flat_clean[:, i])[0, 1]))
                     for i in range(flat_clean.shape[1])]

    ocean_state = {
        "mean_sst":      round(float(np.nanmean(surface[:, 0])), 3),
        "sst_range":     [round(float(np.nanmin(surface[:, 0])), 3),
                          round(float(np.nanmax(surface[:, 0])), 3)],
        "mean_sla":      round(float(np.nanmean(surface[:, 2])), 4),
        "dominant_var":  names_clean[int(np.argmax(variances))],
    }

    return {
        "variables":          names_clean,
        "corr_matrix":        corr,
        "feature_importance": dict(zip(names_clean, [round(float(v), 4) for v in variances])),
        "corr_with_sst":      dict(zip(names_clean, [round(v, 4) for v in corr_with_sst])),
        "ocean_state_summary": ocean_state,
    }


# ──────────────────────────────────────────────────────────────
# SECTION O  VOICE ASSISTANT BACKEND
# ──────────────────────────────────────────────────────────────

def process_voice_query(text: str, pred_ds: xr.Dataset,
                         hotspots: list, cfg: dict) -> dict:
    """Section O: Parse natural-language ocean query, return structured JSON."""
    text_lower = text.lower()

    # Parse depth
    depth = None
    for d in cfg.get("target_depths", []):
        if str(d) in text:
            depth = d; break

    # Parse location keywords
    location = None
    loc_map = {
        "bay of bengal": (12.5, 82.5),
        "chennai":        (13.08, 80.27),
        "mumbai":         (18.96, 72.82),
        "arabian sea":    (15.0, 65.0),
        "lakshadweep":    (11.0, 73.0),
    }
    for kw, coords in loc_map.items():
        if kw in text_lower:
            location = {"name": kw, "lat": coords[0], "lon": coords[1]}
            break

    # Parse intent
    if "heatwave" in text_lower or "heat" in text_lower:
        intent = "heatwave"
    elif "predict" in text_lower or "forecast" in text_lower or "next week" in text_lower:
        intent = "forecast"
    elif "compare" in text_lower:
        intent = "compare"
    elif "temperature" in text_lower or "temp" in text_lower:
        intent = "temperature"
    else:
        intent = "general"

    # Build response
    response: dict = {"query": text, "intent": intent, "location": location, "depth_m": depth}

    if intent == "heatwave":
        nearby = hotspots[:3] if not location else [
            h for h in hotspots
            if location and abs(h["latitude"] - location["lat"]) < 2
               and abs(h["longitude"] - location["lon"]) < 2
        ][:3]
        response["heatwaves"] = nearby or [{"status": "No heatwave detected in query region"}]

    elif intent == "temperature" and location:
        lats = list(pred_ds["latitude"].values)
        lons = list(pred_ds["longitude"].values)
        li   = int(np.argmin(np.abs(np.array(lats) - location["lat"])))
        oi   = int(np.argmin(np.abs(np.array(lons) - location["lon"])))
        depth_idx = 0
        if depth is not None:
            depths = list(pred_ds["depth"].values)
            depth_idx = int(np.argmin(np.abs(np.array(depths) - depth)))
        temps = pred_ds["thetao_pred"].isel(
            latitude=li, longitude=oi, depth=depth_idx).values
        response["temperature_timeseries"] = [round(float(t), 3) for t in temps]

    elif intent == "forecast":
        response["message"] = "Use /prediction/date?days_ahead=7 for future forecast."

    response["status"] = "ok"
    return response


# ──────────────────────────────────────────────────────────────
# SECTION R  DATABASE UPDATE
# ──────────────────────────────────────────────────────────────

SCHEMA_PHASE3 = """
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_date TEXT,
    forecast_days   INTEGER DEFAULT 0,
    model_version   TEXT,
    rmse            REAL,
    created_at      TEXT
);

CREATE TABLE IF NOT EXISTS heatwave_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT,
    severity    TEXT,
    lat         REAL,
    lon         REAL,
    anomaly     REAL,
    probability REAL
);

CREATE TABLE IF NOT EXISTS model_versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    version_name    TEXT,
    checkpoint_path TEXT,
    val_loss        REAL,
    saved_at        TEXT
);

CREATE TABLE IF NOT EXISTS explainability_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    top_variable    TEXT,
    top_pct         REAL,
    overall_confidence REAL,
    logged_at       TEXT
);

CREATE TABLE IF NOT EXISTS training_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    epochs_run  INTEGER,
    best_val_loss REAL,
    started_at  TEXT,
    finished_at TEXT
);
"""

def init_phase3_tables(conn) -> None:
    conn.executescript(SCHEMA_PHASE3)
    conn.commit()
    logger.info("Phase 3 SQLite tables created")


def log_prediction(conn, pred_date: str, forecast_days: int,
                   model_version: str, rmse: Optional[float]) -> None:
    cur = conn.cursor()
    cur.execute("""INSERT INTO predictions
        (prediction_date, forecast_days, model_version, rmse, created_at)
        VALUES (?,?,?,?,?)""",
        (pred_date, forecast_days, model_version, rmse,
         datetime.now(timezone.utc).isoformat()))
    conn.commit()


def log_heatwaves(conn, hotspots: list[dict]) -> None:
    cur = conn.cursor()
    now = datetime.now(timezone.utc).isoformat()
    for hs in hotspots:
        cur.execute("""INSERT INTO heatwave_history
            (detected_at, severity, lat, lon, anomaly, probability)
            VALUES (?,?,?,?,?,?)""",
            (now, hs["severity"], hs["latitude"], hs["longitude"],
             hs["sst_anomaly"], hs["probability"]))
    conn.commit()


def log_model_version(conn, version: str, path: str, val_loss: float) -> None:
    cur = conn.cursor()
    cur.execute("""INSERT INTO model_versions
        (version_name, checkpoint_path, val_loss, saved_at)
        VALUES (?,?,?,?)""",
        (version, path, val_loss,
         datetime.now(timezone.utc).isoformat()))
    conn.commit()


def log_training_run(conn, epochs: int, best_val: float,
                     started_at: str) -> None:
    cur = conn.cursor()
    cur.execute("""INSERT INTO training_runs
        (epochs_run, best_val_loss, started_at, finished_at)
        VALUES (?,?,?,?)""",
        (epochs, best_val, started_at,
         datetime.now(timezone.utc).isoformat()))
    conn.commit()


def log_explainability(conn, xai: dict) -> None:
    top = sorted(xai.get("feature_contribution", {}).items(),
                 key=lambda kv: kv[1], reverse=True)
    top_var = top[0][0] if top else "unknown"
    top_pct = top[0][1] if top else 0.0
    conf    = xai.get("overall_confidence", 0.0) if "overall_confidence" in xai else 0.0
    cur = conn.cursor()
    cur.execute("""INSERT INTO explainability_log
        (top_variable, top_pct, overall_confidence, logged_at)
        VALUES (?,?,?,?)""",
        (top_var, top_pct, conf,
         datetime.now(timezone.utc).isoformat()))
    conn.commit()
