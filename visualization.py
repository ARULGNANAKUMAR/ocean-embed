"""
visualization.py
North Indian Ocean Intelligence Platform — Phase 3
All backend PNG generation for Section T.
"""

import logging
import os

import numpy as np

logger = logging.getLogger("ocean_platform")


def _safe_imshow(ax, arr, **kwargs):
    data = np.array(arr, dtype="float64")
    data[~np.isfinite(data)] = np.nan
    return ax.imshow(data, **kwargs)


def generate_phase3_plots(pred_ds, surface_ds, training_history: list,
                           hotspots: list, xai: dict, uncertainty: dict,
                           argo_result: dict, plots_dir: str) -> None:
    """Section T: Generate all Phase 3 backend visualisations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        os.makedirs(plots_dir, exist_ok=True)

        # ── 1. Temperature Heatmap (mean SST prediction) ──────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        if pred_ds is not None:
            sst_mean = pred_ds["thetao_pred"].isel(depth=0).mean(dim="time").values
            im = _safe_imshow(ax, sst_mean, cmap="RdYlBu_r", origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax, label="Mean SST Prediction (°C)")
            lats = list(pred_ds["latitude"].values)
            lons = list(pred_ds["longitude"].values)
            ax.set_xticks(range(0, len(lons), 4))
            ax.set_xticklabels([f"{lons[i]:.1f}°E" for i in range(0, len(lons), 4)], fontsize=7)
            ax.set_yticks(range(0, len(lats), 4))
            ax.set_yticklabels([f"{lats[i]:.1f}°N" for i in range(0, len(lats), 4)], fontsize=7)
        else:
            ax.text(0.5, 0.5, "No prediction data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Temperature Heatmap — Mean Predicted SST")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "temperature_heatmap.png"), dpi=100)
        plt.close(fig)

        # ── 2. Depth Heatmap (temperature vs depth vs time) ───────────
        fig, ax = plt.subplots(figsize=(12, 5))
        if pred_ds is not None:
            arr = pred_ds["thetao_pred"].mean(dim=["latitude","longitude"]).values  # (T, D)
            im  = _safe_imshow(ax, arr.T, cmap="RdYlBu_r", origin="upper", aspect="auto")
            plt.colorbar(im, ax=ax, label="Temperature (°C)")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Depth Index (shallow→deep)")
        ax.set_title("Depth-Time Temperature Heatmap")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "depth_heatmap.png"), dpi=100)
        plt.close(fig)

        # ── 3. Marine Heatwave Map ─────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        if pred_ds is not None:
            sst_all  = pred_ds["thetao_pred"].isel(depth=0).values
            clim     = np.nanmean(sst_all, axis=0)
            anomaly  = np.nanmean(sst_all[-3:], axis=0) - clim
            im = _safe_imshow(ax, anomaly, cmap="hot", vmin=0, vmax=3,
                              origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax, label="SST Anomaly (°C)")
            # Overlay hotspot markers
            if hotspots:
                lats = list(pred_ds["latitude"].values)
                lons = list(pred_ds["longitude"].values)
                for hs in hotspots[:10]:
                    li = int(np.argmin(np.abs(np.array(lats) - hs["latitude"])))
                    oi = int(np.argmin(np.abs(np.array(lons) - hs["longitude"])))
                    col = {"LOW":"yellow","MEDIUM":"orange","HIGH":"red","EXTREME":"darkred"}
                    ax.plot(oi, li, "^", color=col.get(hs["severity"],"red"),
                            ms=8, markeredgecolor="white", markeredgewidth=0.5)
        ax.set_title("Marine Heatwave Map — SST Anomaly (Last 3 Days)")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "marine_heatwave_map.png"), dpi=100)
        plt.close(fig)

        # ── 4. Attention Map ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 6))
        att = xai.get("attention_map")
        if att:
            arr_att = np.array(att)
            im = _safe_imshow(ax, arr_att, cmap="plasma", vmin=0, vmax=1,
                              origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax, label="Attention Weight")
        else:
            ax.text(0.5, 0.5, "No attention data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title("Spatial Attention Map")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "attention_map.png"), dpi=100)
        plt.close(fig)

        # ── 5. Embedding PCA ──────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 5))
        try:
            from sklearn.decomposition import PCA
            if pred_ds is not None:
                arr_flat = pred_ds["thetao_pred"].values.reshape(
                    pred_ds.sizes["time"], -1)
                valid_cols = np.all(np.isfinite(arr_flat), axis=0)
                if valid_cols.sum() >= 2:
                    pca  = PCA(n_components=2)
                    embs = pca.fit_transform(arr_flat[:, valid_cols])
                    sc   = ax.scatter(embs[:, 0], embs[:, 1],
                                      c=range(len(embs)), cmap="viridis", s=30)
                    plt.colorbar(sc, ax=ax, label="Time step")
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                else:
                    ax.text(0.5, 0.5, "Insufficient data for PCA",
                            ha="center", va="center", transform=ax.transAxes)
        except Exception as e:
            ax.text(0.5, 0.5, f"PCA error: {e}", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title("Embedding PCA — Temporal Trajectory")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "embedding_pca.png"), dpi=100)
        plt.close(fig)

        # ── 6. Training Curve ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 4))
        if training_history:
            epochs    = [r["epoch"] for r in training_history]
            tr_losses = [r.get("train_loss", 0) for r in training_history]
            va_losses = [r.get("val_loss", 0) for r in training_history]
            ax.plot(epochs, tr_losses, label="Train loss", color="#2196F3", linewidth=1.8)
            ax.plot(epochs, va_losses, label="Val loss",   color="#F44336", linewidth=1.8)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No training history", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title("Training & Validation Loss Curve")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "training_curve.png"), dpi=100)
        plt.close(fig)

        # ── 7. RMSE vs Depth ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 7))
        depth_metrics = argo_result.get("depth_metrics", {})
        if depth_metrics:
            dm_depths = [float(d) for d in depth_metrics.keys()]
            dm_rmse   = [m["rmse"] for m in depth_metrics.values()]
            ax.barh(dm_depths, dm_rmse, height=8, color="#2196F3", alpha=0.8)
            ax.set_xlabel("RMSE (°C)")
            ax.set_ylabel("Depth (m)")
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "No ARGO depth metrics", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title("RMSE vs Depth — ARGO Validation")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "rmse_vs_depth.png"), dpi=100)
        plt.close(fig)

        # ── 8. Correlation vs Depth ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 7))
        if pred_ds is not None:
            depths   = list(pred_ds["depth"].values)
            arr_pred = pred_ds["thetao_pred"].values   # (T, D, H, W)
            corrs    = []
            for di in range(len(depths)):
                layer = arr_pred[:, di].flatten()
                layer = layer[np.isfinite(layer)]
                # Proxy: autocorrelation lag-1
                if len(layer) > 2:
                    c = float(np.corrcoef(layer[:-1], layer[1:])[0, 1])
                else:
                    c = 0.0
                corrs.append(round(c, 4))
            ax.barh(depths, corrs, height=15, color="#4CAF50", alpha=0.8)
            ax.set_xlabel("Temporal Autocorrelation")
            ax.set_ylabel("Depth (m)")
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "No prediction data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title("Temporal Autocorrelation vs Depth")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "correlation_vs_depth.png"), dpi=100)
        plt.close(fig)

        # ── 9. Confidence Map ─────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 6))
        conf_map = uncertainty.get("confidence_map")
        if conf_map:
            arr_c = np.array(conf_map)
            if arr_c.ndim == 3:
                arr_c = arr_c.mean(axis=0)
            im = _safe_imshow(ax, arr_c, cmap="RdYlGn", vmin=0, vmax=1,
                              origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax, label="Confidence [0-1]")
        else:
            ax.text(0.5, 0.5, "No confidence data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title("Prediction Confidence Map")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "confidence_map.png"), dpi=100)
        plt.close(fig)

        # ── 10. Uncertainty Map ───────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 6))
        unc_map = uncertainty.get("uncertainty_map")
        if unc_map:
            arr_u = np.array(unc_map)
            if arr_u.ndim == 3:
                arr_u = arr_u.mean(axis=0)
            im = _safe_imshow(ax, arr_u, cmap="YlOrRd", origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax, label="Uncertainty (°C std)")
        else:
            ax.text(0.5, 0.5, "No uncertainty data", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title("Prediction Uncertainty Map (MC-Dropout)")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "uncertainty_map.png"), dpi=100)
        plt.close(fig)

        # ── 11. Prediction Difference Map ────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 6))
        if pred_ds is not None and pred_ds.sizes["time"] >= 2:
            first = pred_ds["thetao_pred"].isel(time=0, depth=0).values
            last  = pred_ds["thetao_pred"].isel(time=-1, depth=0).values
            diff  = last - first
            vmax  = float(np.nanpercentile(np.abs(diff), 95)) if np.isfinite(diff).any() else 1.0
            im = _safe_imshow(ax, diff, cmap="RdBu_r",
                              vmin=-vmax, vmax=vmax, origin="lower", aspect="auto")
            plt.colorbar(im, ax=ax, label="ΔTemperature (°C)")
        else:
            ax.text(0.5, 0.5, "Insufficient time steps", ha="center", va="center",
                    transform=ax.transAxes)
        ax.set_title("Prediction Difference Map (Last − First Day, Surface)")
        plt.tight_layout()
        fig.savefig(os.path.join(plots_dir, "prediction_difference_map.png"), dpi=100)
        plt.close(fig)

        logger.info(f"Phase 3 plots saved to {plots_dir}")

    except Exception as e:
        logger.error(f"Phase 3 plot generation error: {e}")
