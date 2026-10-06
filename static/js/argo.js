/* ═══════════════════════════════════════════════════════════
   argo.js — ARGO validation display (Section 18 Analytics tie-in)
   Used on dashboard/analytics pages to show RMSE/MAE/Bias/Corr.
   ═══════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", async () => {
  await loadArgoValidationPanel();
  await loadMultivarAnalysis();
});

async function loadArgoValidationPanel() {
  const panel = document.getElementById("argo-validation-panel");
  if (!panel) return;
  try {
    const data = await OceanApp.api("/api/argo/validation");
    if (data.status !== "PASS") {
      panel.innerHTML = `<div class="glass-card">ARGO validation status: ${data.status || "unavailable"}</div>`;
      return;
    }
    panel.innerHTML = `
      <div class="widget-grid cols-4">
        <div class="glass-card"><div class="card-title">RMSE</div><div class="card-value">${data.rmse}<span class="card-unit">°C</span></div></div>
        <div class="glass-card"><div class="card-title">MAE</div><div class="card-value">${data.mae}<span class="card-unit">°C</span></div></div>
        <div class="glass-card"><div class="card-title">Bias</div><div class="card-value">${data.bias}<span class="card-unit">°C</span></div></div>
        <div class="glass-card"><div class="card-title">Correlation</div><div class="card-value">${data.correlation}</div></div>
      </div>
    `;
    renderDepthRmseChart(data.depth_metrics);
  } catch (e) {
    panel.innerHTML = `<div class="glass-card">ARGO validation data unavailable.</div>`;
  }
}

function renderDepthRmseChart(depthMetrics) {
  const canvas = document.getElementById("depth-rmse-chart");
  if (!canvas || !depthMetrics || typeof Chart === "undefined") return;
  const depths = Object.keys(depthMetrics).map(Number).sort((a, b) => a - b);
  const rmses = depths.map((d) => depthMetrics[d]?.rmse ?? null);
  OceanCharts.horizontalBarChart(canvas, depths.map((d) => `${d}m`), rmses, "RMSE (°C)");
}

async function loadMultivarAnalysis() {
  const corrContainer = document.getElementById("correlation-matrix-container");
  const importanceCanvas = document.getElementById("feature-importance-chart");
  try {
    const data = await OceanApp.api("/api/multivar");
    if (corrContainer && data.corr_matrix && data.variables) {
      OceanCharts.heatmapMatrix(corrContainer, data.corr_matrix, data.variables);
    }
    if (importanceCanvas && data.feature_importance) {
      const labels = Object.keys(data.feature_importance);
      const values = Object.values(data.feature_importance);
      OceanCharts.barChart(importanceCanvas, labels, values, "Feature Importance");
    }
    const summaryEl = document.getElementById("ocean-state-summary");
    if (summaryEl && data.ocean_state_summary) {
      const s = data.ocean_state_summary;
      summaryEl.innerHTML = Object.entries(s).map(([k, v]) =>
        `<div class="gi-row"><span class="gi-label">${k.replace(/_/g," ")}</span><span class="gi-value">${Array.isArray(v) ? v.join(" – ") : v}</span></div>`
      ).join("");
    }
  } catch (e) { /* silent */ }
}
