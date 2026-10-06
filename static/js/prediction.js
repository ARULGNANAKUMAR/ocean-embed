/* ═══════════════════════════════════════════════════════════
   prediction.js — Future Prediction page (Section 12)
   Time slider (Past/Today/+7/+15/+30), difference heatmap,
   trend graph, satellite embedding visualization (Section 10).
   ═══════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", async () => {
  initTimeSlider();
  await loadTrendGraph();
  await loadEmbeddingVisualization();
});

const HORIZON_LABELS = ["Past", "Today", "+7 Days", "+15 Days", "+30 Days"];
const HORIZON_DAYS   = [-7, 0, 7, 15, 30];

function initTimeSlider() {
  const slider = document.getElementById("prediction-time-slider");
  const label  = document.getElementById("prediction-time-label");
  if (!slider) return;
  slider.min = 0; slider.max = HORIZON_LABELS.length - 1; slider.value = 1;

  async function update() {
    const idx = parseInt(slider.value);
    if (label) label.textContent = HORIZON_LABELS[idx];
    await loadFieldForHorizon(HORIZON_DAYS[idx]);
  }
  slider.addEventListener("input", update);
  update();
}

async function loadFieldForHorizon(days) {
  const canvas = document.getElementById("prediction-heatmap-canvas");
  const statsEl = document.getElementById("prediction-field-stats");
  try {
    let data;
    if (days <= 0) {
      data = await OceanApp.api(`/api/map/layer?layer=sst&time_idx=${days === 0 ? -1 : 0}`);
    } else {
      data = await OceanApp.api(`/api/prediction/date?days_ahead=${days}`);
    }
    const grid = data.grid || (data.temperature_field ? data.temperature_field[0] : null);
    if (canvas && grid) {
      OceanHeatmap.drawGridToCanvas(canvas, grid, { min: 20, max: 32 });
    }
    if (statsEl) {
      const flat = (grid || []).flat().filter((v) => v != null && isFinite(v));
      const mean = flat.length ? flat.reduce((a, b) => a + b, 0) / flat.length : null;
      statsEl.textContent = mean != null ? `Mean: ${mean.toFixed(2)}°C` : "No data";
    }
  } catch (e) {
    if (statsEl) statsEl.textContent = "Data unavailable";
  }
}

async function loadTrendGraph() {
  const canvas = document.getElementById("trend-chart");
  if (!canvas || typeof Chart === "undefined") return;
  try {
    const stats = await OceanApp.api("/api/statistics");
    const rows = (stats.statistics || []).filter((r) => r.variable && r.mean != null);
    new Chart(canvas, {
      type: "bar",
      data: {
        labels: rows.map((r) => r.variable),
        datasets: [{
          label: "Mean Value",
          data: rows.map((r) => r.mean),
          backgroundColor: "rgba(34,229,255,0.5)",
          borderColor: "#22e5ff",
          borderWidth: 1,
        }],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: {
          x: { ticks: { color: "#9db4cc" }, grid: { display: false } },
          y: { ticks: { color: "#9db4cc" }, grid: { color: "rgba(255,255,255,0.05)" } },
        },
        plugins: { legend: { display: false } },
      },
    });
  } catch (e) { console.warn("Trend graph failed", e); }
}

// ── Satellite Embedding Visualization (Section 10) ──
async function loadEmbeddingVisualization() {
  const canvas = document.getElementById("embedding-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  canvas.width = canvas.clientWidth; canvas.height = 320;

  let attention = null;
  try {
    const att = await OceanApp.api("/api/attention");
    attention = att.attention_map;
    const explEl = document.getElementById("xai-explanation");
    if (explEl) explEl.textContent = att.explanation || "No explanation available.";
  } catch (e) { /* continue with animation only */ }

  const particles = [];
  const stages = ["INPUT", "ENCODER", "EMBEDDING", "ATTENTION", "DECODER", "PREDICTION"];
  const stageX = stages.map((_, i) => (canvas.width / (stages.length - 1)) * i);

  for (let i = 0; i < 60; i++) {
    particles.push({
      x: 0, y: Math.random() * canvas.height,
      stage: 0, progress: Math.random(),
      speed: 0.004 + Math.random() * 0.006,
    });
  }

  function frame() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw stage lines + labels
    stageX.forEach((x, i) => {
      ctx.strokeStyle = "rgba(34,229,255,0.15)";
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
      ctx.fillStyle = "#22e5ff";
      ctx.font = "10px Orbitron, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(stages[i], x, 16);
    });

    // Draw particles flowing through stages
    particles.forEach((p) => {
      p.progress += p.speed;
      if (p.progress > 1) { p.progress = 0; p.stage = (p.stage + 1) % (stages.length - 1); }
      const x0 = stageX[p.stage], x1 = stageX[p.stage + 1] ?? stageX[p.stage];
      const x = x0 + (x1 - x0) * p.progress;
      ctx.beginPath();
      ctx.fillStyle = "rgba(34,229,255,0.85)";
      ctx.shadowColor = "#22e5ff";
      ctx.shadowBlur = 6;
      ctx.arc(x, p.y, 2.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
    });

    requestAnimationFrame(frame);
  }
  frame();

  // Attention heatmap mini-canvas
  if (attention) {
    const attCanvas = document.getElementById("attention-mini-canvas");
    if (attCanvas) OceanHeatmap.drawGridToCanvas(attCanvas, attention, { min: 0, max: 1 });
  }
}
