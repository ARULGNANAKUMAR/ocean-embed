/* ═══════════════════════════════════════════════════════════
   dashboard.js — Prediction Dashboard (Section 11)
   Widgets: SST, prediction, 15-depth chart, confidence,
   heatwave alert, wind/current speed, ocean health score.
   ═══════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", async () => {
  await Promise.all([
    loadSummaryWidgets(),
    loadDepthChart(),
    loadForecastCards(),
    loadOceanHealthScore(),
  ]);
});

async function loadSummaryWidgets() {
  try {
    const [latest, conf, heatwave, surfaceLayer] = await Promise.all([
      OceanApp.api("/api/prediction/latest"),
      OceanApp.api("/api/confidence"),
      OceanApp.api("/api/heatwave"),
      OceanApp.api("/api/map/layer?layer=sst&time_idx=-1").catch(() => null),
    ]);

    setWidget("widget-sst", latest.surface_temp_mean, "°C");
    setWidget("widget-prediction", latest.surface_temp_max, "°C");
    setWidget("widget-confidence", conf.overall_confidence != null ? (conf.overall_confidence * 100).toFixed(0) : "-", "%");
    setWidget("widget-heatwave-count", heatwave.total, "zones");

    const alertEl = document.getElementById("widget-heatwave-alert");
    if (alertEl) {
      const worst = heatwave.hotspots?.[0];
      alertEl.textContent = worst ? worst.severity : "NONE";
      alertEl.className = `card-value badge-${(worst?.severity || "low").toLowerCase()}`;
    }
  } catch (e) {
    console.warn("Dashboard widgets failed to load", e);
  }
}

function setWidget(id, value, unit = "") {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = `${value != null && value !== "-" ? Number(value).toFixed?.(2) ?? value : "—"}<span class="card-unit">${unit}</span>`;
}

async function loadDepthChart() {
  const canvas = document.getElementById("depth-temp-chart");
  if (!canvas || typeof Chart === "undefined") return;
  try {
    const data = await OceanApp.api("/api/prediction/date?days_ahead=0");
    const depths = data.depths || [];
    const profile = data.depth_profile || [];
    new Chart(canvas, {
      type: "line",
      data: {
        labels: depths,
        datasets: [{
          label: "Temperature (°C)",
          data: profile,
          borderColor: "#22e5ff",
          backgroundColor: "rgba(34,229,255,0.12)",
          fill: true,
          tension: 0.35,
          pointBackgroundColor: "#22e5ff",
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "Depth (m)", color: "#9db4cc" }, ticks: { color: "#9db4cc" }, grid: { color: "rgba(255,255,255,0.05)" } },
          y: { ticks: { color: "#9db4cc" }, grid: { color: "rgba(255,255,255,0.05)" } },
        },
        plugins: { legend: { labels: { color: "#eaf6ff" } } },
      },
    });
  } catch (e) {
    console.warn("Depth chart failed", e);
  }
}

async function loadForecastCards() {
  const row = document.getElementById("forecast-row");
  if (!row) return;
  const horizons = [0, 7, 15, 30];
  row.innerHTML = horizons.map((d) => `
    <div class="glass-card forecast-card" data-days="${d}">
      <div class="fc-day">${d === 0 ? "Today" : `+${d} Days`}</div>
      <div class="fc-temp skeleton" style="height:32px;width:60px;margin:0 auto"></div>
      <div class="fc-conf">Loading…</div>
    </div>
  `).join("");

  for (const d of horizons) {
    try {
      const data = d === 0
        ? await OceanApp.api("/api/prediction/date?days_ahead=0")
        : await OceanApp.api(`/api/prediction/date?days_ahead=${d}`);
      const card = row.querySelector(`[data-days="${d}"]`);
      if (!card) continue;
      let tempVal = "—";
      if (d === 0 && data.depth_profile) tempVal = data.depth_profile[0]?.toFixed(1);
      else if (data.temperature_field) {
        const flat = data.temperature_field.flat(2).filter((v) => v != null && isFinite(v));
        tempVal = flat.length ? (flat.reduce((a, b) => a + b, 0) / flat.length).toFixed(1) : "—";
      }
      card.querySelector(".fc-temp").outerHTML = `<div class="fc-temp">${tempVal}°C</div>`;
      card.querySelector(".fc-conf").textContent = data.confidence
        ? `${(data.confidence * 100).toFixed(0)}% confidence` : "—";
    } catch (e) { /* leave skeleton */ }
  }
}

async function loadOceanHealthScore() {
  const el = document.getElementById("ocean-health-score");
  if (!el) return;
  try {
    const [conf, heatwave] = await Promise.all([
      OceanApp.api("/api/confidence"),
      OceanApp.api("/api/heatwave"),
    ]);
    const confScore = (conf.overall_confidence || 0.5) * 100;
    const heatwavePenalty = Math.min(40, (heatwave.total || 0) * 4);
    const score = Math.max(0, Math.round(confScore - heatwavePenalty));
    el.textContent = score;
    el.style.color = score > 70 ? "#2ee6a8" : score > 40 ? "#ffb020" : "#ff3b5c";
  } catch (e) {
    el.textContent = "—";
  }
}
