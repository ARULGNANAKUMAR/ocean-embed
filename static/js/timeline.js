/* ═══════════════════════════════════════════════════════════
   heatmap.js is color utils; THIS file (renamed logically)
   drives the Marine Heatwave Center page (Sections 13, 14).
   File kept name-consistent with template: heatwave is loaded
   inline via dashboard-style widget calls below.
   ═══════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", async () => {
  await loadHeatwaveHotspots();
  await loadImpactCards();
});

async function loadHeatwaveHotspots() {
  const listEl = document.getElementById("hotspot-list");
  const countEl = document.getElementById("heatwave-total-count");
  if (!listEl) return;
  listEl.innerHTML = `<div class="spinner"></div>`;
  try {
    const data = await OceanApp.api("/api/heatwave");
    const hotspots = data.hotspots || [];
    if (countEl) countEl.textContent = data.total;

    if (!hotspots.length) {
      listEl.innerHTML = `<div class="glass-card">No active marine heatwave hotspots detected.</div>`;
      return;
    }

    listEl.innerHTML = hotspots.map((h, i) => `
      <div class="hotspot-row sev-${h.severity}" data-idx="${i}">
        <div>
          <strong>${h.latitude.toFixed(2)}°N, ${h.longitude.toFixed(2)}°E</strong>
          <div style="font-size:0.72rem;color:var(--text-dim2)">
            Anomaly +${h.sst_anomaly}°C · Depth ${h.affected_depth_m}m · Duration ${h.expected_duration_days}d
          </div>
        </div>
        <span class="badge badge-${h.severity.toLowerCase()}">${h.severity}</span>
      </div>
    `).join("");

    listEl.querySelectorAll(".hotspot-row").forEach((row) => {
      row.addEventListener("click", () => showHotspotDetail(hotspots[parseInt(row.dataset.idx)]));
    });

    renderHeatwaveTimeline(hotspots);
    renderSeverityBreakdown(hotspots);
  } catch (e) {
    listEl.innerHTML = `<div class="glass-card">Heatwave data unavailable.</div>`;
  }
}

function showHotspotDetail(h) {
  let modal = document.getElementById("hotspot-modal");
  if (!modal) {
    modal = document.createElement("div");
    modal.id = "hotspot-modal";
    modal.className = "modal-backdrop";
    document.body.appendChild(modal);
  }
  modal.innerHTML = `
    <div class="modal-box glass-card relative">
      <button class="modal-close" onclick="document.getElementById('hotspot-modal').classList.remove('open')">×</button>
      <div class="card-title">Hotspot Detail</div>
      <div class="gi-row"><span class="gi-label">Location</span><span class="gi-value">${h.latitude.toFixed(2)}°N, ${h.longitude.toFixed(2)}°E</span></div>
      <div class="gi-row"><span class="gi-label">Severity</span><span class="gi-value badge badge-${h.severity.toLowerCase()}">${h.severity}</span></div>
      <div class="gi-row"><span class="gi-label">SST Anomaly</span><span class="gi-value">+${h.sst_anomaly}°C</span></div>
      <div class="gi-row"><span class="gi-label">Probability</span><span class="gi-value">${(h.probability * 100).toFixed(0)}%</span></div>
      <div class="gi-row"><span class="gi-label">Expected Start</span><span class="gi-value">${h.expected_start}</span></div>
      <div class="gi-row"><span class="gi-label">Duration</span><span class="gi-value">${h.expected_duration_days} days</span></div>
      <div class="gi-row"><span class="gi-label">Affected Depth</span><span class="gi-value">${h.affected_depth_m}m</span></div>
      <div class="gi-row"><span class="gi-label">Affected SST</span><span class="gi-value">${h.affected_sst}°C</span></div>
      <div class="gi-row"><span class="gi-label">Subsurface Layers</span><span class="gi-value">${h.subsurface_layers}</span></div>
    </div>
  `;
  requestAnimationFrame(() => modal.classList.add("open"));
  modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.remove("open"); });
}

function renderHeatwaveTimeline(hotspots) {
  const canvas = document.getElementById("heatwave-timeline-chart");
  if (!canvas || typeof Chart === "undefined") return;
  const sorted = [...hotspots].sort((a, b) => new Date(a.expected_start) - new Date(b.expected_start));
  OceanCharts.lineChart(canvas,
    sorted.map((h) => h.expected_start),
    [{ label: "SST Anomaly (°C)", data: sorted.map((h) => h.sst_anomaly), borderColor: "#ff3b5c", backgroundColor: "rgba(255,59,92,0.15)", fill: true }]
  );
}

function renderSeverityBreakdown(hotspots) {
  const canvas = document.getElementById("severity-breakdown-chart");
  if (!canvas || typeof Chart === "undefined") return;
  const counts = { LOW: 0, MEDIUM: 0, HIGH: 0, EXTREME: 0 };
  hotspots.forEach((h) => { counts[h.severity] = (counts[h.severity] || 0) + 1; });
  new Chart(canvas, {
    type: "doughnut",
    data: {
      labels: Object.keys(counts),
      datasets: [{
        data: Object.values(counts),
        backgroundColor: ["#7ee787", "#ffb020", "#ff3b5c", "#ff0844"],
        borderColor: "#050e1a", borderWidth: 2,
      }],
    },
    options: { plugins: { legend: { labels: { color: "#eaf6ff" } } } },
  });
}

// ── Impact Prediction cards (Section 14) ──
async function loadImpactCards() {
  const container = document.getElementById("impact-cards-container");
  if (!container) return;
  try {
    const data = await OceanApp.api("/api/heatwave/forecast");
    const impacts = data.impacts || [];
    if (!impacts.length) {
      container.innerHTML = `<div class="glass-card">No impact predictions at this time.</div>`;
      return;
    }
    container.innerHTML = impacts.map((entry) => `
      <div class="glass-card" style="margin-bottom:16px">
        <div class="card-title">Hotspot @ ${entry.hotspot.latitude.toFixed(2)}°N, ${entry.hotspot.longitude.toFixed(2)}°E</div>
        ${(entry.impacts || []).map((imp) => `
          <div class="impact-card animated-border">
            <div class="impact-title">${imp.impact}</div>
            <div class="impact-severity-bar"><div class="impact-severity-fill" style="width:${(imp.severity * 100).toFixed(0)}%"></div></div>
            <div class="impact-reason">${imp.reason}</div>
            <div class="impact-action">→ ${imp.preventive_action}</div>
          </div>
        `).join("") || `<div style="color:var(--text-dim2);font-size:0.8rem">No specific impacts identified.</div>`}
      </div>
    `).join("");
  } catch (e) {
    container.innerHTML = `<div class="glass-card">Impact data unavailable.</div>`;
  }
}
