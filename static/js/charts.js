/* ═══════════════════════════════════════════════════════════
   charts.js — Reusable Chart.js builders (Section 18 Analytics)
   ═══════════════════════════════════════════════════════════ */

const OceanCharts = (() => {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: "#eaf6ff" } } },
    scales: {
      x: { ticks: { color: "#9db4cc" }, grid: { color: "rgba(255,255,255,0.05)" } },
      y: { ticks: { color: "#9db4cc" }, grid: { color: "rgba(255,255,255,0.05)" } },
    },
  };

  function lineChart(canvas, labels, datasets) {
    if (!canvas || typeof Chart === "undefined") return null;
    return new Chart(canvas, {
      type: "line",
      data: { labels, datasets: datasets.map((d) => ({ tension: 0.3, ...d })) },
      options: commonOptions,
    });
  }

  function barChart(canvas, labels, data, label = "", color = "#22e5ff") {
    if (!canvas || typeof Chart === "undefined") return null;
    return new Chart(canvas, {
      type: "bar",
      data: { labels, datasets: [{ label, data, backgroundColor: `${color}88`, borderColor: color, borderWidth: 1 }] },
      options: commonOptions,
    });
  }

  function horizontalBarChart(canvas, labels, data, label = "") {
    if (!canvas || typeof Chart === "undefined") return null;
    return new Chart(canvas, {
      type: "bar",
      data: { labels, datasets: [{ label, data, backgroundColor: "rgba(34,229,255,0.5)", borderColor: "#22e5ff" }] },
      options: { ...commonOptions, indexAxis: "y" },
    });
  }

  function radarChart(canvas, labels, data, label = "") {
    if (!canvas || typeof Chart === "undefined") return null;
    return new Chart(canvas, {
      type: "radar",
      data: { labels, datasets: [{ label, data, backgroundColor: "rgba(34,229,255,0.2)", borderColor: "#22e5ff" }] },
      options: {
        responsive: true, maintainAspectRatio: false,
        scales: { r: { angleLines: { color: "rgba(255,255,255,0.1)" }, grid: { color: "rgba(255,255,255,0.1)" }, ticks: { color: "#9db4cc", backdropColor: "transparent" }, pointLabels: { color: "#eaf6ff" } } },
        plugins: { legend: { labels: { color: "#eaf6ff" } } },
      },
    });
  }

  function heatmapMatrix(container, matrix, labels) {
    if (!container || !matrix) return;
    container.innerHTML = "";
    const table = document.createElement("table");
    table.className = "data-table";
    const thead = document.createElement("tr");
    thead.innerHTML = `<th></th>${labels.map((l) => `<th>${l}</th>`).join("")}`;
    table.appendChild(thead);
    matrix.forEach((row, i) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<th>${labels[i]}</th>` + row.map((v) => {
        const pct = Math.abs(v);
        const bg = v >= 0 ? `rgba(34,229,255,${pct * 0.6})` : `rgba(255,59,92,${pct * 0.6})`;
        return `<td style="background:${bg}">${v.toFixed(2)}</td>`;
      }).join("");
      table.appendChild(tr);
    });
    container.appendChild(table);
  }

  return { lineChart, barChart, horizontalBarChart, radarChart, heatmapMatrix };
})();
