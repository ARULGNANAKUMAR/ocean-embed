/* ═══════════════════════════════════════════════════════════
   heatmap.js — Shared color-scale utilities (Section 7)
   Cold→Blue, Moderate→Green, Warm→Yellow, Hot→Orange, Extreme→Red
   ═══════════════════════════════════════════════════════════ */

const OceanHeatmap = (() => {
  const STOPS = [
    [0.00, [28, 79, 216]],   // cold - blue
    [0.30, [46, 230, 168]],  // moderate - green
    [0.55, [255, 228, 94]],  // warm - yellow
    [0.75, [255, 138, 32]],  // hot - orange
    [1.00, [255, 8, 68]],    // extreme - red
  ];

  function tempToColor(t, min = 5, max = 32) {
    const pct = Math.max(0, Math.min(1, (t - min) / (max - min)));
    for (let i = 0; i < STOPS.length - 1; i++) {
      const [p0, c0] = STOPS[i], [p1, c1] = STOPS[i + 1];
      if (pct >= p0 && pct <= p1) {
        const t2 = (pct - p0) / (p1 - p0);
        const r = Math.round(c0[0] + (c1[0] - c0[0]) * t2);
        const g = Math.round(c0[1] + (c1[1] - c0[1]) * t2);
        const b = Math.round(c0[2] + (c1[2] - c0[2]) * t2);
        return `rgb(${r},${g},${b})`;
      }
    }
    return "rgb(200,200,200)";
  }

  function severityColor(sev) {
    return { LOW: "#7ee787", MEDIUM: "#ffb020", HIGH: "#ff3b5c", EXTREME: "#ff0844" }[sev] || "#5d7c90";
  }

  function confidenceColor(c) {
    // 0-1 → red-yellow-green
    const pct = Math.max(0, Math.min(1, c));
    if (pct < 0.5) return `rgb(${255}, ${Math.round(pct * 2 * 176)}, 32)`;
    return `rgb(${Math.round((1 - pct) * 2 * 255)}, 200, 80)`;
  }

  /** Draw a 2D grid array onto a canvas as a heatmap. */
  function drawGridToCanvas(canvas, grid, opts = {}) {
    if (!canvas || !grid || !grid.length) return;
    const ctx = canvas.getContext("2d");
    const rows = grid.length, cols = grid[0].length;
    canvas.width = cols; canvas.height = rows;
    const imgData = ctx.createImageData(cols, rows);
    const min = opts.min ?? 5, max = opts.max ?? 32;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = grid[r][c];
        const idx = (r * cols + c) * 4;
        if (v == null || !isFinite(v)) {
          imgData.data[idx + 3] = 0; // transparent
          continue;
        }
        const [rr, gg, bb] = hexToRgbTuple(tempToColor(v, min, max));
        imgData.data[idx] = rr;
        imgData.data[idx + 1] = gg;
        imgData.data[idx + 2] = bb;
        imgData.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  function hexToRgbTuple(rgbStr) {
    const m = rgbStr.match(/\d+/g);
    return m ? m.map(Number) : [128, 128, 128];
  }

  return { tempToColor, severityColor, confidenceColor, drawGridToCanvas };
})();
