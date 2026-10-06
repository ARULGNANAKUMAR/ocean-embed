/* ═══════════════════════════════════════════════════════════
   earth.js companion — home page uses main.js only.
   This file (weather.js) provides the animated wave background
   and live Indian Ocean status widget for the home page (Section 1).
   ═══════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {
  initWaveCanvas();
  initAIGreeting();
  initOceanStatus();
});

function initWaveCanvas() {
  const canvas = document.getElementById("wave-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let w, h, t = 0;

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = 220;
  }
  window.addEventListener("resize", resize);
  resize();

  const layers = [
    { amp: 14, freq: 0.008, speed: 0.015, color: "rgba(10,90,168,0.35)", yOff: 0.55 },
    { amp: 20, freq: 0.006, speed: 0.010, color: "rgba(34,229,255,0.18)", yOff: 0.68 },
    { amp: 10, freq: 0.012, speed: 0.022, color: "rgba(10,132,255,0.28)", yOff: 0.8 },
  ];

  function frame() {
    ctx.clearRect(0, 0, w, h);
    layers.forEach((layer) => {
      ctx.beginPath();
      ctx.moveTo(0, h);
      for (let x = 0; x <= w; x += 6) {
        const y = h * layer.yOff + Math.sin(x * layer.freq + t * layer.speed) * layer.amp;
        ctx.lineTo(x, y);
      }
      ctx.lineTo(w, h);
      ctx.closePath();
      ctx.fillStyle = layer.color;
      ctx.fill();
    });
    t += 1;
    requestAnimationFrame(frame);
  }
  frame();
}

function initAIGreeting() {
  const el = document.getElementById("ai-greeting");
  if (!el) return;
  const hour = new Date().getHours();
  let greeting = "Good evening, Commander";
  if (hour < 12) greeting = "Good morning, Commander";
  else if (hour < 18) greeting = "Good afternoon, Commander";
  el.textContent = `${greeting} — Ocean Intelligence systems online.`;
}

async function initOceanStatus() {
  const el = document.getElementById("live-ocean-status");
  if (!el) return;
  try {
    const status = await OceanApp.api("/api/status");
    const hw = await OceanApp.api("/api/heatwave").catch(() => ({ total: 0 }));
    el.innerHTML = `
      <span class="badge ${hw.total > 0 ? "badge-high" : "badge-low"}">
        ${hw.total > 0 ? `${hw.total} ACTIVE HEATWAVE ZONES` : "OCEAN STABLE"}
      </span>
      <span style="margin-left:12px;color:var(--text-dim2)">Model: ${status.model}</span>
    `;
  } catch (e) {
    el.innerHTML = `<span class="badge badge-medium">SYSTEMS INITIALIZING</span>`;
  }
}
