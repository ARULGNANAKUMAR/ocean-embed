/* ═══════════════════════════════════════════════════════════
   main.js — App shell: starfield, live clock, toasts, socket,
   ripple effect, loading overlay. Loaded on every page.
   ═══════════════════════════════════════════════════════════ */

const OceanApp = (() => {
  let socket = null;

  // ── Starfield background (canvas 2D — lightweight, all pages) ──
  function initStarfield() {
    const canvas = document.getElementById("bg-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let stars = [];
    let w, h;

    function resize() {
      w = canvas.width = window.innerWidth;
      h = canvas.height = window.innerHeight;
    }
    window.addEventListener("resize", resize);
    resize();

    const STAR_COUNT = Math.min(180, Math.floor((w * h) / 9000));
    for (let i = 0; i < STAR_COUNT; i++) {
      stars.push({
        x: Math.random() * w,
        y: Math.random() * h,
        r: Math.random() * 1.4 + 0.2,
        speed: Math.random() * 0.15 + 0.02,
        twinkle: Math.random() * Math.PI * 2,
      });
    }

    function frame() {
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "rgba(3,7,17,1)";
      for (const s of stars) {
        s.twinkle += 0.02;
        const alpha = 0.4 + 0.6 * Math.abs(Math.sin(s.twinkle));
        ctx.beginPath();
        ctx.fillStyle = `rgba(180,220,255,${alpha})`;
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
        s.y += s.speed;
        if (s.y > h) { s.y = 0; s.x = Math.random() * w; }
      }
      requestAnimationFrame(frame);
    }
    frame();
  }

  // ── Live UTC / IST clock ──
  function initClock() {
    const dateEl = document.getElementById("live-date");
    const utcEl  = document.getElementById("live-utc");
    if (!dateEl && !utcEl) return;
    function tick() {
      const now = new Date();
      if (dateEl) {
        dateEl.textContent = now.toLocaleDateString("en-IN", {
          weekday: "short", year: "numeric", month: "short", day: "numeric",
        });
      }
      if (utcEl) {
        utcEl.textContent = now.toUTCString().split(" ")[4] + " UTC";
      }
    }
    tick();
    setInterval(tick, 1000);
  }

  // ── Toast notification system ──
  function toast(message, level = "info") {
    let stack = document.getElementById("toast-stack");
    if (!stack) {
      stack = document.createElement("div");
      stack.id = "toast-stack";
      document.body.appendChild(stack);
    }
    const el = document.createElement("div");
    el.className = `toast ${level}`;
    el.textContent = message;
    stack.appendChild(el);
    setTimeout(() => {
      el.classList.add("closing");
      setTimeout(() => el.remove(), 320);
    }, 4200);
  }

  // ── SocketIO connection for live notifications (Section 24) ──
  function initSocket() {
    if (typeof io === "undefined") return null;
    socket = io();
    socket.on("notification", (data) => {
      toast(data.message, data.level || "info");
    });
    return socket;
  }

  // ── Ripple click effect (Section 27) ──
  function initRipples() {
    document.addEventListener("click", (e) => {
      const target = e.target.closest(".btn-neon, .dock-btn, .glass-card, .sidebar-nav-item");
      if (!target) return;
      const rect = target.getBoundingClientRect();
      const ripple = document.createElement("span");
      const size = Math.max(rect.width, rect.height);
      ripple.className = "ripple";
      ripple.style.width = ripple.style.height = `${size}px`;
      ripple.style.left = `${e.clientX - rect.left - size / 2}px`;
      ripple.style.top = `${e.clientY - rect.top - size / 2}px`;
      target.style.position = target.style.position || "relative";
      target.style.overflow = "hidden";
      target.appendChild(ripple);
      setTimeout(() => ripple.remove(), 700);
    });
  }

  // ── Loading overlay control ──
  function showLoading(text = "Initializing Ocean Intelligence Platform") {
    let overlay = document.getElementById("loading-overlay");
    if (!overlay) {
      overlay = document.createElement("div");
      overlay.id = "loading-overlay";
      overlay.innerHTML = `
        <div class="spinner"></div>
        <div class="loading-text">${text}</div>
        <div class="loading-bar"><div class="loading-bar-fill" id="loading-bar-fill"></div></div>
      `;
      document.body.appendChild(overlay);
    }
    overlay.classList.remove("hidden");
  }

  function setLoadingProgress(pct) {
    const fill = document.getElementById("loading-bar-fill");
    if (fill) fill.style.width = `${Math.min(100, Math.max(0, pct))}%`;
  }

  function hideLoading() {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) overlay.classList.add("hidden");
  }

  // ── Fetch helper ──
  async function api(path, opts = {}) {
    try {
      const res = await fetch(path, opts);
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.error || `HTTP ${res.status}`);
      }
      return await res.json();
    } catch (err) {
      console.error(`API error [${path}]:`, err.message);
      throw err;
    }
  }

  // ── Copy to clipboard helper (lat/lon click-to-copy — Section 4) ──
  function copyToClipboard(text) {
    navigator.clipboard?.writeText(text).then(() => {
      toast(`Copied: ${text}`, "success");
    }).catch(() => {
      toast("Copy failed", "error");
    });
  }

  // ── Init on DOM ready ──
  document.addEventListener("DOMContentLoaded", () => {
    initStarfield();
    initClock();
    initRipples();
    initSocket();
  });

  return { toast, api, copyToClipboard, showLoading, hideLoading, setLoadingProgress, getSocket: () => socket };
})();
