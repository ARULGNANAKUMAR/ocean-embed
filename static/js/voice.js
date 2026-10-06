/* ═══════════════════════════════════════════════════════════
   voice.js — Voice Assistant Backend UI (Section 19)
   Text command input, mic animation, auto-navigation.
   ═══════════════════════════════════════════════════════════ */

document.addEventListener("DOMContentLoaded", () => {
  initVoiceForm();
  initMicButton();
  initExampleChips();
});

function initVoiceForm() {
  const form = document.getElementById("voice-query-form");
  const input = document.getElementById("voice-query-input");
  if (!form) return;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;
    await submitVoiceQuery(query);
  });
}

async function submitVoiceQuery(query) {
  const responseEl = document.getElementById("voice-response");
  appendVoiceMessage(query, "user");
  responseEl && (responseEl.innerHTML += `<div class="spinner" style="margin:10px 0"></div>`);

  try {
    const data = await OceanApp.api(`/api/voice/query?query=${encodeURIComponent(query)}`, { method: "POST" });
    document.querySelector("#voice-response .spinner")?.remove();
    renderVoiceResult(data);
    maybeAutoNavigate(data);
  } catch (e) {
    document.querySelector("#voice-response .spinner")?.remove();
    appendVoiceMessage("Sorry, I couldn't process that request. The prediction model may still be loading.", "assistant");
  }
}

function appendVoiceMessage(text, role) {
  const responseEl = document.getElementById("voice-response");
  if (!responseEl) return;
  const bubble = document.createElement("div");
  bubble.className = `voice-bubble voice-bubble-${role} fade-in`;
  bubble.textContent = text;
  responseEl.appendChild(bubble);
  responseEl.scrollTop = responseEl.scrollHeight;
}

function renderVoiceResult(data) {
  const responseEl = document.getElementById("voice-response");
  if (!responseEl) return;
  const bubble = document.createElement("div");
  bubble.className = "voice-bubble voice-bubble-assistant fade-in glass-card";

  let html = `<div style="font-size:0.72rem;color:var(--text-dim2);text-transform:uppercase;margin-bottom:6px">Intent: ${data.intent}</div>`;

  if (data.intent === "heatwave" && data.heatwaves) {
    html += data.heatwaves.map((h) => h.status
      ? `<p>${h.status}</p>`
      : `<div class="hotspot-row sev-${h.severity}">
           <span>${h.latitude.toFixed(2)}°N, ${h.longitude.toFixed(2)}°E</span>
           <span class="badge badge-${h.severity.toLowerCase()}">${h.severity}</span>
         </div>`
    ).join("");
  } else if (data.intent === "temperature" && data.temperature_timeseries) {
    html += `<canvas id="voice-temp-chart" height="140"></canvas>`;
  } else if (data.message) {
    html += `<p>${data.message}</p>`;
  } else {
    html += `<p>Query processed for ${data.location ? data.location.name : "the requested region"}.</p>`;
  }

  bubble.innerHTML = html;
  responseEl.appendChild(bubble);
  responseEl.scrollTop = responseEl.scrollHeight;

  if (data.intent === "temperature" && data.temperature_timeseries) {
    setTimeout(() => {
      const canvas = document.getElementById("voice-temp-chart");
      if (canvas) OceanCharts.lineChart(canvas,
        data.temperature_timeseries.map((_, i) => `Day ${i + 1}`),
        [{ label: "Temperature (°C)", data: data.temperature_timeseries, borderColor: "#22e5ff", backgroundColor: "rgba(34,229,255,0.15)", fill: true }]
      );
    }, 50);
  }
}

function maybeAutoNavigate(data) {
  // Auto-navigate assistant (Section 19: "Assistant automatically navigates")
  const navHint = document.getElementById("voice-nav-hint");
  let target = null;
  if (data.intent === "heatwave") target = "/heatwave";
  else if (data.intent === "forecast") target = "/prediction";
  if (target && navHint) {
    navHint.innerHTML = `<a href="${target}" class="btn-neon">Open ${data.intent === "heatwave" ? "Heatwave Center" : "Prediction Dashboard"} →</a>`;
  }
}

function initMicButton() {
  const micBtn = document.getElementById("mic-button");
  const input = document.getElementById("voice-query-input");
  if (!micBtn) return;

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    micBtn.title = "Speech recognition not supported — use text input";
    return;
  }
  const recognition = new SpeechRecognition();
  recognition.lang = "en-IN";
  recognition.continuous = false;

  micBtn.addEventListener("click", () => {
    micBtn.classList.add("mic-listening");
    recognition.start();
  });
  recognition.onresult = (e) => {
    const transcript = e.results[0][0].transcript;
    if (input) input.value = transcript;
    micBtn.classList.remove("mic-listening");
  };
  recognition.onerror = () => micBtn.classList.remove("mic-listening");
  recognition.onend = () => micBtn.classList.remove("mic-listening");
}

function initExampleChips() {
  document.querySelectorAll(".voice-example-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const input = document.getElementById("voice-query-input");
      if (input) { input.value = chip.textContent.trim(); }
      submitVoiceQuery(chip.textContent.trim());
    });
  });
}
