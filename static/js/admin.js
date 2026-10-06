/* ═══════════════════════════════════════════════════════════
   admin.js — Admin Dashboard Backend controller (Section 20)
   Login, dataset upload (multi-type), retrain, logs, history.
   ═══════════════════════════════════════════════════════════ */

const ADMIN_TOKEN_KEY = "sih_admin_token";

document.addEventListener("DOMContentLoaded", () => {
  initAdminLogin();
  if (getAdminToken()) {
    showAdminPanel();
  }
});

function getAdminToken() {
  return sessionStorage.getItem(ADMIN_TOKEN_KEY);
}

async function sha256Hex(str) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(str));
  return Array.from(new Uint8Array(buf)).map((b) => b.toString(16).padStart(2, "0")).join("");
}

function initAdminLogin() {
  const form = document.getElementById("admin-login-form");
  if (!form) return;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const pw = document.getElementById("admin-password").value;
    const token = await sha256Hex(pw);
    try {
      const res = await fetch(`/api/admin/history?token=${token}`);
      if (res.status === 403) {
        OceanApp.toast("Invalid admin credentials", "error");
        return;
      }
      sessionStorage.setItem(ADMIN_TOKEN_KEY, token);
      OceanApp.toast("Admin authenticated", "success");
      showAdminPanel();
    } catch (err) {
      OceanApp.toast("Login failed — server unreachable", "error");
    }
  });
}

function showAdminPanel() {
  document.getElementById("admin-login-screen")?.classList.add("hidden-panel");
  document.getElementById("admin-panel")?.classList.remove("hidden-panel");
  loadAdminStats();
  loadAdminHistory();
  initUploadZones();
  initRetrainButton();
}

async function loadAdminStats() {
  const token = getAdminToken();
  try {
    const data = await OceanApp.api(`/api/admin/history?token=${token}`);
    const stats = data.statistics || {};
    document.getElementById("admin-stat-datasets").textContent = stats.datasets_count ?? "-";
    document.getElementById("admin-stat-predictions").textContent = stats.predictions_count ?? "-";
    document.getElementById("admin-stat-heatwaves").textContent = stats.heatwave_history_count ?? "-";
    document.getElementById("admin-stat-uploads").textContent = stats.admin_uploads_count ?? "-";

    renderHistoryTable("admin-logs-table", data.logs, ["action", "detail", "performed_at"]);
    renderHistoryTable("admin-predictions-table", data.predictions,
      ["prediction_date", "forecast_days", "model_version", "rmse"]);
    renderHistoryTable("admin-training-table", data.training,
      ["epochs_run", "best_val_loss", "started_at", "finished_at"]);
  } catch (e) {
    OceanApp.toast("Failed to load admin statistics", "error");
  }
}

function renderHistoryTable(tableId, rows, cols) {
  const table = document.getElementById(tableId);
  if (!table) return;
  if (!rows || !rows.length) {
    table.innerHTML = `<tr><td colspan="${cols.length}" style="text-align:center;color:var(--text-dim2)">No records</td></tr>`;
    return;
  }
  const thead = `<tr>${cols.map((c) => `<th>${c.replace(/_/g, " ")}</th>`).join("")}</tr>`;
  const tbody = rows.slice(0, 20).map((r) =>
    `<tr>${cols.map((c) => `<td>${r[c] ?? "-"}</td>`).join("")}</tr>`
  ).join("");
  table.innerHTML = thead + tbody;
}

async function loadAdminHistory() {
  const token = getAdminToken();
  try {
    const data = await OceanApp.api(`/api/admin/models?token=${token}`);
    renderHistoryTable("admin-models-table", data.models,
      ["version_name", "val_loss", "epochs", "created_at"]);
  } catch (e) { /* silent */ }
}

// ── Dataset upload zones (Section 20: GLORYS/SST/SSS/Wind/SLA/ARGO) ──
const DATASET_TYPES = [
  { id: "glorys", label: "GLORYS", type: "TARGET" },
  { id: "sst",    label: "SST",    type: "CORE_INPUT" },
  { id: "sss",    label: "SSS",    type: "CORE_INPUT" },
  { id: "wind",   label: "Wind",   type: "CORE_INPUT" },
  { id: "sla",    label: "SLA",    type: "CORE_INPUT" },
  { id: "argo",   label: "ARGO",   type: "VALIDATION" },
];

function initUploadZones() {
  const container = document.getElementById("upload-zones-container");
  if (!container || container.dataset.init) return;
  container.dataset.init = "true";

  container.innerHTML = DATASET_TYPES.map((d) => `
    <div class="glass-card">
      <div class="card-title">${d.label} Upload</div>
      <div class="upload-zone" id="zone-${d.id}">
        <input type="file" id="file-${d.id}" accept=".nc,.nc4,.cdf,.csv" />
        <p style="color:var(--text-dim2);font-size:0.8rem">Drag file or click to upload ${d.label}</p>
      </div>
    </div>
  `).join("");

  DATASET_TYPES.forEach((d) => {
    const zone = document.getElementById(`zone-${d.id}`);
    const input = document.getElementById(`file-${d.id}`);
    zone.addEventListener("click", () => input.click());
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("drag-over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("drag-over");
      if (e.dataTransfer.files.length) uploadDataset(e.dataTransfer.files[0], d.type, d.label);
    });
    input.addEventListener("change", () => {
      if (input.files.length) uploadDataset(input.files[0], d.type, d.label);
    });
  });
}

async function uploadDataset(file, datasetType, label) {
  const token = getAdminToken();
  const formData = new FormData();
  formData.append("file", file);
  try {
    OceanApp.toast(`Uploading ${label}: ${file.name}…`, "info");
    const res = await fetch(`/api/admin/upload?token=${token}&dataset_type=${datasetType}`, {
      method: "POST", body: formData,
    });
    const data = await res.json();
    if (res.ok) {
      OceanApp.toast(`${label} uploaded successfully`, "success");
      loadAdminStats();
    } else {
      OceanApp.toast(data.error || "Upload failed", "error");
    }
  } catch (e) {
    OceanApp.toast("Upload failed — network error", "error");
  }
}

// ── Retrain trigger ──
function initRetrainButton() {
  const btn = document.getElementById("btn-retrain-model");
  if (!btn || btn.dataset.init) return;
  btn.dataset.init = "true";
  btn.addEventListener("click", async () => {
    const token = getAdminToken();
    const progressBar = document.getElementById("retrain-progress");
    if (progressBar) progressBar.style.width = "15%";
    try {
      const res = await fetch(`/api/admin/retrain?token=${token}`, { method: "POST" });
      const data = await res.json();
      if (res.ok) {
        OceanApp.toast(data.message || "Retrain queued", "success");
        if (progressBar) progressBar.style.width = "100%";
      } else {
        OceanApp.toast(data.error || "Retrain failed", "error");
      }
    } catch (e) {
      OceanApp.toast("Retrain request failed", "error");
    }
  });
}
