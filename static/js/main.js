/* =====================================================
 * Medicine ChatBot — Training Console (Frontend)
 * Cleaned + optimized main.js
 * - Single, reliable overlay sequence (one line at a time)
 * - Old line fades out & is removed (requires #processingLines + CSS .hide)
 * - Overlay always stops + hides on success/error
 * - Faster timings + cleaner UI helpers
 * - Removed unused chat-bubble code
 * ===================================================== */
"use strict";

/* =========================
   1) BACKEND CONFIG
   ========================= */

const API_BASE = ""; // keep "" when same origin
const INGEST_ENDPOINT = `${API_BASE}/api/train`;
const CRAWL_ENDPOINT = `${API_BASE}/api/crawl`;

/* =========================
   2) DOM HELPERS
   ========================= */

const $ = (sel) => document.querySelector(sel);
const on = (el, evt, fn, opts) => el && el.addEventListener(evt, fn, opts);

/* =========================
   3) ELEMENT REFERENCES
   ========================= */

// Top status area
const statusDot = $("#statusDot");
const statusText = $("#statusText");
const phaseText = $("#phaseText");
const progressBar = $("#progressBar");
const progressPct = $("#progressPct");

// Metrics
const mFiles = $("#mFiles");
const mChunks = $("#mChunks");
const mSkipped = $("#mSkipped");

// Endpoint
const endpointText = $("#endpointText");

// File selection area
const dropzone = $("#dropzone");
const fileInput = $("#fileInput");
const fileList = $("#fileList");
const fileCount = $("#fileCount");

// Buttons
const uploadBtn = $("#uploadBtn");
const clearBtn = $("#clearBtn");
const copyLogBtn = $("#copyLogBtn");

// NEW: dropdown references
const trainingTypeDropdown = $("#trainingTypeDropdown");
const trainingTypeItems = document.querySelectorAll("[data-train-type]");

// NEW: inline error message under Train button
const trainTypeError = $("#trainTypeError");

// Console log area
const consoleLog = $("#consoleLog");

// Overlay
const processingOverlay = $("#processingOverlay");
const lottieOverlayEl = $("#lottieOverlayAnim");

// New overlay container (preferred)
const processingLines = $("#processingLines");

// Legacy overlay elements (if your HTML still has them)
const processingTitle = $("#processingTitle");
const processingSub = $("#processingSub");
const dots = $("#dots");

// Particles
const particlesCanvas = $("#aiParticles");

/* =========================
   4) APP STATE
   ========================= */

let selectedFiles = [];
let eventsBound = false;

// Simulated progress
let progressTimer = null;

// Lottie
let overlayLottie = null;
let overlayLottieLoaded = false;

// Overlay sequencing
let overlayRunId = 0;

// NEW: training type state
let selectedTrainType = "";

/* =========================
   5) SMALL UTILITIES
   ========================= */

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function nowTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function bytesToSize(bytes) {
  const n = Number(bytes) || 0;
  if (n < 1024) return `${n} B`;
  const units = ["KB", "MB", "GB"];
  let v = n / 1024;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v.toFixed(v >= 10 ? 0 : 1)} ${units[i]}`;
}

function clamp(num, min, max) {
  return Math.max(min, Math.min(max, num));
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function setDisabled(disabled) {
  const d = !!disabled;
  if (uploadBtn) uploadBtn.disabled = d;
  if (clearBtn) clearBtn.disabled = d;
  if (fileInput) fileInput.disabled = d;
  if (dropzone) dropzone.setAttribute("aria-disabled", d ? "true" : "false");
}

// Dropdown label helper
function setTrainingTypeLabel(type) {
  const t = String(type || "").toLowerCase();
  const label = t === "herbal" ? "Herbal" : t === "pharma" ? "Pharma" : t === "other" ? "Other": "Select Type";

  if (trainingTypeDropdown) {
    trainingTypeDropdown.innerHTML = `<i class="bi bi-sliders"></i> ${label}`;
  }
}

// NEW: reset training type to default
function resetTrainingType() {
  selectedTrainType = "";
  setTrainingTypeLabel("");
  hideTrainTypeError();
}

// NEW: show/hide red message under Train button
function showTrainTypeError() {
  if (!trainTypeError) return;
  trainTypeError.classList.remove("d-none");
}

function hideTrainTypeError() {
  if (!trainTypeError) return;
  trainTypeError.classList.add("d-none");
}

/* =========================
   6) UI HELPERS
   ========================= */

function setDot(mode) {
  if (!statusDot) return;
  statusDot.classList.remove("online", "error");
  if (mode) statusDot.classList.add(mode);
}

function setStatus(text, mode = "online") {
  if (statusText) statusText.textContent = text ?? "";
  setDot(mode);
}

function setPhase(label) {
  if (phaseText) phaseText.textContent = label ?? "";
}

function setProgress(pct) {
  const p = clamp(Number(pct) || 0, 0, 100);
  if (progressBar) progressBar.style.width = `${p}%`;
  if (progressPct) progressPct.textContent = `${Math.round(p)}%`;
}

function setMetrics({ files = 0, chunks = 0, skipped = 0 } = {}) {
  if (mFiles) mFiles.textContent = String(files);
  if (mChunks) mChunks.textContent = String(chunks);
  if (mSkipped) mSkipped.textContent = String(skipped);
}

function log(line) {
  if (!consoleLog) return;
  const prefix = `[${nowTime()}] `;
  consoleLog.textContent += `\n${prefix}${line}`;
  consoleLog.scrollTop = consoleLog.scrollHeight;
}

function clearLog() {
  if (!consoleLog) return;
  consoleLog.textContent = "[system] Ready.";
}

function copyLogToClipboard() {
  const txt = consoleLog?.textContent || "";
  if (!txt) return;
  navigator.clipboard?.writeText(txt).then(
    () => log("Console copied to clipboard."),
    () => log("Could not copy console (clipboard blocked).")
  );
}

function updateFileCount() {
  if (!fileCount) return;
  fileCount.textContent = `${selectedFiles.length} file${selectedFiles.length === 1 ? "" : "s"}`;
}

function renderFileList() {
  if (!fileList) return;

  fileList.innerHTML = "";

  // Empty state
  if (!selectedFiles.length) {
    fileList.innerHTML = `<div class="ai-muted small">No files selected.</div>`;
    setMetrics?.({
      files: Number(mFiles?.textContent || 0),
      chunks: Number(mChunks?.textContent || 0),
      skipped: Number(mSkipped?.textContent || 0),
    });
    return;
  }

  // Render rows
  selectedFiles.forEach((f, idx) => {
    const row = document.createElement("div");
    row.className = "ai-file";

    row.innerHTML = `
      <div class="d-flex flex-column">
        <div class="ai-file-name">${escapeHtml(f.name)}</div>
        <div class="ai-file-meta">${bytesToSize(f.size)}</div>
      </div>
      <button class="ai-file-x" type="button" title="Remove" aria-label="Remove file" data-idx="${idx}">
        <i class="bi bi-x-circle"></i>
      </button>
    `;

    fileList.appendChild(row);
  });

  // Update counts
  updateFileCount();
  setMetrics?.({
    files: selectedFiles.length,
    chunks: Number(mChunks?.textContent || 0),
    skipped: Number(mSkipped?.textContent || 0),
  });

  // Remove handlers (per button)
  fileList.querySelectorAll("button.ai-file-x").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const i = Number(e.currentTarget.dataset.idx);
      if (!Number.isFinite(i)) return;

      selectedFiles.splice(i, 1);
      renderFileList();
    });
  });
}

/* =========================
   7) PROCESSING OVERLAY (ANIMATION + TEXT)
   ========================= */

// One line at a time
const OVERLAY_LINES = [
  "Okay, Uploading your Files to my workspace.. This won't take long!",
  "Now I'm reading through all the text in your documents. Fascinating stuff!",
  "Breaking down the contents into small chapters so that i can understand it better.",
  "Converting everything into a format my AI brain can process. This is where the magic happens!",
  "Building my knowledge index.. Almost there!",
  "Perfect I've learned everything from your documents. I'm ready to use this new knowledge.",
  "Thank you for these interesting new information !",
];

// SPEED SETTINGS
const LINE_TYPE_SPEED_MS = 60; // ms per char
const LINE_GAP_BASE_MS = 210; // base wait (scaled by PDF count, capped)
const LINE_FADE_MS = 350; // should match CSS transition

function initOverlayLottie() {
  if (overlayLottieLoaded) return;
  if (!lottieOverlayEl) return;
  if (typeof window.lottie === "undefined") return;

  const url = lottieOverlayEl.dataset.lottieUrl;
  if (!url) return;

  overlayLottie = window.lottie.loadAnimation({
    container: lottieOverlayEl,
    renderer: "svg",
    loop: true,
    autoplay: false,
    path: url,
  });

  overlayLottieLoaded = true;
}

function cancelOverlaySequence() {
  overlayRunId++;

  if (processingLines) processingLines.innerHTML = "";
  if (processingSub) processingSub.textContent = "";

  if (dots) dots.textContent = "";
}

function ensureLinesContainer() {
  if (processingLines) return processingLines;
  return processingSub;
}

async function typewriterLine(el, text, runId) {
  const s = String(text || "");
  el.textContent = "";

  const caret = document.createElement("span");
  caret.className = "tw-caret";
  el.appendChild(caret);

  for (let i = 0; i < s.length; i++) {
    if (runId !== overlayRunId) return;

    caret.remove();
    el.append(s[i]);
    el.appendChild(caret);

    const ch = s[i];
    const extra = ch === "." || ch === "!" || ch === "?" || ch === "," ? LINE_TYPE_SPEED_MS * 3 : 0;
    await sleep(LINE_TYPE_SPEED_MS + extra);
  }

  caret.remove();
}

async function runOverlaySequence(filesCount = 1) {
  const container = ensureLinesContainer();
  if (!container) return;

  const runId = ++overlayRunId;

  const useNodes = container === processingLines;
  if (useNodes) container.innerHTML = "";
  else container.textContent = "";

  const count = Math.max(1, Number(filesCount) || 1);
  const gapMs = Math.min(1200, LINE_GAP_BASE_MS * count);

  let currentNode = null;

  for (const line of OVERLAY_LINES) {
    if (runId !== overlayRunId) return;

    if (useNodes && currentNode) {
      currentNode.classList.remove("show");
      currentNode.classList.add("hide");
      await sleep(LINE_FADE_MS);
      currentNode.remove();
      currentNode = null;
    }

    if (useNodes) {
      const node = document.createElement("div");
      node.className = "status-line";
      container.appendChild(node);
      currentNode = node;

      requestAnimationFrame(() => node.classList.add("show"));
      await typewriterLine(node, line, runId);
    } else {
      await typewriterLine(container, line, runId);
    }

    await sleep(gapMs);
  }
}

function showProcessingOverlay(filesCount = 1) {
  if (!processingOverlay) return;

  cancelOverlaySequence();

  if (processingTitle) processingTitle.style.display = "none";

  processingOverlay.classList.remove("hidden");
  processingOverlay.setAttribute("aria-hidden", "false");

  initOverlayLottie();
  overlayLottie?.goToAndPlay(0, true);

  runOverlaySequence(filesCount);
}

async function showProcessingSuccess() {
  cancelOverlaySequence();

  const container = ensureLinesContainer();
  if (!container) return hideProcessingOverlay();

  const runId = overlayRunId;

  const msg = "Done. Training complete — I'm ready to answer using your Files.";

  if (container === processingLines) {
    const node = document.createElement("div");
    node.className = "status-line show";
    container.appendChild(node);
    await typewriterLine(node, msg, runId);
  } else {
    await typewriterLine(container, msg, runId);
  }

  await sleep(240);
  clearFilesAfterSuccess();
  hideProcessingOverlay();

  // NEW: after overlay completion, reset type to default
  resetTrainingType();
}

function clearFilesAfterSuccess() {
  selectedFiles = [];
  renderFileList();
}

function hideProcessingOverlay() {
  cancelOverlaySequence();
  overlayLottie?.stop();

  if (!processingOverlay) return;
  processingOverlay.classList.add("hidden");
  processingOverlay.setAttribute("aria-hidden", "true");
}

/* =========================
   8) FILE HANDLING
   ========================= */

function isPdf(file) {
  if (!file) return false;
  const nameOk = /\.pdf$/i.test(file.name || "");
  const typeOk = (file.type || "").toLowerCase() === "application/pdf";
  return nameOk || typeOk;
}

function isSupportedFile(file) {
  if (!file) return false;

  const name = (file.name || "").toLowerCase();
  const type = (file.type || "").toLowerCase();

  const typeOk = /\.(pdf|doc|docx|txt)$/.test(name);

  const mimeOk = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document", // .docx
    "text/plain",
  ].includes(type);

  return typeOk || mimeOk;
}

function fileKey(file) {
  return `${file.name}::${file.size}::${file.lastModified}`;
}

function addFiles(fileListLike) {
  const incoming = Array.from(fileListLike || []).filter(isSupportedFile);
  if (!incoming.length) {
    setStatus("Please select PDF files only.", "error");
    return;
  }

  const existing = new Set(selectedFiles.map(fileKey));
  let added = 0;

  for (const f of incoming) {
    const k = fileKey(f);
    if (existing.has(k)) continue;
    selectedFiles.push(f);
    existing.add(k);
    added++;
  }

  renderFileList();
  updateFileCount();
  setMetrics({
    files: selectedFiles.length,
    chunks: Number(mChunks?.textContent || 0),
    skipped: Number(mSkipped?.textContent || 0),
  });

  setStatus(added ? `Added ${added} File${added === 1 ? "" : "s"}.` : "These Files are already added.", "online");
}

function clearFiles() {
  selectedFiles = [];
  renderFileList();
  updateFileCount();
  setMetrics({ files: 0, chunks: 0, skipped: 0 });
  setProgress(0);
  setPhase("Idle");
  setStatus("Ready.", "online");
  log("Cleared file list.");
}

/* =========================
   9) UPLOAD / INGEST
   ========================= */

function startSimulatedProgress() {
  if (progressTimer) clearInterval(progressTimer);

  let p = 5;
  setProgress(p);

  progressTimer = setInterval(() => {
    p = Math.min(90, p + Math.max(0.8, (90 - p) * 0.08));
    setProgress(p);
  }, 260);
}

function stopSimulatedProgress(finalPct = 100) {
  if (progressTimer) clearInterval(progressTimer);
  progressTimer = null;
  setProgress(finalPct);
}

async function uploadAndIngest() {
  if (!selectedFiles.length) {
    setStatus("No Files selected. Add files first.", "error");
    log("Training blocked: no files selected.");
    return;
  }

  setDisabled(true);
  setStatus("Training in progress…", "online");
  setPhase("Uploading & indexing");
  log(`Starting Training: ${selectedFiles.length} File(s).`);

  showProcessingOverlay(selectedFiles.length);
  startSimulatedProgress();

  try {
    const form = new FormData();
    selectedFiles.forEach((f) => form.append("files", f, f.name));

    // OPTIONAL: if you later want to send type to backend, uncomment:
    form.append("train_type", selectedTrainType);

    const res = await fetch(INGEST_ENDPOINT, { method: "POST", body: form });

    const ct = res.headers.get("content-type") || "";
    const data = ct.includes("application/json") ? await res.json() : { message: await res.text() };

    if (!res.ok) {
      const msg = data?.error || data?.message || `Request failed (${res.status})`;
      throw new Error(msg);
    }

    const files = data?.files ?? selectedFiles.length;
    const chunks = data?.chunks ?? data?.total_chunks ?? 0;
    const skipped = data?.skipped ?? data?.skipped_chunks ?? 0;

    setMetrics({ files, chunks, skipped });
    stopSimulatedProgress(100);
    setPhase("Complete");
    setStatus("Training complete.", "online");
    log("Training complete.");
    log(`Metrics: files=${files}, Chapters=${chunks}, skipped=${skipped}`);

    await showProcessingSuccess();
  } catch (err) {
    stopSimulatedProgress(0);
    setPhase("Error");
    setStatus("Training failed.", "error");
    log(`ERROR: ${err?.message || err}`);
    hideProcessingOverlay();
    alert(`Training failed: ${err?.message || err}`);
  } finally {
    setDisabled(false);
  }
}

// Crawl
async function runCrawl() {
  setDisabled(true);
  setStatus("Auto training in progress…", "online");
  setPhase("Auto Trainig From Website");
  log("Starting auto-download and training…");

  setProgress(10);

  try {
    const res = await fetch(CRAWL_ENDPOINT, { method: "POST" });
    setProgress(25);
    const ct = res.headers.get("content-type") || "";
    const data = ct.includes("application/json") ? await res.json() : { message: await res.text() };
    setProgress(90);

    if (!res.ok) {
      const msg = data?.error || data?.message || `Request failed (${res.status})`;
      throw new Error(msg);
    }

    const files = data?.files ?? 0;
    const chunks = data?.chunks ?? 0;
    const skipped = data?.skipped ?? 0;

    setMetrics({ files, chunks, skipped });
    setProgress(100);
    setPhase("Complete");
    setStatus("Auto Training complete.", "online");
    log("Auto Training complete.");
    log(`Metrics: files=${files}, Chapters=${chunks}, skipped=${skipped}`);
  } catch (err) {
    setProgress(0);
    setPhase("Error");
    setStatus("Training failed.", "error");
    log(`ERROR: ${err?.message || err}`);
    alert(`Auto Download and Training stopped: ${err?.message || err}`);
  } finally {
    setDisabled(false);
  }
}

/* =========================
   10) EVENTS
   ========================= */

function bindEvents() {
  if (eventsBound) return;
  eventsBound = true;

  if (endpointText) endpointText.textContent = INGEST_ENDPOINT;

  on(uploadBtn, "click", (e) => {
    e.preventDefault();

    // NEW: must pick training type first
    if (!selectedTrainType) {
      showTrainTypeError();
      return;
    }

    uploadAndIngest();
  });

  on(clearBtn, "click", (e) => {
    e.preventDefault();
    clearFiles();
  });

  on(copyLogBtn, "click", (e) => {
    e.preventDefault();
    copyLogToClipboard();
  });

  trainingTypeItems.forEach((item) => {
    item.addEventListener("click", (e) => {
      e.preventDefault();

      selectedTrainType = (item.dataset.trainType || "").toLowerCase();
      setTrainingTypeLabel(selectedTrainType);

      // NEW: clear red message once selected
      hideTrainTypeError();
    });
  });

  on(dropzone, "click", (e) => {
    if (e.target === fileInput || fileInput?.contains(e.target)) return;
    fileInput?.click();
  });

  on(fileInput, "click", (e) => e.stopPropagation());
  on(fileInput, "change", (e) => {
    const files = e.target?.files;
    if (files?.length) addFiles(files);
    if (fileInput) fileInput.value = "";
  });

  on(dropzone, "dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  on(dropzone, "dragleave", () => dropzone.classList.remove("dragover"));

  on(dropzone, "drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    const files = e.dataTransfer?.files;
    if (files?.length) addFiles(files);
  });

  // Chat confirmation modal behavior
  const crawlerLink = document.getElementById("chatLink");
  const crawlerYesBtn = document.getElementById("chatConfirmYesBtn");

  if (crawlerLink) {
    crawlerLink.addEventListener("click", (e) => {
      e.preventDefault();
    });
  }

  if (crawlerYesBtn && crawlerLink) {
    crawlerYesBtn.addEventListener("click", async () => {
      const modalEl = document.getElementById("chatConfirmModal");
      const modalInstance = window.bootstrap?.Modal.getInstance(modalEl);
      modalInstance?.hide();

      await runCrawl();
    });
  }
}

/* =========================
   11) PARTICLES (optional)
   ========================= */

function initParticles() {
  if (!particlesCanvas) return;

  const ctx = particlesCanvas.getContext("2d");
  if (!ctx) return;

  let w = 0;
  let h = 0;
  let rafId = null;

  const PARTICLE_FILL = "rgba(2,127,15,.40)";
  const LINK_STROKE = "rgba(2,127,15,.30)";

  const cfg = { count: 48, maxLinkDist: 115, maxSpeed: 0.35 };
  let particles = [];

  function resize() {
    const rect = particlesCanvas.getBoundingClientRect();
    w = Math.max(1, Math.floor(rect.width));
    h = Math.max(1, Math.floor(rect.height));
    particlesCanvas.width = w * devicePixelRatio;
    particlesCanvas.height = h * devicePixelRatio;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  }

  function spawn() {
    particles = Array.from({ length: cfg.count }, () => ({
      x: Math.random() * w,
      y: Math.random() * h,
      vx: (Math.random() - 0.5) * cfg.maxSpeed * 2,
      vy: (Math.random() - 0.5) * cfg.maxSpeed * 2,
      r: 1 + Math.random() * 2.2,
    }));
  }

  function tick() {
    ctx.clearRect(0, 0, w, h);

    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > w) p.vx *= -1;
      if (p.y < 0 || p.y > h) p.vy *= -1;
    }

    ctx.strokeStyle = LINK_STROKE;
    ctx.globalAlpha = 0.15;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i];
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < cfg.maxLinkDist) {
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }
    ctx.globalAlpha = 1;

    ctx.fillStyle = PARTICLE_FILL;
    for (const p of particles) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }

    rafId = requestAnimationFrame(tick);
  }

  resize();
  spawn();
  tick();

  const ro = new ResizeObserver(() => {
    resize();
    spawn();
  });
  ro.observe(particlesCanvas);

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (rafId) cancelAnimationFrame(rafId);
      rafId = null;
    } else if (!rafId) {
      tick();
    }
  });
}

/* =========================
   12) INIT
   ========================= */

function init() {
  clearLog();
  updateFileCount();
  setMetrics({ files: 0, chunks: 0, skipped: 0 });
  setProgress(0);
  setPhase("Idle");
  setStatus("Ready.", "online");

  // NEW: ensure default label + hide error on load
  resetTrainingType();

  bindEvents();
  initParticles();
}

document.addEventListener("DOMContentLoaded", init);
