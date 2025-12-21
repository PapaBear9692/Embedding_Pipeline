/*******************************************************
 * Medicine AI - PDF Ingestion Console (Frontend)
 *
 * What this file does (high level):
 * 1) Lets you select / drag-drop PDF files
 * 2) Shows a file list (with remove buttons)
 * 3) Uploads files to the backend endpoint: /api/ingest
 * 4) Shows progress + console log updates
 * 5) Shows AI operator messages with typing effect
 * 6) Runs a simple particle animation in the background
 *
 * IMPORTANT NOTE:
 * Your CSS is the "source of truth" for class names.
 * This JS must use the SAME class names that style.css expects.
 *******************************************************/

/* =========================
   1) BACKEND CONFIG
   ========================= */

// If your frontend and backend are on the same domain,
// you can keep API_BASE as an empty string.
const API_BASE = "";

// Backend endpoint that receives PDFs
const INGEST_ENDPOINT = `${API_BASE}/api/ingest`;

/* =========================
   2) TYPEWRITER SPEED CONTROL
   ========================= */

// Base delay range (ms) per character for typewriter effect
const TYPE_BASE_MIN = 6;
const TYPE_BASE_MAX = 18;

// Pause slightly longer after punctuation
const TYPE_PUNCTUATION_BONUS = 60;

/* =========================
   3) ELEMENT REFERENCES
   ========================= */

// Helper: safely get an element (returns null if missing)
const $ = (sel) => document.querySelector(sel);

// Top bar / header pieces
const statusDot   = $(".ai-dot");
const statusText  = $("#statusText");
const phaseText   = $("#phaseText");
const progressBar = $("#progressBar");
const progressPct = $("#progressPct");

// File selection area
const dropzone   = $("#dropzone");
const fileInput  = $("#fileInput");
const fileList   = $("#fileList");

// Buttons
const uploadBtn  = $("#uploadBtn");
const clearBtn   = $("#clearBtn");

// Console log area
const consoleLog = $("#consoleLog");

// Chat area
const aiChat = $("#aiChat");
const aiMood = $("#aiMood");

/* =========================
   4) APP STATE
   ========================= */

// Where we store selected files before upload
let selectedFiles = [];

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
  if (!Number.isFinite(bytes)) return "";
  const sizes = ["B", "KB", "MB", "GB"];
  let i = 0;
  let n = bytes;

  while (n >= 1024 && i < sizes.length - 1) {
    n /= 1024;
    i++;
  }
  return `${n.toFixed(i === 0 ? 0 : 1)} ${sizes[i]}`;
}

function escapeHtml(str) {
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

/* =========================
   6) UI HELPERS
   ========================= */

function setDot(mode) {
  // The tiny status dot in the top bar.
  // Your CSS supports two "states":
  //   - .ai-dot.online  (green)
  //   - .ai-dot.error   (red)
  //
  // So we only ever add/remove those class names here.
  if (!statusDot) return;

  // Remove any previous state classes first
  statusDot.classList.remove("online", "error");

  // Add the requested state (if any)
  if (mode) statusDot.classList.add(mode);
}

function setProgress(pct) {
  const clamped = Math.max(0, Math.min(100, pct));
  if (progressBar) progressBar.style.width = `${clamped}%`;
  if (progressPct) progressPct.textContent = `${Math.round(clamped)}%`;
}

function setPhase(label) {
  if (phaseText) phaseText.textContent = label;
}

function setStatus(text, dotMode) {
  if (statusText) statusText.textContent = text;

  // dotMode should be "online" or "error" (or empty string)
  setDot(dotMode || "");
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

/* =========================
   7) CHAT UI (AI OPERATOR)
   ========================= */

function aiAppendBubble(text, type = "ai") {
  if (!aiChat) return null;

  const wrap = document.createElement("div");
  wrap.className = `ai-msg ${type}`;

  wrap.innerHTML = `
    <div class="ai-bubble">${escapeHtml(text)}</div>
  `;

  aiChat.appendChild(wrap);
  aiChat.scrollTop = aiChat.scrollHeight;
  return wrap;
}

async function typewriterToBubble(el, text) {
  if (!el) return;
  const bubble = el.querySelector(".ai-bubble");
  if (!bubble) return;

  const safe = escapeHtml(text);
  bubble.textContent = "";

  for (let i = 0; i < safe.length; i++) {
    bubble.textContent += safe[i];

    // Random jitter makes typing feel more human
    const jitter = TYPE_BASE_MIN + Math.random() * (TYPE_BASE_MAX - TYPE_BASE_MIN);

    // Add a little extra pause after punctuation
    const ch = safe[i];
    const bonus = /[.!?]/.test(ch) ? TYPE_PUNCTUATION_BONUS : 0;

    await sleep(jitter + bonus);
  }

  aiChat.scrollTop = aiChat.scrollHeight;
}

function aiTypingBubble(show) {
  if (!aiChat) return null;

  // If there is already a typing bubble, remove it first
  const existing = aiChat.querySelector(".ai-typing");
  if (existing) existing.remove();

  if (!show) return null;

  const wrap = document.createElement("div");
  wrap.className = "ai-msg ai ai-typing";

  wrap.innerHTML = `
    <div class="ai-bubble">
      <span class="ai-typing-dots">
        <span></span><span></span><span></span>
      </span>
    </div>
  `;

  aiChat.appendChild(wrap);
  aiChat.scrollTop = aiChat.scrollHeight;
  return wrap;
}

/* =========================
   8) FILE HANDLING
   ========================= */

function addFiles(fileListLike) {
  if (!fileListLike) return;

  // Convert FileList to a normal array and filter PDFs only
  const incoming = Array.from(fileListLike).filter((f) => {
    const isPdf = f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf");
    return isPdf;
  });

  if (!incoming.length) {
    aiAppendBubble("No valid PDFs detected. Please add .pdf files only.");
    return;
  }

  // Add to our state
  selectedFiles = selectedFiles.concat(incoming);

  // Re-render file list UI
  renderFileList();
}

function renderFileList() {
  if (!fileList) return;

  fileList.innerHTML = "";

  if (!selectedFiles.length) {
    fileList.innerHTML = `<div class="text-muted small">No files selected.</div>`;
    return;
  }

  selectedFiles.forEach((f, idx) => {
    const row = document.createElement("div");
    row.className = "ai-file";

    // Build one "file row" using the same class names defined in style.css.
    // This is important: if the class names don't match, the row will work
    // but it won't look the way you designed it.
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

  // Attach click handlers to remove buttons
  fileList.querySelectorAll("button.ai-file-x").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      const i = Number(e.currentTarget.dataset.idx);
      if (!Number.isFinite(i)) return;

      // Remove from state and re-render UI
      selectedFiles.splice(i, 1);
      renderFileList();
    });
  });
}

/* =========================
   9) UPLOAD / INGEST
   ========================= */

async function uploadAndIngest() {
  if (!selectedFiles.length) {
    aiAppendBubble("No PDFs selected. Add files first.");
    return;
  }

  // Lock UI during upload
  uploadBtn?.setAttribute("disabled", "true");
  clearBtn?.setAttribute("disabled", "true");
  fileInput?.setAttribute("disabled", "true");

  // Show “busy” mood
  if (aiMood) aiMood.textContent = "processing";

  // Status dot should be GREEN while working (CSS: .ai-dot.online)
  setStatus("Uploading PDFs…", "online");
  setPhase("Upload");
  setProgress(5);
  log("Starting ingestion request...");

  const typing = aiTypingBubble(true);

  try {
    const formData = new FormData();

    // Backend expects "files" (common pattern). If your backend expects a different key,
    // change "files" below.
    selectedFiles.forEach((f) => formData.append("files", f, f.name));

    setProgress(20);
    log(`Uploading ${selectedFiles.length} file(s) to ${INGEST_ENDPOINT}`);

    const res = await fetch(INGEST_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    setProgress(60);

    if (!res.ok) {
      // If server gives JSON error, try to read it; otherwise show plain status
      let errText = `HTTP ${res.status}`;
      try {
        const data = await res.json();
        errText = data?.error || data?.message || errText;
      } catch (_) {
        // ignore JSON parse errors
      }
      throw new Error(errText);
    }

    const data = await res.json().catch(() => ({}));

    setProgress(95);

    // Success state
    setStatus("Ingestion complete.", "online");
    setPhase("Complete");
    log("Ingestion complete.");

    // Optional metrics (if backend returns them)
    const files = data?.files ?? selectedFiles.length;
    const chunks = data?.chunks ?? "unknown";
    const seconds = data?.seconds ?? "unknown";
    log(`Metrics: files=${files}, chunks=${chunks}, seconds=${seconds}`);

    // Replace typing indicator with a typewritten reply
    aiTypingBubble(false);
    const bubble = aiAppendBubble("", "ai");
    await typewriterToBubble(
      bubble,
      `Ingestion successful ✅\nProcessed ${files} file(s). Ready for your next upload.`
    );

    // Clear file state after successful ingestion
    selectedFiles = [];
    renderFileList();
    setProgress(100);
  } catch (err) {
    aiTypingBubble(false);

    // Error state: RED dot (CSS: .ai-dot.error)
    setStatus("Error during ingestion.", "error");
    setPhase("Failed");
    setProgress(0);
    log(`ERROR: ${String(err?.message || err)}`);

    const bubble = aiAppendBubble("", "ai");
    await typewriterToBubble(
      bubble,
      `I hit an error talking to the server.\nReason: ${String(err?.message || err)}\nCheck the console log and try again.`
    );
  } finally {
    // Re-enable UI no matter what happened
    uploadBtn?.removeAttribute("disabled");
    clearBtn?.removeAttribute("disabled");
    fileInput?.removeAttribute("disabled");

    // Back to “online”
    if (aiMood) aiMood.textContent = "online";
  }
}

/* =========================
   10) EVENTS (CLICK, DRAG-DROP, BUTTONS)
   ========================= */

function bindEvents() {
  if (dropzone) {
    // Clicking the dropzone opens the file picker
    dropzone.addEventListener("click", () => fileInput?.click());

    // When the user drags files over the drop area, prevent the browser from opening the file
    // and add the "dragover" class so CSS can highlight the dropzone.
    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault(); // IMPORTANT: allows drop
      dropzone.classList.add("dragover"); // matches .ai-dropzone.dragover in CSS
    });

    // Remove highlight when leaving
    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });

    // Handle dropping files
    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      addFiles(e.dataTransfer.files);
    });
  }

  if (fileInput) {
    // File picker -> add selected PDFs
    fileInput.addEventListener("change", (e) => {
      addFiles(e.target.files);
      // Reset input so selecting the same file again still triggers change
      e.target.value = "";
    });
  }

  uploadBtn?.addEventListener("click", uploadAndIngest);

  clearBtn?.addEventListener("click", () => {
    selectedFiles = [];
    renderFileList();
    clearLog();
    aiChat && (aiChat.innerHTML = "");
    aiAppendBubble("Cleared. Ready for new PDFs.");
    setStatus("Idle. Awaiting PDFs…", "");
    setPhase("Ready");
    setProgress(0);
  });
}

/* =========================
   11) PARTICLES (BACKGROUND CANVAS)
   ========================= */

function initParticles() {
  const canvas = document.getElementById("bgParticles");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let w, h;
  let particles = [];

  function resize() {
    w = canvas.width = window.innerWidth;
    h = canvas.height = window.innerHeight;
  }

  function spawn(count = 80) {
    particles = [];
    for (let i = 0; i < count; i++) {
      particles.push({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.4,
        vy: (Math.random() - 0.5) * 0.4,
        r: 1 + Math.random() * 1.5,
      });
    }
  }

  function step() {
    ctx.clearRect(0, 0, w, h);

    // Draw particles
    ctx.globalAlpha = 0.65;
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;

      // Wrap around edges
      if (p.x < 0) p.x = w;
      if (p.x > w) p.x = 0;
      if (p.y < 0) p.y = h;
      if (p.y > h) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }

    // Light connecting lines
    ctx.globalAlpha = 0.12;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i];
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 110) {
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }
      }
    }

    requestAnimationFrame(step);
  }

  window.addEventListener("resize", () => {
    resize();
    spawn();
  });

  resize();
  spawn();
  step();
}

/* =========================
   12) INIT
   ========================= */

function initUI() {
  setStatus("Idle. Awaiting PDFs…", "");
  setPhase("Ready");
  setProgress(0);
  clearLog();

  aiChat && (aiChat.innerHTML = "");
  aiAppendBubble("Console online. Drop PDFs and press “Upload & Embed”.");
}

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  initUI();
  initParticles();
});
