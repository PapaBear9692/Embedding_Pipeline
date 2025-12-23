/*******************************************************
 * Medicine ChatBot - Training Console (Frontend)
 *
 * FIXES INCLUDED:
 * ✅ Prevents file picker opening twice:
 *   - bindEvents() guarded (no double listener attachment)
 *   - dropzone click ignores clicks from/inside file input
 *   - fileInput click stops propagation safely (only if fileInput exists)
 *******************************************************/

/* =========================
   1) BACKEND CONFIG
   ========================= */

const API_BASE = "";
const INGEST_ENDPOINT = `${API_BASE}/api/ingest`;

/* =========================
   2) TYPEWRITER SPEED CONTROL
   ========================= */

const TYPE_BASE_MIN = 6;
const TYPE_BASE_MAX = 18;
const TYPE_PUNCTUATION_BONUS = 60;

// Separate speed control for processingSub typewriter (ms per character)
const SUB_TYPE_SPEED_MS = 60;

/* =========================
   3) ELEMENT REFERENCES
   ========================= */

const $ = (sel) => document.querySelector(sel);

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

// File selection area
const dropzone = $("#dropzone");
const fileInput = $("#fileInput");
const fileList = $("#fileList");
const fileCount = $("#fileCount");

// Buttons
const uploadBtn = $("#uploadBtn");
const clearBtn = $("#clearBtn");
const copyLogBtn = $("#copyLogBtn");

// Console log area
const consoleLog = $("#consoleLog");

// Chat area
const aiChat = $("#aiChat");
const aiMood = $("#aiMood");

/* =========================
   4) APP STATE
   ========================= */

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

function clampInt(n, min, max) {
  const x = Number(n);
  if (!Number.isFinite(x)) return min;
  return Math.max(min, Math.min(max, Math.trunc(x)));
}

/* =========================
   6) UI HELPERS
   ========================= */

function setDot(mode) {
  if (!statusDot) return;
  statusDot.classList.remove("online", "error");
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
  setDot(dotMode || "");
}

function setMetrics({ files = 0, chunks = 0, skipped = 0 } = {}) {
  if (mFiles) mFiles.textContent = String(clampInt(files, 0, 1_000_000));
  if (mChunks) mChunks.textContent = String(clampInt(chunks, 0, 9_999_999));
  if (mSkipped) mSkipped.textContent = String(clampInt(skipped, 0, 9_999_999));
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

function setFileCount(n) {
  if (!fileCount) return;
  fileCount.textContent = String(clampInt(n, 0, 1_000_000));
}


/* =========================
   PROCESSING OVERLAY (ANIMATION)
   ========================= */

const processingOverlay = document.getElementById("processingOverlay");
const processingTitle = document.getElementById("processingTitle");
const processingSub = document.getElementById("processingSub");
const dots = document.getElementById("dots");

let dotsTimer = null;

let __subTypeRunId = 0;

async function typewriterProcessingSub(text) {
  if (!processingSub) return;

  const runId = ++__subTypeRunId;
  processingSub.textContent = "";

  const s = String(text);

  for (let i = 0; i < s.length; i++) {
    // cancel if a new run started
    if (runId !== __subTypeRunId) return;

    processingSub.textContent += s[i];

    // small natural pauses
    const ch = s[i];
    const extra =
      (ch === "." || ch === "!" || ch === "?" || ch === ",") ? SUB_TYPE_SPEED_MS * 3 :
      (ch === "\n") ? SUB_TYPE_SPEED_MS * 6 :
      0;

    await sleep(SUB_TYPE_SPEED_MS + extra);
  }
}

function showProcessingOverlay() {
  if (!processingOverlay) return;

  processingOverlay.classList.remove("hidden");
  processingOverlay.setAttribute("aria-hidden", "false");

  if (processingTitle && processingTitle.firstChild) {
  processingTitle.firstChild.nodeValue = "Please, give me some time, processing the files ";}

  typewriterProcessingSub("I'm reading the documents and trying to understand them.");

  // animated dots ( ... )
  if (dots) {
    let n = 0;
    dots.textContent = "...";
    if (dotsTimer) clearInterval(dotsTimer);
    dotsTimer = setInterval(() => {
      n = (n + 1) % 4;
      dots.textContent = ".".repeat(n) || "";
    }, 350);
  }
}

async function showProcessingSuccess() {
  if (!processingOverlay) return;

  if (dotsTimer) clearInterval(dotsTimer);
  dotsTimer = null;
  if (dots) dots.textContent = "";

  if (processingTitle && processingTitle.firstChild) {
  processingTitle.firstChild.nodeValue = "Got it, task has been executed successful ";}

  // auto-hide after a short moment
  await typewriterProcessingSub("Your files were processed and added to my knowledge base.");
  await sleep(350);
  hideProcessingOverlay();
}

function hideProcessingOverlay() {

  __subTypeRunId++; // cancel any running processingSub typewriter

  if (dotsTimer) clearInterval(dotsTimer);
  dotsTimer = null;

  if (!processingOverlay) return;
  processingOverlay.classList.add("hidden");
  processingOverlay.setAttribute("aria-hidden", "true");
}



/* =========================
   7) CHAT UI (AI OPERATOR)
   ========================= */

function aiAppendBubble(text, type = "ai") {
  if (!aiChat) return null;
  const wrap = document.createElement("div");
  wrap.className = `ai-msg ${type}`;
  wrap.innerHTML = `<div class="ai-bubble">${escapeHtml(text)}</div>`;
  aiChat.appendChild(wrap);
  aiChat.scrollTop = aiChat.scrollHeight;
  return wrap;
}

function formatLite(input) {
  const esc = escapeHtml(String(input));

  let s = esc.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  const lines = s.split(/\r?\n/);
  let html = "";
  let inList = false;

  const closeList = () => {
    if (inList) {
      html += "</ul>";
      inList = false;
    }
  };

  for (const line of lines) {
    const trimmed = line.trim();

    const isBullet = /^[-•]\s+/.test(trimmed);
    if (isBullet) {
      if (!inList) {
        closeList();
        html += "<ul>";
        inList = true;
      }
      html += `<li>${trimmed.replace(/^[-•]\s+/, "")}</li>`;
      continue;
    }

    closeList();

    if (trimmed === "") {
      html += "<p></p>";
    } else {
      html += `<p>${line}</p>`;
    }
  }

  closeList();
  return html;
}

async function typewriterToBubble(el, text) {
  if (!el) return;
  const bubble = el.querySelector(".ai-bubble");
  if (!bubble) return;

  bubble.innerHTML = `<div class="fmt"></div>`;
  const holder = bubble.querySelector(".fmt");
  if (!holder) return;

  const tokens = String(text).split(/(\s+)/);
  let out = "";

  for (let i = 0; i < tokens.length; i++) {
    out += tokens[i];

    if (i % 4 === 0 || i === tokens.length - 1) {
      holder.innerHTML = formatLite(out);
      aiChat.scrollTop = aiChat.scrollHeight;
    }

    const last = tokens[i] || "";
    const base = 18 + Math.random() * 40;
    const punct = /[.!?]\s*$/.test(last) ? 120 : 0;
    const nl = /\n/.test(last) ? 160 : 0;

    await sleep(base + punct + nl);
  }

  aiChat.scrollTop = aiChat.scrollHeight;
}

function aiTypingBubble(show) {
  if (!aiChat) return null;

  const existing = aiChat.querySelector(".ai-msg.ai-typing");
  if (existing) existing.remove();

  if (!show) return null;

  const wrap = document.createElement("div");
  wrap.className = "ai-msg ai ai-typing";
  wrap.innerHTML = `
    <div class="ai-bubble">
      <span class="ai-typing-dots" aria-label="AI is typing">
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

  const incoming = Array.from(fileListLike).filter((f) => {
    const isPdf = f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf");
    return isPdf;
  });

  if (!incoming.length) {
    aiAppendBubble("No valid PDFs detected. Please add .pdf files only.");
    return;
  }

  selectedFiles = selectedFiles.concat(incoming);
  renderFileList();
}

function renderFileList() {
  if (!fileList) return;

  fileList.innerHTML = "";

  if (!selectedFiles.length) {
    fileList.innerHTML = `<div class="ai-muted small">No files selected.</div>`;
    setFileCount(0);
    return;
  }

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

  setFileCount(selectedFiles.length);

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
   9) UPLOAD / INGEST
   ========================= */

async function uploadAndIngest() {

  let ingestSucceeded = false;

  if (!selectedFiles.length) {
    aiAppendBubble("No PDFs selected. Add files first.");
    return;
  }

  uploadBtn?.setAttribute("disabled", "true");
  clearBtn?.setAttribute("disabled", "true");
  fileInput?.setAttribute("disabled", "true");
  dropzone?.setAttribute("aria-disabled", "true");

  if (aiMood) aiMood.textContent = "processing";

  setStatus("Uploading PDFs…", "online");
  setPhase("Upload");
  setProgress(5);
  log("Starting training request...");

  aiTypingBubble(true);

  // animation overlay
  // ✅ ADD THIS
  showProcessingOverlay();
  // animation overlay


  try {
    const formData = new FormData();
    selectedFiles.forEach((f) => formData.append("files", f, f.name));

    setProgress(20);
    log(`Uploading ${selectedFiles.length} file(s) to server...`);

    const res = await fetch(INGEST_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    setProgress(60);

    if (!res.ok) {
      let errText = `HTTP ${res.status}`;
      try {
        const data = await res.json();
        errText = data?.error || data?.message || errText;
      } catch (_) {}
      throw new Error(errText);
    }

    const data = await res.json().catch(() => ({}));
    setProgress(95);

    const files = Number.isFinite(Number(data?.files)) ? Number(data.files) : selectedFiles.length;
    const chunks = Number.isFinite(Number(data?.chunks)) ? Number(data.chunks) : 0;
    const skipped = Number.isFinite(Number(data?.skipped)) ? Number(data.skipped) : 0;

    setMetrics({ files, chunks, skipped });

    setStatus("Training complete.", "online");
    setPhase("Complete");
    log("Training complete.");
    
    // overlay
    ingestSucceeded = true;
    // ✅ ADD THIS
    await showProcessingSuccess();
    // overlay

    log(`Metrics: files=${files}, chunks=${chunks}, skipped=${skipped}`);

    aiTypingBubble(false);
    const bubble = aiAppendBubble("", "ai");
    await typewriterToBubble(
      bubble,
      `**Training successful ✅**\n- Processed **${files}** file(s)\n- Ready for your next upload`
    );

    selectedFiles = [];
    renderFileList();
    setProgress(100);
  } catch (err) {
    
    // overlay animation
    // ✅ ADD THIS
    if (!ingestSucceeded) hideProcessingOverlay();
    // overlay animation

    aiTypingBubble(false);

    setStatus("Error during training.", "error");
    setPhase("Failed");
    setProgress(0);
    log(`ERROR: ${String(err?.message || err)}`);

    const bubble = aiAppendBubble("", "ai");
    await typewriterToBubble(
      bubble,
      `I hit an error talking to the server.\n\n**Reason:** \`${String(err?.message || err)}\`\n\nCheck the console log and try again.`
    );
  } finally {

    // animation overlay
    // ✅ ADD THIS (safety in case any path didn’t close it)
    if (!ingestSucceeded) hideProcessingOverlay();
    // animation overlay

    uploadBtn?.removeAttribute("disabled");
    clearBtn?.removeAttribute("disabled");
    fileInput?.removeAttribute("disabled");
    dropzone?.removeAttribute("aria-disabled");

    if (aiMood) aiMood.textContent = "online";
  }
}

/* =========================
   10) EVENTS
   ========================= */

function bindEvents() {
  // ✅ Guard: prevents duplicate listener attachment (main cause of double file picker)
  if (window.__medicineAiEventsBound) return;
  window.__medicineAiEventsBound = true;

  if (dropzone) {
    // Click opens file picker (once)
    dropzone.addEventListener("click", (e) => {
      // If click came from the input (or inside it), do nothing
      if (fileInput && (e.target === fileInput || fileInput.contains(e.target))) return;
      e.preventDefault();
      fileInput?.click();
    });

    // Keyboard support (Enter/Space)
    dropzone.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        fileInput?.click();
      }
    });

    dropzone.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzone.classList.add("dragover");
    });

    dropzone.addEventListener("dragleave", () => {
      dropzone.classList.remove("dragover");
    });

    dropzone.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzone.classList.remove("dragover");
      addFiles(e.dataTransfer.files);
    });
  }

  // ✅ Always null-check fileInput
  if (fileInput) {
    // Prevent bubbling to dropzone which would trigger fileInput.click() again
    fileInput.addEventListener("click", (e) => e.stopPropagation());

    fileInput.addEventListener("change", (e) => {
      addFiles(e.target.files);
      e.target.value = ""; // allows selecting same file again
    });
  }

  uploadBtn?.addEventListener("click", uploadAndIngest);

  clearBtn?.addEventListener("click", () => {
    selectedFiles = [];
    renderFileList();
    clearLog();
    if (aiChat) aiChat.innerHTML = "";
    aiAppendBubble("Cleared. Ready for new PDFs.");
    setStatus("Idle. Awaiting PDFs…", "");
    setPhase("Ready");
    setProgress(0);
    setMetrics({ files: 0, chunks: 0, skipped: 0 });
  });

  copyLogBtn?.addEventListener("click", async () => {
    const txt = consoleLog?.textContent || "";
    if (!txt.trim()) return;

    try {
      await navigator.clipboard.writeText(txt);
      log("Copied console log to clipboard.");
    } catch {
      try {
        const ta = document.createElement("textarea");
        ta.value = txt;
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
        log("Copied console log to clipboard.");
      } catch {
        log("Copy failed. Your browser blocked clipboard access.");
      }
    }
  });
}

/* =========================
   11) PARTICLES
   ========================= */

function initParticles() {
  const canvas = document.getElementById("aiParticles");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  ctx.fillStyle = "rgba(2,127,15,.45)";
  ctx.strokeStyle = "rgba(2,127,15,.35)";

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
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        r: 1 + Math.random() * 1.5,
      });
    }
  }

  function step() {
    ctx.clearRect(0, 0, w, h);

    ctx.globalAlpha = 0.65;
    for (const p of particles) {
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0) p.x = w;
      if (p.x > w) p.x = 0;
      if (p.y < 0) p.y = h;
      if (p.y > h) p.y = 0;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }

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

  setMetrics({ files: 0, chunks: 0, skipped: 0 });

  if (aiChat) aiChat.innerHTML = "";
  aiAppendBubble("Console online. Drop PDFs and press “Upload & Train.");
  renderFileList();
}

document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  initUI();
  initParticles();
});
