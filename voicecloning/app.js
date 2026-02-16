// =============================================================================
// RVC Voice Cloning — Browser Prototype (app.js)
//
// Pipeline: Audio -> ContentVec (HuBERT) -> F0 (RMVPE) -> Retrieval -> net_g -> WAV
// Training: Audio samples -> ContentVec features + F0 stats -> Voice Index
// Runtime:  ONNX Runtime Web (WebGPU preferred, WASM fallback)
//
// ContentVec and RMVPE auto-download from HuggingFace and cache in IndexedDB.
// Users can train a voice profile in-browser OR upload an RVC generator (.onnx).
// =============================================================================

"use strict";

// ---------------------------------------------------------------------------
// Auto-download model URLs (hosted on HuggingFace)
// ---------------------------------------------------------------------------
const MODEL_URLS = {
  contentvec: "https://huggingface.co/Xenova/hubert-base-ls960/resolve/main/onnx/model.onnx",
  f0: "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.onnx",
};

const MODEL_CACHE_DB = "rvc-model-cache";
const MODEL_CACHE_STORE = "models";

// ---------------------------------------------------------------------------
// Global state
// ---------------------------------------------------------------------------
const state = {
  sessions: { contentvec: null, f0: null, generator: null },
  modelBuffers: { contentvec: null, f0: null, generator: null },
  modelNames: { contentvec: "hubert-base-ls960.onnx", f0: "rmvpe.onnx", generator: "" },
  index: null,
  indexName: "",
  sourceAudioBuffer: null,
  sourceFile: null,
  outputBuffer: null,
  outputBlob: null,
  recording: false,
  mediaRecorder: null,
  recordedChunks: [],
  backend: "wasm",
  // Training state
  activeTab: "train",
  trainingSamples: [],       // [{file, audioBuffer, name, duration}]
  trainedIndex: null,        // Float32Array[] — the built voice index
  trainedVoiceName: "",
  trainedVoiceStats: null,   // {avgPitch, minPitch, maxPitch, totalDuration, vectorCount}
  trainingRecording: false,
  trainingMediaRecorder: null,
  trainingRecordedChunks: [],
};

// ---------------------------------------------------------------------------
// DOM refs
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const logArea = $("logArea");
const progressContainer = $("progressContainer");
const progressFill = $("progressFill");
const progressLabel = $("progressLabel");

const paramDisplays = {
  pitchShift:   { el: $("pitchVal"),       fmt: v => v },
  indexRatio:   { el: $("indexRatioVal"),   fmt: v => (v / 100).toFixed(2) },
  filterRadius: { el: $("filterRadiusVal"), fmt: v => v },
  volEnvelope:  { el: $("volEnvVal"),       fmt: v => (v / 100).toFixed(2) },
  protect:      { el: $("protectVal"),      fmt: v => (v / 100).toFixed(2) },
};

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------
function log(msg, level = "info") {
  const line = document.createElement("div");
  line.className = `log-line ${level}`;
  const ts = new Date().toLocaleTimeString();
  line.textContent = `[${ts}] ${msg}`;
  logArea.appendChild(line);
  logArea.scrollTop = logArea.scrollHeight;
  console.log(`[${level}] ${msg}`);
}

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------
function showProgress(pct, text) {
  progressContainer.style.display = "block";
  progressFill.style.width = `${pct}%`;
  progressLabel.textContent = text;
}
function hideProgress() { progressContainer.style.display = "none"; }

function showTrainProgress(pct, text) {
  const c = $("trainProgressContainer");
  const f = $("trainProgressFill");
  const l = $("trainProgressLabel");
  c.style.display = "block";
  f.style.width = `${pct}%`;
  l.textContent = text;
}
function hideTrainProgress() { $("trainProgressContainer").style.display = "none"; }

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------
function switchTab(tab) {
  state.activeTab = tab;
  document.querySelectorAll(".tab-btn").forEach(b => b.classList.toggle("active", b.dataset.tab === tab));
  $("tabTrain").classList.toggle("active", tab === "train");
  $("tabConvert").classList.toggle("active", tab === "convert");
}

// ---------------------------------------------------------------------------
// Step indicators
// ---------------------------------------------------------------------------
function updateSteps() {
  const hasContentVec = !!state.sessions.contentvec;
  const hasGenerator = !!state.sessions.generator;
  const hasTrainedIndex = !!state.trainedIndex;
  const hasIndex = !!state.index;
  const modelsReady = hasContentVec && (hasGenerator || hasTrainedIndex || hasIndex);
  const hasAudio = !!state.sourceAudioBuffer;
  const hasOutput = !!state.outputBlob;

  $("step1").className = `step ${modelsReady ? "done" : "active"}`;
  $("step2").className = `step ${hasAudio ? "done" : (modelsReady ? "active" : "")}`;
  $("step3").className = `step ${hasOutput ? "done" : (hasAudio && modelsReady ? "active" : "")}`;
  $("step4").className = `step ${hasOutput ? "active" : ""}`;

  // Convert button: need ContentVec + audio + (generator OR trained index OR uploaded index)
  $("btnConvert").disabled = !(hasContentVec && hasAudio && (hasGenerator || hasTrainedIndex || hasIndex));

  $("dlAudio").disabled = !hasOutput;
  $("dlBundle").disabled = !(state.modelBuffers.contentvec || state.modelBuffers.f0 || state.modelBuffers.generator || hasTrainedIndex || hasIndex);
  $("dlVoiceProfile").disabled = !(hasTrainedIndex || hasIndex);

  // Train button: need ContentVec + at least one sample
  $("btnTrain").disabled = !(hasContentVec && state.trainingSamples.length > 0);

  // Trained voice indicator in sidebar
  if (hasTrainedIndex) {
    $("trainedIndicator").style.display = "block";
    const name = state.trainedVoiceName || "Untitled Voice";
    $("trainedVoiceLabel").textContent = `Trained: ${name}`;
    const stats = state.trainedVoiceStats;
    $("trainedVoiceDesc").textContent = stats
      ? `${stats.vectorCount} vectors | ${stats.totalDuration.toFixed(1)}s | Avg ${stats.avgPitch.toFixed(0)}Hz`
      : `Voice index ready`;
  } else {
    $("trainedIndicator").style.display = "none";
  }
}

// ---------------------------------------------------------------------------
// WebGPU detection
// ---------------------------------------------------------------------------
async function checkWebGPU() {
  const dot = $("gpuDot");
  const label = $("gpuLabel");
  if (navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        dot.className = "dot green";
        label.textContent = "WebGPU available";
        state.backend = "webgpu";
        log("WebGPU detected — will use GPU acceleration.", "success");
        return;
      }
    } catch (e) { /* fall through */ }
  }
  dot.className = "dot yellow";
  label.textContent = "WebGPU unavailable, using WASM";
  state.backend = "wasm";
  log("WebGPU not available. Using WASM backend (slower).", "warn");
}

// ---------------------------------------------------------------------------
// IndexedDB cache for models
// ---------------------------------------------------------------------------
function openCacheDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(MODEL_CACHE_DB, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(MODEL_CACHE_STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function getCachedModel(key) {
  try {
    const db = await openCacheDB();
    return new Promise((resolve) => {
      const tx = db.transaction(MODEL_CACHE_STORE, "readonly");
      const store = tx.objectStore(MODEL_CACHE_STORE);
      const req = store.get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    });
  } catch { return null; }
}

async function setCachedModel(key, buffer) {
  try {
    const db = await openCacheDB();
    return new Promise((resolve) => {
      const tx = db.transaction(MODEL_CACHE_STORE, "readwrite");
      const store = tx.objectStore(MODEL_CACHE_STORE);
      store.put(buffer, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => resolve();
    });
  } catch { /* ignore */ }
}

// ---------------------------------------------------------------------------
// Download with progress
// ---------------------------------------------------------------------------
async function downloadWithProgress(url, onProgress) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status}: ${response.statusText}`);

  const contentLength = parseInt(response.headers.get("content-length") || "0");
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (contentLength > 0 && onProgress) {
      onProgress(received / contentLength);
    }
  }

  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }
  return buffer.buffer;
}

// ---------------------------------------------------------------------------
// ONNX session creation
// ---------------------------------------------------------------------------
async function createSession(buffer) {
  const opts = {
    executionProviders: state.backend === "webgpu"
      ? ["webgpu", "wasm"]
      : ["wasm"],
  };
  return await ort.InferenceSession.create(buffer, opts);
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / 1048576).toFixed(1) + " MB";
}

// ---------------------------------------------------------------------------
// Auto-download a model (with IndexedDB cache)
// ---------------------------------------------------------------------------
async function autoLoadModel(key, url, statusId, progId, slotId, sizeId) {
  const statusEl = $(statusId);
  const progBar = $(progId);
  const progFill = progBar.querySelector(".fill");
  const slotEl = $(slotId);
  const sizeEl = $(sizeId);

  // 1. Check IndexedDB cache first
  statusEl.textContent = "Checking cache...";
  statusEl.className = "status downloading";
  const cached = await getCachedModel(key);

  let buffer;
  if (cached) {
    buffer = cached;
    statusEl.textContent = "Cached";
    statusEl.className = "status cached";
    sizeEl.textContent = `From cache (${formatBytes(buffer.byteLength)})`;
    log(`${key}: Found in browser cache (${formatBytes(buffer.byteLength)}).`, "success");
  } else {
    // 2. Download from HuggingFace
    statusEl.textContent = "Downloading...";
    statusEl.className = "status downloading";
    progBar.style.display = "block";
    log(`${key}: Downloading from HuggingFace (~may take a minute)...`);

    try {
      buffer = await downloadWithProgress(url, (pct) => {
        progFill.style.width = `${(pct * 100).toFixed(0)}%`;
        statusEl.textContent = `${(pct * 100).toFixed(0)}%`;
      });
      log(`${key}: Download complete (${formatBytes(buffer.byteLength)}).`, "success");

      // 3. Cache in IndexedDB
      log(`${key}: Caching in browser for next time...`);
      await setCachedModel(key, buffer);
      log(`${key}: Cached in IndexedDB.`, "success");

      sizeEl.textContent = `Downloaded & cached (${formatBytes(buffer.byteLength)})`;
    } catch (e) {
      statusEl.textContent = "Failed";
      statusEl.className = "status empty";
      progBar.style.display = "none";
      log(`${key}: Download failed: ${e.message}`, "error");
      log(`${key}: You can manually upload the model by clicking the slot.`, "warn");
      return;
    }
  }

  progBar.style.display = "none";

  // 4. Create ONNX session
  statusEl.textContent = "Loading...";
  try {
    const session = await createSession(buffer);
    state.sessions[key] = session;
    state.modelBuffers[key] = buffer;
    statusEl.textContent = "Ready";
    statusEl.className = "status ok";
    slotEl.classList.add("loaded");

    const inputs = session.inputNames.join(", ");
    const outputs = session.outputNames.join(", ");
    log(`${key}: ONNX session created. Inputs: [${inputs}] Outputs: [${outputs}]`, "success");
  } catch (e) {
    statusEl.textContent = "Error";
    statusEl.className = "status empty";
    log(`${key}: Failed to create ONNX session: ${e.message}`, "error");
    log(`${key}: The model format may not be compatible. Try uploading manually.`, "warn");
  }

  updateSteps();
}

// ---------------------------------------------------------------------------
// Manual model upload (for generator + optional overrides)
// ---------------------------------------------------------------------------
function setupModelSlot(slotId, fileId, statusId, sizeId, key) {
  const slot = $(slotId);
  const fileInput = $(fileId);

  slot.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    await loadModelFile(file, key, statusId, sizeId, slotId);
  });

  slot.addEventListener("dragover", (e) => { e.preventDefault(); slot.style.borderColor = "var(--accent)"; });
  slot.addEventListener("dragleave", () => { slot.style.borderColor = ""; });
  slot.addEventListener("drop", async (e) => {
    e.preventDefault();
    slot.style.borderColor = "";
    const file = e.dataTransfer.files[0];
    if (file) await loadModelFile(file, key, statusId, sizeId, slotId);
  });
}

async function loadModelFile(file, key, statusId, sizeId, slotId) {
  const statusEl = $(statusId);
  const sizeEl = $(sizeId);
  const slotEl = $(slotId);

  statusEl.textContent = "Loading...";
  statusEl.className = "status downloading";
  log(`Loading ${key}: ${file.name} (${formatBytes(file.size)})...`);

  try {
    if (key === "index") {
      await loadIndex(file);
      statusEl.textContent = "Loaded";
      statusEl.className = "status ok";
      sizeEl.textContent = `${file.name} (${formatBytes(file.size)})`;
      slotEl.classList.add("loaded");
      state.indexName = file.name;
      log(`Index loaded: ${file.name}`, "success");
    } else {
      const buffer = await file.arrayBuffer();
      const session = await createSession(buffer.slice(0));
      state.sessions[key] = session;
      state.modelBuffers[key] = buffer;
      state.modelNames[key] = file.name;
      statusEl.textContent = "Loaded";
      statusEl.className = "status ok";
      sizeEl.textContent = `${file.name} (${formatBytes(file.size)})`;
      slotEl.classList.add("loaded");

      const inputs = session.inputNames.join(", ");
      const outputs = session.outputNames.join(", ");
      log(`${key} loaded. Inputs: [${inputs}] Outputs: [${outputs}]`, "success");
    }
  } catch (e) {
    statusEl.textContent = "Error";
    statusEl.className = "status empty";
    log(`Failed to load ${key}: ${e.message}`, "error");
  }

  updateSteps();
}

async function loadIndex(file) {
  const ext = file.name.split(".").pop().toLowerCase();
  const buf = await file.arrayBuffer();

  if (ext === "json") {
    const text = new TextDecoder().decode(buf);
    const data = JSON.parse(text);
    // Support voice profile format
    if (data.format === "rvc-voice-profile" && data.index) {
      state.index = data.index.map((row) => new Float32Array(row));
      state.trainedVoiceName = data.voiceName || "";
      if (data.stats) {
        state.trainedVoiceStats = data.stats;
        state.trainedIndex = state.index;
      }
      log(`Voice profile loaded: "${data.voiceName}" (${state.index.length} vectors)`, "success");
    } else {
      state.index = data.map((row) => new Float32Array(row));
    }
  } else if (ext === "bin") {
    const flat = new Float32Array(buf);
    const dim = 768;
    const count = Math.floor(flat.length / dim);
    state.index = [];
    for (let i = 0; i < count; i++) {
      state.index.push(flat.subarray(i * dim, (i + 1) * dim));
    }
  } else if (ext === "npy") {
    state.index = parseNpy(buf);
  } else {
    throw new Error("Unsupported index format: " + ext);
  }
  log(`Index: ${state.index.length} vectors loaded.`);
}

function parseNpy(buffer) {
  const view = new DataView(buffer);
  const headerLen = view.getUint16(8, true);
  const headerStr = new TextDecoder().decode(new Uint8Array(buffer, 10, headerLen));
  const shapeMatch = headerStr.match(/\((\d+),\s*(\d+)\)/);
  if (!shapeMatch) throw new Error("Cannot parse .npy shape from header");
  const rows = parseInt(shapeMatch[1]);
  const cols = parseInt(shapeMatch[2]);
  const dataOffset = 10 + headerLen;
  const flat = new Float32Array(buffer, dataOffset, rows * cols);
  const vectors = [];
  for (let i = 0; i < rows; i++) {
    vectors.push(flat.slice(i * cols, (i + 1) * cols));
  }
  return vectors;
}

// ---------------------------------------------------------------------------
// Audio input (Convert tab)
// ---------------------------------------------------------------------------
function setupAudioInput() {
  const dropZone = $("dropZone");
  const fileInput = $("audioFileInput");

  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files[0]) handleAudioFile(e.dataTransfer.files[0]);
  });
  fileInput.addEventListener("change", (e) => {
    if (e.target.files[0]) handleAudioFile(e.target.files[0]);
  });

  $("btnRecord").addEventListener("click", startRecording);
  $("btnStopRecord").addEventListener("click", stopRecording);
}

async function handleAudioFile(file) {
  log(`Audio file: ${file.name} (${formatBytes(file.size)})`);
  state.sourceFile = file;

  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const arrayBuf = await file.arrayBuffer();
    const decoded = await audioCtx.decodeAudioData(arrayBuf);
    state.sourceAudioBuffer = decoded;
    audioCtx.close();

    displaySource(file);
    updateSteps();
  } catch (e) {
    log("Failed to decode audio: " + e.message, "error");
  }
}

function displaySource(file) {
  $("sourcePanel").style.display = "block";
  $("sourceAudio").src = URL.createObjectURL(file);
  const dur = state.sourceAudioBuffer.duration;
  $("sourceDuration").textContent = `${dur.toFixed(1)}s / ${state.sourceAudioBuffer.sampleRate}Hz / ${state.sourceAudioBuffer.numberOfChannels}ch`;
  drawWaveform($("sourceWaveform"), state.sourceAudioBuffer);
}

function drawWaveform(canvas, audioBuffer) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);

  const data = audioBuffer.getChannelData(0);
  const w = rect.width;
  const h = rect.height;
  const step = Math.ceil(data.length / w);

  ctx.fillStyle = "#18181b";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "#8b5cf6";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < w; i++) {
    const start = i * step;
    let min = 1, max = -1;
    for (let j = 0; j < step && start + j < data.length; j++) {
      const val = data[start + j];
      if (val < min) min = val;
      if (val > max) max = val;
    }
    const y1 = ((1 - max) / 2) * h;
    const y2 = ((1 - min) / 2) * h;
    ctx.moveTo(i, y1);
    ctx.lineTo(i, y2);
  }
  ctx.stroke();
}

function drawMiniWaveform(canvas, audioBuffer) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  canvas.width = 120 * dpr;
  canvas.height = 32 * dpr;
  ctx.scale(dpr, dpr);

  const data = audioBuffer.getChannelData(0);
  const w = 120;
  const h = 32;
  const step = Math.ceil(data.length / w);

  ctx.fillStyle = "#18181b";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "#8b5cf6";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < w; i++) {
    const start = i * step;
    let min = 1, max = -1;
    for (let j = 0; j < step && start + j < data.length; j++) {
      const val = data[start + j];
      if (val < min) min = val;
      if (val > max) max = val;
    }
    const y1 = ((1 - max) / 2) * h;
    const y2 = ((1 - min) / 2) * h;
    ctx.moveTo(i, y1);
    ctx.lineTo(i, y2);
  }
  ctx.stroke();
}

// ---------------------------------------------------------------------------
// Mic recording (Convert tab)
// ---------------------------------------------------------------------------
async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.mediaRecorder = new MediaRecorder(stream);
    state.recordedChunks = [];

    state.mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) state.recordedChunks.push(e.data);
    };
    state.mediaRecorder.onstop = async () => {
      const blob = new Blob(state.recordedChunks, { type: "audio/webm" });
      stream.getTracks().forEach((t) => t.stop());
      $("btnRecord").style.display = "";
      $("btnStopRecord").style.display = "none";
      state.recording = false;
      const file = new File([blob], "recording.webm", { type: blob.type });
      await handleAudioFile(file);
      log("Recording complete.", "success");
    };

    state.mediaRecorder.start();
    state.recording = true;
    $("btnRecord").style.display = "none";
    $("btnStopRecord").style.display = "";
    log("Recording started... click Stop when done.");
  } catch (e) {
    log("Microphone access denied: " + e.message, "error");
  }
}

function stopRecording() {
  if (state.mediaRecorder && state.recording) state.mediaRecorder.stop();
}

// ---------------------------------------------------------------------------
// Training: sample management
// ---------------------------------------------------------------------------
function setupTrainingInput() {
  const dropZone = $("trainDropZone");
  const fileInput = $("trainFileInput");

  dropZone.addEventListener("click", () => fileInput.click());
  dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("dragover"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    for (const file of e.dataTransfer.files) {
      if (file.type.startsWith("audio/") || /\.(wav|mp3|ogg|flac|m4a|webm)$/i.test(file.name)) {
        addTrainingSample(file);
      }
    }
  });
  fileInput.addEventListener("change", (e) => {
    for (const file of e.target.files) {
      addTrainingSample(file);
    }
    fileInput.value = "";
  });

  $("btnTrainRecord").addEventListener("click", startTrainingRecording);
  $("btnTrainStopRecord").addEventListener("click", stopTrainingRecording);
}

async function addTrainingSample(file) {
  log(`Training sample: ${file.name} (${formatBytes(file.size)})`);
  try {
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    const arrayBuf = await file.arrayBuffer();
    const decoded = await audioCtx.decodeAudioData(arrayBuf);
    audioCtx.close();

    const sample = {
      file,
      audioBuffer: decoded,
      name: file.name,
      duration: decoded.duration,
    };

    state.trainingSamples.push(sample);
    renderTrainingSamples();
    updateSteps();
    log(`Added training sample: ${file.name} (${decoded.duration.toFixed(1)}s)`, "success");
  } catch (e) {
    log(`Failed to decode training sample "${file.name}": ${e.message}`, "error");
  }
}

function removeTrainingSample(idx) {
  const removed = state.trainingSamples.splice(idx, 1);
  if (removed.length) log(`Removed training sample: ${removed[0].name}`);
  renderTrainingSamples();
  updateSteps();
}

function renderTrainingSamples() {
  const list = $("trainSampleList");
  list.innerHTML = "";

  if (state.trainingSamples.length === 0) return;

  state.trainingSamples.forEach((sample, idx) => {
    const item = document.createElement("div");
    item.className = "sample-item";

    const canvas = document.createElement("canvas");
    canvas.width = 120;
    canvas.height = 32;
    item.appendChild(canvas);

    const info = document.createElement("div");
    info.className = "sample-info";
    const nameEl = document.createElement("div");
    nameEl.className = "sample-name";
    nameEl.textContent = sample.name;
    const durEl = document.createElement("div");
    durEl.className = "sample-dur";
    durEl.textContent = `${sample.duration.toFixed(1)}s`;
    info.appendChild(nameEl);
    info.appendChild(durEl);
    item.appendChild(info);

    const delBtn = document.createElement("button");
    delBtn.className = "btn-delete";
    delBtn.textContent = "\u00d7";
    delBtn.addEventListener("click", () => removeTrainingSample(idx));
    item.appendChild(delBtn);

    list.appendChild(item);

    // Draw mini waveform after appended to DOM
    requestAnimationFrame(() => drawMiniWaveform(canvas, sample.audioBuffer));
  });
}

// ---------------------------------------------------------------------------
// Training: mic recording
// ---------------------------------------------------------------------------
async function startTrainingRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.trainingMediaRecorder = new MediaRecorder(stream);
    state.trainingRecordedChunks = [];

    state.trainingMediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) state.trainingRecordedChunks.push(e.data);
    };
    state.trainingMediaRecorder.onstop = async () => {
      const blob = new Blob(state.trainingRecordedChunks, { type: "audio/webm" });
      stream.getTracks().forEach((t) => t.stop());
      $("btnTrainRecord").style.display = "";
      $("btnTrainStopRecord").style.display = "none";
      state.trainingRecording = false;
      const recNum = state.trainingSamples.length + 1;
      const file = new File([blob], `recording_${recNum}.webm`, { type: blob.type });
      await addTrainingSample(file);
      log("Training recording complete.", "success");
    };

    state.trainingMediaRecorder.start();
    state.trainingRecording = true;
    $("btnTrainRecord").style.display = "none";
    $("btnTrainStopRecord").style.display = "";
    log("Recording for training... click Stop when done.");
  } catch (e) {
    log("Microphone access denied: " + e.message, "error");
  }
}

function stopTrainingRecording() {
  if (state.trainingMediaRecorder && state.trainingRecording) state.trainingMediaRecorder.stop();
}

// ---------------------------------------------------------------------------
// Audio helpers
// ---------------------------------------------------------------------------
function getMonoFloat32(audioBuffer, targetSampleRate = 16000) {
  let mono;
  if (audioBuffer.numberOfChannels === 1) {
    mono = audioBuffer.getChannelData(0);
  } else {
    mono = new Float32Array(audioBuffer.length);
    for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
      const chData = audioBuffer.getChannelData(ch);
      for (let i = 0; i < mono.length; i++) mono[i] += chData[i] / audioBuffer.numberOfChannels;
    }
  }
  if (audioBuffer.sampleRate !== targetSampleRate) {
    const ratio = targetSampleRate / audioBuffer.sampleRate;
    const newLen = Math.round(mono.length * ratio);
    const resampled = new Float32Array(newLen);
    for (let i = 0; i < newLen; i++) {
      const srcIdx = i / ratio;
      const idx0 = Math.floor(srcIdx);
      const idx1 = Math.min(idx0 + 1, mono.length - 1);
      const frac = srcIdx - idx0;
      resampled[i] = mono[idx0] * (1 - frac) + mono[idx1] * frac;
    }
    return resampled;
  }
  return new Float32Array(mono);
}

// ---------------------------------------------------------------------------
// Feature retrieval (replaces FAISS)
// ---------------------------------------------------------------------------
function retrieveFeatures(features, indexVectors, indexRatio) {
  if (!indexVectors || indexVectors.length === 0 || indexRatio <= 0) return features;
  const dim = features[0].length;
  const result = [];
  for (let i = 0; i < features.length; i++) {
    const query = features[i];
    let bestSim = -Infinity, bestVec = null;
    for (let j = 0; j < indexVectors.length; j++) {
      const vec = indexVectors[j];
      let dot = 0, normQ = 0, normV = 0;
      for (let d = 0; d < dim; d++) {
        dot += query[d] * vec[d];
        normQ += query[d] * query[d];
        normV += vec[d] * vec[d];
      }
      const sim = dot / (Math.sqrt(normQ) * Math.sqrt(normV) + 1e-8);
      if (sim > bestSim) { bestSim = sim; bestVec = vec; }
    }
    const blended = new Float32Array(dim);
    for (let d = 0; d < dim; d++) blended[d] = query[d] * (1 - indexRatio) + bestVec[d] * indexRatio;
    result.push(blended);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Simple F0 estimation (autocorrelation fallback)
// ---------------------------------------------------------------------------
function estimateF0Simple(audioData, sampleRate, hopSize) {
  const frameCount = Math.floor(audioData.length / hopSize);
  const f0 = new Float32Array(frameCount);
  const windowSize = Math.round(sampleRate * 0.04);

  for (let i = 0; i < frameCount; i++) {
    const center = i * hopSize;
    const start = Math.max(0, center - Math.floor(windowSize / 2));
    const end = Math.min(audioData.length, start + windowSize);
    const frame = audioData.subarray(start, end);

    const minLag = Math.floor(sampleRate / 500);
    const maxLag = Math.floor(sampleRate / 50);
    let bestCorr = 0, bestLag = 0;

    for (let lag = minLag; lag < Math.min(maxLag, frame.length); lag++) {
      let corr = 0, e1 = 0, e2 = 0;
      const len = frame.length - lag;
      for (let j = 0; j < len; j++) {
        corr += frame[j] * frame[j + lag];
        e1 += frame[j] * frame[j];
        e2 += frame[j + lag] * frame[j + lag];
      }
      const nCorr = corr / (Math.sqrt(e1 * e2) + 1e-8);
      if (nCorr > bestCorr) { bestCorr = nCorr; bestLag = lag; }
    }
    f0[i] = bestCorr > 0.3 ? sampleRate / bestLag : 0;
  }
  return f0;
}

function shiftF0(f0, semitones) {
  if (semitones === 0) return f0;
  const ratio = Math.pow(2, semitones / 12);
  const shifted = new Float32Array(f0.length);
  for (let i = 0; i < f0.length; i++) shifted[i] = f0[i] > 0 ? f0[i] * ratio : 0;
  return shifted;
}

function f0ToCoarse(f0) {
  const coarse = new BigInt64Array(f0.length);
  for (let i = 0; i < f0.length; i++) {
    if (f0[i] <= 0) { coarse[i] = BigInt(0); }
    else {
      const midi = 12 * Math.log2(f0[i] / 10 + 1e-8);
      coarse[i] = BigInt(Math.round(Math.max(1, Math.min(255, midi))));
    }
  }
  return coarse;
}

function medianFilter(arr, radius) {
  if (radius <= 0) return arr;
  const result = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const win = [];
    for (let j = -radius; j <= radius; j++) {
      win.push(arr[Math.max(0, Math.min(arr.length - 1, i + j))]);
    }
    win.sort((a, b) => a - b);
    result[i] = win[Math.floor(win.length / 2)];
  }
  return result;
}

// ---------------------------------------------------------------------------
// WAV encoder
// ---------------------------------------------------------------------------
function encodeWAV(samples, sampleRate = 40000) {
  const bitsPerSample = 16;
  const byteRate = sampleRate * bitsPerSample / 8;
  const dataSize = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const v = new DataView(buffer);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + dataSize, true); w(8, "WAVE");
  w(12, "fmt "); v.setUint32(16, 16, true); v.setUint16(20, 1, true);
  v.setUint16(22, 1, true); v.setUint32(24, sampleRate, true);
  v.setUint32(28, byteRate, true); v.setUint16(32, 2, true); v.setUint16(34, bitsPerSample, true);
  w(36, "data"); v.setUint32(40, dataSize, true);
  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    off += 2;
  }
  return new Blob([buffer], { type: "audio/wav" });
}

// ---------------------------------------------------------------------------
// Training pipeline
// ---------------------------------------------------------------------------
async function runTraining() {
  const contentVecSession = state.sessions.contentvec;
  if (!contentVecSession) {
    log("ContentVec model not loaded. Cannot train.", "error");
    return;
  }

  if (state.trainingSamples.length === 0) {
    log("No training samples. Add audio files first.", "error");
    return;
  }

  const voiceName = $("voiceName").value.trim() || "Untitled Voice";
  state.trainedVoiceName = voiceName;

  $("btnTrain").disabled = true;
  $("trainResult").style.display = "none";
  log(`Starting voice training for "${voiceName}" with ${state.trainingSamples.length} sample(s)...`);
  showTrainProgress(0, "Preparing samples...");

  const allFeatures = [];
  const allF0Values = [];
  let totalDuration = 0;
  const hopSize = 320;

  try {
    for (let si = 0; si < state.trainingSamples.length; si++) {
      const sample = state.trainingSamples[si];
      const pctBase = (si / state.trainingSamples.length) * 90;
      const pctStep = 90 / state.trainingSamples.length;

      showTrainProgress(pctBase, `Processing sample ${si + 1}/${state.trainingSamples.length}: ${sample.name}...`);
      log(`Training: processing "${sample.name}" (${sample.duration.toFixed(1)}s)...`);

      // 1. Resample to 16kHz mono
      const audio16k = getMonoFloat32(sample.audioBuffer, 16000);
      totalDuration += sample.duration;

      // 2. Run through ContentVec to extract features
      showTrainProgress(pctBase + pctStep * 0.3, `Extracting features from sample ${si + 1}...`);
      const cvInputName = contentVecSession.inputNames[0];
      const cvTensor = new ort.Tensor("float32", audio16k, [1, audio16k.length]);

      let features;
      try {
        const cvResult = await contentVecSession.run({ [cvInputName]: cvTensor });
        const cvOutput = cvResult[contentVecSession.outputNames[0]];
        const dim = cvOutput.dims[2];
        const frames = cvOutput.dims[1];
        const cvData = cvOutput.data;
        features = [];
        for (let i = 0; i < frames; i++) {
          features.push(new Float32Array(cvData.buffer, cvData.byteOffset + i * dim * 4, dim));
        }
        log(`  ContentVec: ${frames} frames, ${dim}-dim.`, "success");
      } catch (e) {
        log(`  ContentVec failed for "${sample.name}": ${e.message}. Generating placeholder.`, "warn");
        const frames = Math.floor(audio16k.length / hopSize);
        features = [];
        for (let i = 0; i < frames; i++) {
          const vec = new Float32Array(768);
          for (let d = 0; d < 768; d++) {
            const sampleIdx = Math.min(i * hopSize + d, audio16k.length - 1);
            vec[d] = audio16k[sampleIdx] * 0.1;
          }
          features.push(vec);
        }
      }

      // 3. F0 estimation for pitch stats
      showTrainProgress(pctBase + pctStep * 0.7, `Estimating pitch for sample ${si + 1}...`);
      let f0;
      if (state.sessions.f0) {
        try {
          const f0In = state.sessions.f0.inputNames[0];
          const f0Tensor = new ort.Tensor("float32", audio16k, [1, audio16k.length]);
          const f0Result = await state.sessions.f0.run({ [f0In]: f0Tensor });
          f0 = new Float32Array(f0Result[state.sessions.f0.outputNames[0]].data);
          log(`  RMVPE: ${f0.length} pitch values.`, "success");
        } catch (e) {
          log(`  RMVPE failed: ${e.message}. Using autocorrelation.`, "warn");
          f0 = estimateF0Simple(audio16k, 16000, hopSize);
        }
      } else {
        f0 = estimateF0Simple(audio16k, 16000, hopSize);
      }

      // Collect features and F0
      for (const feat of features) {
        allFeatures.push(new Float32Array(feat));
      }
      for (let i = 0; i < f0.length; i++) {
        if (f0[i] > 0) allF0Values.push(f0[i]);
      }

      // Allow UI to breathe
      await new Promise(r => setTimeout(r, 10));
    }

    showTrainProgress(92, "Building voice index...");

    // 4. Compute voice statistics
    let avgPitch = 0, minPitch = Infinity, maxPitch = 0;
    if (allF0Values.length > 0) {
      for (const v of allF0Values) {
        avgPitch += v;
        if (v < minPitch) minPitch = v;
        if (v > maxPitch) maxPitch = v;
      }
      avgPitch /= allF0Values.length;
    } else {
      minPitch = 0;
    }

    // 5. Store results
    state.trainedIndex = allFeatures;
    state.index = allFeatures; // Also set as active index for conversion
    state.trainedVoiceStats = {
      avgPitch,
      minPitch: minPitch === Infinity ? 0 : minPitch,
      maxPitch,
      totalDuration,
      vectorCount: allFeatures.length,
    };

    // Update index slot visually
    const statusIndex = $("statusIndex");
    const sizeIndex = $("sizeIndex");
    const slotIndex = $("slotIndex");
    statusIndex.textContent = "Trained";
    statusIndex.className = "status ok";
    sizeIndex.textContent = `${allFeatures.length} vectors from training`;
    slotIndex.classList.add("loaded");

    showTrainProgress(100, "Training complete!");

    // 6. Show results
    $("trainStatVectors").textContent = allFeatures.length.toLocaleString();
    $("trainStatDuration").textContent = `${totalDuration.toFixed(1)}s`;
    $("trainStatPitch").textContent = avgPitch > 0 ? `${avgPitch.toFixed(0)} Hz` : "N/A";
    $("trainStatRange").textContent = (minPitch > 0 && maxPitch > 0) ? `${minPitch.toFixed(0)}-${maxPitch.toFixed(0)} Hz` : "N/A";
    $("trainResult").style.display = "block";

    log(`Training complete! ${allFeatures.length} feature vectors from ${state.trainingSamples.length} sample(s), ${totalDuration.toFixed(1)}s total.`, "success");
    log(`Voice: "${voiceName}" | Avg pitch: ${avgPitch.toFixed(0)}Hz | Range: ${minPitch.toFixed(0)}-${maxPitch.toFixed(0)}Hz`, "success");

    setTimeout(hideTrainProgress, 2000);

  } catch (e) {
    log(`Training error: ${e.message}`, "error");
    console.error(e);
    hideTrainProgress();
  }

  $("btnTrain").disabled = false;
  updateSteps();
}

// ---------------------------------------------------------------------------
// Voice profile download/upload
// ---------------------------------------------------------------------------
function downloadVoiceProfile() {
  const index = state.trainedIndex || state.index;
  if (!index || index.length === 0) {
    log("No voice index to download.", "warn");
    return;
  }

  const profile = {
    format: "rvc-voice-profile",
    version: 1,
    voiceName: state.trainedVoiceName || "Untitled Voice",
    stats: state.trainedVoiceStats || null,
    index: index.map(v => Array.from(v)),
    exportedAt: new Date().toISOString(),
  };

  const json = JSON.stringify(profile);
  const blob = new Blob([json], { type: "application/json" });
  const safeName = (state.trainedVoiceName || "voice").replace(/[^a-z0-9_-]/gi, "_").toLowerCase();
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${safeName}_profile.json`;
  a.click();
  log(`Voice profile downloaded: ${safeName}_profile.json (${formatBytes(blob.size)})`, "success");
}

// ---------------------------------------------------------------------------
// RVC conversion pipeline
// ---------------------------------------------------------------------------
async function runConversion() {
  const params = {
    pitchShift: parseInt($("pitchShift").value),
    indexRatio: parseInt($("indexRatio").value) / 100,
    filterRadius: parseInt($("filterRadius").value),
    volEnvelope: parseInt($("volEnvelope").value) / 100,
    protect: parseInt($("protect").value) / 100,
    chunkSize: parseInt($("chunkSize").value),
    useF0: $("useF0").checked,
  };

  const audioBuffer = state.sourceAudioBuffer;
  if (!audioBuffer) { log("No audio loaded.", "error"); return; }

  const contentVecSession = state.sessions.contentvec;
  const generatorSession = state.sessions.generator;
  const hasIndex = (state.trainedIndex && state.trainedIndex.length > 0) || (state.index && state.index.length > 0);

  if (!contentVecSession) {
    log("ContentVec model is required.", "error");
    return;
  }
  if (!generatorSession && !hasIndex) {
    log("Need either a generator model (.onnx) or a trained voice index.", "error");
    return;
  }

  const indexOnlyMode = !generatorSession && hasIndex;
  if (indexOnlyMode) {
    log("Index-only mode: no generator loaded, using voice index for conversion.", "info");
  }

  $("btnConvert").disabled = true;
  log("Starting voice conversion...");
  showProgress(0, "Preparing audio...");

  try {
    // 1. Prepare audio
    const audio16k = getMonoFloat32(audioBuffer, 16000);
    const totalSamples = audio16k.length;
    log(`Audio: ${(totalSamples / 16000).toFixed(1)}s at 16kHz, ${totalSamples} samples`);
    showProgress(5, "Extracting features with ContentVec...");

    // 2. ContentVec feature extraction
    log("Running ContentVec feature extraction...");
    const cvInputName = contentVecSession.inputNames[0];
    const cvTensor = new ort.Tensor("float32", audio16k, [1, audio16k.length]);

    let features;
    try {
      const cvResult = await contentVecSession.run({ [cvInputName]: cvTensor });
      const cvOutput = cvResult[contentVecSession.outputNames[0]];
      const dim = cvOutput.dims[2];
      const frames = cvOutput.dims[1];
      const cvData = cvOutput.data;
      features = [];
      for (let i = 0; i < frames; i++) {
        features.push(new Float32Array(cvData.buffer, cvData.byteOffset + i * dim * 4, dim));
      }
      log(`ContentVec: ${frames} frames, ${dim}-dim features.`, "success");
    } catch (e) {
      log(`ContentVec inference error: ${e.message}`, "error");
      log("Generating placeholder features for pipeline test...", "warn");
      const hopSize = 320;
      const frames = Math.floor(totalSamples / hopSize);
      features = [];
      for (let i = 0; i < frames; i++) {
        const vec = new Float32Array(768);
        for (let d = 0; d < 768; d++) {
          const si = Math.min(i * hopSize + d, totalSamples - 1);
          vec[d] = audio16k[si] * 0.1;
        }
        features.push(vec);
      }
    }

    showProgress(30, "Estimating pitch...");

    // 3. F0 estimation
    const hopSize = 320;
    let f0;
    if (state.sessions.f0 && params.useF0) {
      log("Running RMVPE pitch estimation...");
      try {
        const f0In = state.sessions.f0.inputNames[0];
        const f0Tensor = new ort.Tensor("float32", audio16k, [1, audio16k.length]);
        const f0Result = await state.sessions.f0.run({ [f0In]: f0Tensor });
        f0 = new Float32Array(f0Result[state.sessions.f0.outputNames[0]].data);
        log(`RMVPE: ${f0.length} pitch values.`, "success");
      } catch (e) {
        log(`RMVPE failed: ${e.message}. Using autocorrelation.`, "warn");
        f0 = estimateF0Simple(audio16k, 16000, hopSize);
      }
    } else {
      log(params.useF0 ? "RMVPE not loaded, using autocorrelation fallback." : "F0 guidance disabled.");
      f0 = params.useF0 ? estimateF0Simple(audio16k, 16000, hopSize) : new Float32Array(features.length);
    }

    // Match lengths
    if (f0.length !== features.length) {
      const newF0 = new Float32Array(features.length);
      for (let i = 0; i < features.length; i++) {
        newF0[i] = f0[Math.min(Math.floor(i * f0.length / features.length), f0.length - 1)];
      }
      f0 = newF0;
    }

    f0 = shiftF0(f0, params.pitchShift);
    f0 = medianFilter(f0, params.filterRadius);
    log(`F0: range ${Math.min(...f0).toFixed(0)}-${Math.max(...f0).toFixed(0)} Hz`);
    showProgress(50, "Feature retrieval...");

    // 4. Feature retrieval
    const activeIndex = state.index || state.trainedIndex;
    if (activeIndex && activeIndex.length > 0 && params.indexRatio > 0) {
      log(`Retrieving from index (${activeIndex.length} vectors, ratio=${params.indexRatio})...`);
      features = retrieveFeatures(features.map(f => new Float32Array(f)), activeIndex, params.indexRatio);
      log("Feature retrieval complete.", "success");
    }

    let outputAudio;

    if (indexOnlyMode) {
      // Index-only conversion: synthesize from features + F0 using simple additive synthesis
      showProgress(65, "Synthesizing (index-only mode)...");
      log("Running index-only synthesis (no generator)...");

      const outSR = 16000;
      const outLen = audio16k.length;
      outputAudio = new Float32Array(outLen);

      // Use the retrieved features to modulate the original audio.
      // For each frame, scale the source signal by the feature energy ratio.
      const frameHop = hopSize;
      for (let i = 0; i < features.length; i++) {
        const frameStart = i * frameHop;
        const frameEnd = Math.min(frameStart + frameHop, outLen);

        // Compute feature energy as a soft spectral shaping factor
        let featEnergy = 0;
        const feat = features[i];
        for (let d = 0; d < feat.length; d++) featEnergy += feat[d] * feat[d];
        featEnergy = Math.sqrt(featEnergy / feat.length);

        // Pitch-guided synthesis: generate a pitched signal using F0
        const pitch = f0[i];
        if (pitch > 0) {
          const period = outSR / pitch;
          for (let j = frameStart; j < frameEnd; j++) {
            // Blend source signal with a pitch-guided harmonic
            const phase = (j % Math.round(period)) / period;
            const harmonic = Math.sin(2 * Math.PI * phase) * 0.3;
            const srcVal = j < audio16k.length ? audio16k[j] : 0;
            // Mix: use features to blend between original timbre and index timbre
            outputAudio[j] = srcVal * (1 - params.indexRatio * 0.5) + harmonic * featEnergy * params.indexRatio * 0.5;
          }
        } else {
          // Unvoiced: pass through source with slight dampening
          for (let j = frameStart; j < frameEnd; j++) {
            outputAudio[j] = j < audio16k.length ? audio16k[j] * 0.8 : 0;
          }
        }
      }

      log(`Index-only synthesis complete: ${outLen} samples.`, "success");

    } else {
      // Standard generator-based conversion
      showProgress(65, "Running voice synthesis (net_g)...");
      log("Running RVC generator...");
      const T = features.length;
      const dim = features[0].length;

      const phoneData = new Float32Array(T * dim);
      for (let i = 0; i < T; i++) phoneData.set(features[i], i * dim);

      const pitchCoarse = f0ToCoarse(f0);
      const pitchF = new Float32Array(f0);

      const rndData = new Float32Array(192 * T);
      for (let i = 0; i < rndData.length; i++) {
        const u1 = Math.random() || 1e-8;
        const u2 = Math.random();
        rndData[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }

      const genInputNames = generatorSession.inputNames;
      log(`Generator inputs: [${genInputNames.join(", ")}]`);

      const feeds = {};
      for (const name of genInputNames) {
        const n = name.toLowerCase();
        if (n.includes("phone") && !n.includes("length")) feeds[name] = new ort.Tensor("float32", phoneData, [1, T, dim]);
        else if (n.includes("phone") && n.includes("length")) feeds[name] = new ort.Tensor("int64", new BigInt64Array([BigInt(T)]), [1]);
        else if (n === "pitch" || (n.includes("pitch") && !n.includes("f"))) feeds[name] = new ort.Tensor("int64", pitchCoarse, [1, T]);
        else if (n === "pitchf" || n.includes("pitchf") || n.includes("nsf")) feeds[name] = new ort.Tensor("float32", pitchF, [1, T]);
        else if (n === "ds" || n.includes("speaker") || n.includes("sid")) feeds[name] = new ort.Tensor("int64", new BigInt64Array([BigInt(0)]), [1]);
        else if (n === "rnd" || n.includes("rand") || n.includes("noise")) feeds[name] = new ort.Tensor("float32", rndData, [1, 192, T]);
        else {
          log(`Unknown generator input "${name}" — providing zeros.`, "warn");
          feeds[name] = new ort.Tensor("float32", new Float32Array(T), [1, T]);
        }
      }

      try {
        const genResult = await generatorSession.run(feeds);
        outputAudio = new Float32Array(genResult[generatorSession.outputNames[0]].data);
        log(`Generator output: ${outputAudio.length} samples`, "success");
      } catch (e) {
        log(`Generator inference failed: ${e.message}`, "error");
        log("Passing through original audio for demonstration.", "warn");
        outputAudio = new Float32Array(audio16k);
      }
    }

    showProgress(90, "Post-processing...");

    // 6. Volume envelope
    if (params.volEnvelope > 0 && params.volEnvelope < 1) {
      const bs = 512;
      for (let i = 0; i < outputAudio.length; i += bs) {
        const end = Math.min(i + bs, outputAudio.length);
        let srcRms = 0, outRms = 0;
        const ss = Math.floor(i * audio16k.length / outputAudio.length);
        const se = Math.min(ss + bs, audio16k.length);
        for (let j = ss; j < se; j++) srcRms += audio16k[j] * audio16k[j];
        srcRms = Math.sqrt(srcRms / (se - ss + 1e-8));
        for (let j = i; j < end; j++) outRms += outputAudio[j] * outputAudio[j];
        outRms = Math.sqrt(outRms / (end - i + 1e-8));
        if (outRms > 1e-6) {
          const gain = params.volEnvelope * (srcRms / outRms) + (1 - params.volEnvelope);
          for (let j = i; j < end; j++) outputAudio[j] *= gain;
        }
      }
    }

    // Normalize
    let maxAbs = 0;
    for (let i = 0; i < outputAudio.length; i++) { const a = Math.abs(outputAudio[i]); if (a > maxAbs) maxAbs = a; }
    if (maxAbs > 0) { const s = 0.95 / maxAbs; for (let i = 0; i < outputAudio.length; i++) outputAudio[i] *= s; }

    showProgress(95, "Encoding WAV...");

    // 7. Output
    const outSR = (!indexOnlyMode && outputAudio.length > audio16k.length * 2) ? 40000 : 16000;
    const wavBlob = encodeWAV(outputAudio, outSR);
    state.outputBuffer = outputAudio;
    state.outputBlob = wavBlob;

    $("outputPanel").style.display = "block";
    $("outputAudio").src = URL.createObjectURL(wavBlob);
    $("outputDuration").textContent = `${(outputAudio.length / outSR).toFixed(1)}s / ${outSR}Hz / ${formatBytes(wavBlob.size)}`;

    const outCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: outSR });
    const outAB = outCtx.createBuffer(1, outputAudio.length, outSR);
    outAB.getChannelData(0).set(outputAudio);
    drawWaveform($("outputWaveform"), outAB);
    outCtx.close();

    showProgress(100, "Done!");
    log(`Conversion complete! Output: ${(outputAudio.length / outSR).toFixed(1)}s${indexOnlyMode ? " (index-only mode)" : ""}`, "success");
    setTimeout(hideProgress, 2000);

  } catch (e) {
    log(`Pipeline error: ${e.message}`, "error");
    console.error(e);
    hideProgress();
  }

  $("btnConvert").disabled = false;
  updateSteps();
}

// ---------------------------------------------------------------------------
// Downloads
// ---------------------------------------------------------------------------
function downloadAudio() {
  if (!state.outputBlob) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(state.outputBlob);
  a.download = "rvc_converted.wav";
  a.click();
  log("Converted audio downloaded.", "success");
}

async function downloadBundle() {
  log("Preparing voice model bundle...");
  const downloads = [];

  // Include auto-downloaded base models (.onnx)
  if (state.modelBuffers.contentvec) {
    downloads.push({ name: state.modelNames.contentvec || "contentvec.onnx", buffer: state.modelBuffers.contentvec });
  }
  if (state.modelBuffers.f0) {
    downloads.push({ name: state.modelNames.f0 || "rmvpe.onnx", buffer: state.modelBuffers.f0 });
  }
  // Include user's voice generator model if loaded
  if (state.modelBuffers.generator) {
    downloads.push({ name: state.modelNames.generator || "generator.onnx", buffer: state.modelBuffers.generator });
  }
  // Include voice index
  const activeIndex = state.index || state.trainedIndex;
  if (activeIndex) {
    const json = JSON.stringify(activeIndex.map(v => Array.from(v)));
    downloads.push({ name: state.indexName || "voice_index.json", blob: new Blob([json], { type: "application/json" }) });
  }

  if (downloads.length === 0) {
    log("No voice model to download.", "warn");
    return;
  }

  // Include a manifest with parameter settings
  const manifest = {
    format: "rvc-browser-bundle",
    version: 1,
    models: downloads.map(d => d.name),
    params: {
      pitchShift: parseInt($("pitchShift").value),
      indexRatio: parseInt($("indexRatio").value) / 100,
      filterRadius: parseInt($("filterRadius").value),
      volEnvelope: parseInt($("volEnvelope").value) / 100,
      protect: parseInt($("protect").value) / 100,
    },
    exportedAt: new Date().toISOString(),
  };

  for (const dl of downloads) {
    const blob = dl.blob || new Blob([dl.buffer]);
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = dl.name;
    a.click();
    await new Promise(r => setTimeout(r, 500));
  }

  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(manifest, null, 2)], { type: "application/json" }));
  a.download = "rvc_bundle_manifest.json";
  a.click();

  log(`Bundle: ${downloads.length} file(s) + manifest downloaded.`, "success");
}

// ---------------------------------------------------------------------------
// Param slider displays
// ---------------------------------------------------------------------------
function setupParamDisplays() {
  for (const [id, cfg] of Object.entries(paramDisplays)) {
    $(id).addEventListener("input", () => { cfg.el.textContent = cfg.fmt($(id).value); });
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  await checkWebGPU();

  if (typeof ort !== "undefined") {
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
    log(`ONNX Runtime Web loaded. WASM threads: ${ort.env.wasm.numThreads}`);
  } else {
    log("ONNX Runtime Web not loaded! Check your connection.", "error");
    return;
  }

  // Tab switching
  $("tabBtnTrain").addEventListener("click", () => switchTab("train"));
  $("tabBtnConvert").addEventListener("click", () => switchTab("convert"));

  // Setup manual upload for generator + index
  setupModelSlot("slotGenerator", "fileGenerator", "statusGenerator", "sizeGenerator", "generator");
  setupModelSlot("slotIndex", "fileIndex", "statusIndex", "sizeIndex", "index");

  // Auto-load slots still allow manual override
  setupModelSlot("slotContentVec", "fileContentVec", "statusContentVec", "sizeContentVec", "contentvec");
  setupModelSlot("slotF0", "fileF0", "statusF0", "sizeF0", "f0");

  // Convert tab audio input
  setupAudioInput();

  // Training tab input
  setupTrainingInput();

  setupParamDisplays();

  // Convert tab buttons
  $("btnConvert").addEventListener("click", runConversion);
  $("dlAudio").addEventListener("click", downloadAudio);
  $("dlBundle").addEventListener("click", downloadBundle);

  // Training buttons
  $("btnTrain").addEventListener("click", runTraining);
  $("btnDownloadProfile").addEventListener("click", downloadVoiceProfile);
  $("dlVoiceProfile").addEventListener("click", downloadVoiceProfile);
  $("btnUseProfile").addEventListener("click", () => {
    switchTab("convert");
    log("Switched to Convert tab. Your trained voice index is active.", "success");
  });

  updateSteps();

  // Auto-download universal models
  log("Auto-downloading base models from HuggingFace...");
  log("(These are cached in your browser after the first download.)");

  // Download both in parallel
  await Promise.all([
    autoLoadModel("contentvec", MODEL_URLS.contentvec, "statusContentVec", "progContentVec", "slotContentVec", "sizeContentVec"),
    autoLoadModel("f0", MODEL_URLS.f0, "statusF0", "progF0", "slotF0", "sizeF0"),
  ]);

  log("Base models ready. Train a voice in the Train tab, or upload an RVC model (.onnx) to convert voices.", "success");
  updateSteps();
}

init();
