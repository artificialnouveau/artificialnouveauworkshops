const FACE_SIZE = 300;
const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.14/model/";

const $ = (sel) => document.querySelector(sel);

let modelsLoaded = false;
let allEntries = []; // { img, faces[], included }
let alignedFaces = []; // canvas elements for each aligned face
let allAges = []; // estimated age per detected face

// ── Initialise ──────────────────────────────────────────────

async function init() {
  const status = $("#status");

  try {
    // Load models and images in parallel
    const modelPromise = Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
    ]);

    // Start loading images immediately
    status.textContent = "Loading models and images…";
    const imagePromise = loadPresetImages();

    await modelPromise;
    modelsLoaded = true;

    await imagePromise;

    if (allEntries.length > 0) {
      status.textContent = `Ready — ${allEntries.length} photos loaded. Click "Analyze All Images" to detect faces.`;
      status.classList.add("ready");
      $("#analyze-btn").disabled = false;
    } else {
      status.textContent = "Models loaded — upload photos to get started";
      status.classList.add("ready");
    }

    $("#drop-zone").classList.remove("hidden");
    setupDragDrop();
  } catch (err) {
    status.textContent = "Error loading: " + err.message;
    status.classList.add("error");
    console.error(err);
  }
}

// ── Load preset images from image-list.json ─────────────────

async function loadPresetImages() {
  try {
    const resp = await fetch("./image-list.json");
    if (!resp.ok) return;
    const imageNames = await resp.json();
    if (!imageNames.length) return;

    showProgress(0);

    for (let i = 0; i < imageNames.length; i++) {
      const url = `./images/${imageNames[i]}`;
      try {
        const img = await loadImage(url);
        allEntries.push({ img, faces: [], included: true });
      } catch {
        console.warn("Failed to load", url);
      }
      showProgress((i + 1) / imageNames.length);
    }

    hideProgress();
    renderGallery();
    updateButtons();
  } catch {
    // image-list.json not available — that's fine
  }
}

// ── Drag & Drop ─────────────────────────────────────────────

function setupDragDrop() {
  const dropZone = $("#drop-zone");
  const fileInput = $("#file-input");

  dropZone.addEventListener("click", (e) => {
    if (e.target === fileInput) return;
    fileInput.click();
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("drag-over");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith("image/"));
    if (files.length) addFiles(files);
  });

  fileInput.addEventListener("change", () => {
    const files = [...fileInput.files];
    if (files.length) addFiles(files);
    fileInput.value = "";
  });
}

async function addFiles(files) {
  const status = $("#status");
  status.textContent = `Loading ${files.length} file(s)…`;
  status.className = "status";
  showProgress(0);

  for (let i = 0; i < files.length; i++) {
    const img = await loadImageFromFile(files[i]);
    allEntries.push({ img, faces: [], included: true });
    showProgress((i + 1) / files.length);
  }

  hideProgress();
  renderGallery();
  updateButtons();
  status.textContent = `${allEntries.length} photos loaded. Click "Analyze All Images" to detect faces.`;
  status.classList.add("ready");
}

// ── Image loading helpers ───────────────────────────────────

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = url;
  });
}

function loadImageFromFile(file) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(img.src);
      resolve(img);
    };
    img.src = URL.createObjectURL(file);
  });
}

// ── Gallery rendering ───────────────────────────────────────

function renderGallery() {
  const gallery = $("#gallery");
  gallery.innerHTML = "";
  $("#gallery-title").classList.toggle("hidden", allEntries.length === 0);

  allEntries.forEach((entry, idx) => {
    const div = document.createElement("div");
    div.className = "gallery-item" + (entry.included ? "" : " no-face");
    div.dataset.index = idx;

    const img = document.createElement("img");
    img.src = entry.img.src;
    img.alt = `Photo ${idx + 1}`;
    div.appendChild(img);

    if (entry.faces.length > 0) {
      const badge = document.createElement("span");
      badge.className = "face-count";
      badge.textContent = `${entry.faces.length} face${entry.faces.length > 1 ? "s" : ""}`;
      div.appendChild(badge);
    } else if (entry._detected) {
      const badge = document.createElement("span");
      badge.className = "face-count zero";
      badge.textContent = "0 faces";
      div.appendChild(badge);
    }

    div.addEventListener("click", () => {
      entry.included = !entry.included;
      div.classList.toggle("no-face", !entry.included);
    });

    gallery.appendChild(div);
  });
}

function updateButtons() {
  const hasEntries = allEntries.length > 0;
  $("#analyze-btn").disabled = !hasEntries || !modelsLoaded;
  $("#clear-btn").disabled = !hasEntries;
}

// ── Analyze All Images: detect faces + generate average ─────

async function analyzeAll() {
  if (!modelsLoaded || allEntries.length === 0) return;

  const status = $("#status");
  const btn = $("#analyze-btn");
  btn.disabled = true;

  // Step 1: Detect faces
  status.textContent = "Detecting faces…";
  status.className = "status";
  showProgress(0);

  alignedFaces = [];
  allAges = [];
  let totalFaces = 0;

  for (let i = 0; i < allEntries.length; i++) {
    const entry = allEntries[i];
    const detections = await faceapi
      .detectAllFaces(entry.img)
      .withFaceLandmarks()
      .withAgeAndGender();

    entry.faces = detections;
    entry._detected = true;

    for (const det of detections) {
      const aligned = alignFace(entry.img, det.landmarks);
      alignedFaces.push(aligned);
      allAges.push(Math.round(det.age));
      totalFaces++;
    }

    showProgress((i + 1) / allEntries.length);
  }

  hideProgress();

  // Remove photos without faces
  const totalBefore = allEntries.length;
  allEntries = allEntries.filter((e) => e.faces.length > 0);

  // Rebuild aligned faces and ages arrays to match filtered entries
  alignedFaces = [];
  allAges = [];
  totalFaces = 0;
  for (const entry of allEntries) {
    for (const det of entry.faces) {
      const aligned = alignFace(entry.img, det.landmarks);
      alignedFaces.push(aligned);
      allAges.push(Math.round(det.age));
      totalFaces++;
    }
  }

  renderGallery();
  renderFaceStrip();

  const removed = totalBefore - allEntries.length;
  const avgAge = allAges.length > 0 ? (allAges.reduce((a, b) => a + b, 0) / allAges.length).toFixed(1) : "N/A";
  status.textContent = `Found ${totalFaces} face(s) in ${allEntries.length} photo(s). Removed ${removed} without faces. Generating average…`;

  // Step 2: Generate average face
  generateAverageFace();

  status.textContent = `Done — ${totalFaces} faces from ${allEntries.length} photos | Average age: ${avgAge} years`;
  status.classList.add("ready");
  btn.disabled = false;
}

// ── Face alignment using similarity transform ───────────────

function alignFace(img, landmarks) {
  const pts = landmarks.positions;

  // Eye centres (landmarks 36-41 = left eye, 42-47 = right eye)
  const leftEye = avgPoint(pts.slice(36, 42));
  const rightEye = avgPoint(pts.slice(42, 48));

  // Target positions in the output (normalised to FACE_SIZE)
  const targetLeftEye = { x: FACE_SIZE * 0.3, y: FACE_SIZE * 0.35 };
  const targetRightEye = { x: FACE_SIZE * 0.7, y: FACE_SIZE * 0.35 };

  // Compute similarity transform (rotation + uniform scale + translation)
  const angle = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);
  const targetAngle = Math.atan2(
    targetRightEye.y - targetLeftEye.y,
    targetRightEye.x - targetLeftEye.x
  );
  const rot = targetAngle - angle;

  const dist = Math.hypot(rightEye.x - leftEye.x, rightEye.y - leftEye.y);
  const targetDist = Math.hypot(
    targetRightEye.x - targetLeftEye.x,
    targetRightEye.y - targetLeftEye.y
  );
  const scale = targetDist / dist;

  const canvas = document.createElement("canvas");
  canvas.width = FACE_SIZE;
  canvas.height = FACE_SIZE;
  const ctx = canvas.getContext("2d");

  ctx.save();
  ctx.translate(targetLeftEye.x, targetLeftEye.y);
  ctx.rotate(rot);
  ctx.scale(scale, scale);
  ctx.translate(-leftEye.x, -leftEye.y);
  ctx.drawImage(img, 0, 0);
  ctx.restore();

  return canvas;
}

function avgPoint(points) {
  let x = 0, y = 0;
  for (const p of points) { x += p.x; y += p.y; }
  return { x: x / points.length, y: y / points.length };
}

// ── Render face strip ───────────────────────────────────────

function renderFaceStrip() {
  const section = $("#faces-section");
  const strip = $("#face-strip");
  strip.innerHTML = "";

  if (alignedFaces.length === 0) {
    section.classList.add("hidden");
    return;
  }

  section.classList.remove("hidden");

  for (const faceCanvas of alignedFaces) {
    const display = document.createElement("canvas");
    display.width = 80;
    display.height = 80;
    const ctx = display.getContext("2d");
    ctx.drawImage(faceCanvas, 0, 0, 80, 80);
    strip.appendChild(display);
  }
}

// ── Generate average face ───────────────────────────────────

function generateAverageFace() {
  const includedFaces = [];
  let faceIdx = 0;
  for (const entry of allEntries) {
    for (let j = 0; j < entry.faces.length; j++) {
      if (entry.included) {
        includedFaces.push(alignedFaces[faceIdx]);
      }
      faceIdx++;
    }
  }

  if (includedFaces.length === 0) {
    $("#status").textContent = "No faces found in included photos";
    $("#status").className = "status error";
    return;
  }

  const resultCanvas = $("#average-canvas");
  resultCanvas.width = FACE_SIZE;
  resultCanvas.height = FACE_SIZE;
  const ctx = resultCanvas.getContext("2d");

  const accum = new Float64Array(FACE_SIZE * FACE_SIZE * 4);
  const count = includedFaces.length;

  for (const faceCanvas of includedFaces) {
    const tempCtx = faceCanvas.getContext("2d");
    const data = tempCtx.getImageData(0, 0, FACE_SIZE, FACE_SIZE).data;
    for (let i = 0; i < data.length; i++) {
      accum[i] += data[i];
    }
  }

  const output = ctx.createImageData(FACE_SIZE, FACE_SIZE);
  for (let i = 0; i < accum.length; i++) {
    output.data[i] = Math.round(accum[i] / count);
  }
  for (let i = 3; i < output.data.length; i += 4) {
    output.data[i] = 255;
  }
  ctx.putImageData(output, 0, 0);

  $("#result-section").classList.remove("hidden");
  const estAge = allAges.length > 0 ? (allAges.reduce((a, b) => a + b, 0) / allAges.length).toFixed(1) : "N/A";
  $("#result-meta").textContent = `Averaged ${count} face(s) from ${allEntries.filter((e) => e.included).length} photo(s) | Estimated average age: ${estAge}`;
}

// ── Clear all ───────────────────────────────────────────────

function clearAll() {
  allEntries = [];
  alignedFaces = [];
  allAges = [];
  $("#gallery").innerHTML = "";
  $("#gallery-title").classList.add("hidden");
  $("#face-strip").innerHTML = "";
  $("#faces-section").classList.add("hidden");
  $("#result-section").classList.add("hidden");
  updateButtons();
  $("#status").textContent = "Cleared — upload photos to start again";
  $("#status").className = "status ready";
}

// ── Progress bar ────────────────────────────────────────────

function showProgress(frac) {
  const wrap = $("#progress-wrap");
  const bar = $("#progress-bar");
  wrap.classList.remove("hidden");
  bar.style.width = (frac * 100).toFixed(1) + "%";
}

function hideProgress() {
  $("#progress-wrap").classList.add("hidden");
  $("#progress-bar").style.width = "0%";
}

// ── Event bindings ──────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  init();
  $("#analyze-btn").addEventListener("click", analyzeAll);
  $("#clear-btn").addEventListener("click", clearAll);
});
