const FACE_SIZE = 300; // output face size in pixels
const MODEL_URL = "https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.14/model/";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

let modelsLoaded = false;
let allEntries = []; // { img, file?, url?, faces[], included }
let alignedFaces = []; // { canvas, landmarks } for each detected face

// ── Initialise ──────────────────────────────────────────────

async function init() {
  const status = $("#status");

  try {
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    ]);
    modelsLoaded = true;
    status.textContent = "Models loaded — upload photos to get started";
    status.classList.add("ready");

    // Check if pre-downloaded images are available (local dev only)
    try {
      const resp = await fetch("./image-list.json");
      if (resp.ok) {
        const list = await resp.json();
        if (list.length > 0) {
          $("#load-preset-btn").disabled = false;
          status.textContent = `Models loaded — ${list.length} Instagram photos available, or upload your own`;
        }
      }
    } catch {}

    $("#drop-zone").classList.remove("hidden");
    setupDragDrop();
  } catch (err) {
    status.textContent = "Error loading models: " + err.message;
    status.classList.add("error");
    console.error(err);
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

// ── Load preset images from image-list.json ─────────────────

async function loadPresetImages() {
  const status = $("#status");
  const btn = $("#load-preset-btn");
  btn.disabled = true;
  status.textContent = "Loading image list…";
  status.className = "status";

  try {
    const resp = await fetch("./image-list.json");
    if (!resp.ok) throw new Error("image-list.json not found. Run download_images.py first.");
    const imageNames = await resp.json();

    if (!imageNames.length) {
      throw new Error("image-list.json is empty.");
    }

    status.textContent = `Loading ${imageNames.length} images…`;
    showProgress(0);

    for (let i = 0; i < imageNames.length; i++) {
      const url = `./images/${imageNames[i]}`;
      try {
        const img = await loadImage(url);
        allEntries.push({ img, url, faces: [], included: true });
      } catch {
        console.warn("Failed to load", url);
      }
      showProgress((i + 1) / imageNames.length);
    }

    hideProgress();
    renderGallery();
    await detectAllFaces();
  } catch (err) {
    status.textContent = err.message;
    status.classList.add("error");
    btn.disabled = false;
  }
}

// ── Add user-uploaded files ─────────────────────────────────

async function addFiles(files) {
  const status = $("#status");
  status.textContent = `Loading ${files.length} file(s)…`;
  status.className = "status";
  showProgress(0);

  for (let i = 0; i < files.length; i++) {
    const img = await loadImageFromFile(files[i]);
    allEntries.push({ img, file: files[i], faces: [], included: true });
    showProgress((i + 1) / files.length);
  }

  hideProgress();
  renderGallery();
  await detectAllFaces();
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
  $("#gallery-title").classList.remove("hidden");

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
    } else if (entry.faces._detected) {
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

  updateButtons();
}

function updateButtons() {
  const hasFaces = alignedFaces.length > 0;
  const hasEntries = allEntries.length > 0;
  $("#generate-btn").disabled = !hasFaces;
  $("#clear-btn").disabled = !hasEntries;
}

// ── Face detection ──────────────────────────────────────────

async function detectAllFaces() {
  const status = $("#status");
  status.textContent = "Detecting faces…";
  status.className = "status";
  showProgress(0);

  alignedFaces = [];
  let totalFaces = 0;

  for (let i = 0; i < allEntries.length; i++) {
    const entry = allEntries[i];
    const detections = await faceapi
      .detectAllFaces(entry.img)
      .withFaceLandmarks();

    entry.faces = detections;
    entry.faces._detected = true;

    for (const det of detections) {
      const aligned = alignFace(entry.img, det.landmarks);
      alignedFaces.push(aligned);
      totalFaces++;
    }

    showProgress((i + 1) / allEntries.length);
  }

  hideProgress();
  renderGallery();
  renderFaceStrip();

  status.textContent = `Found ${totalFaces} face(s) in ${allEntries.length} photo(s)`;
  status.classList.add("ready");
}

// ── Face alignment using affine transform ───────────────────

function alignFace(img, landmarks) {
  const pts = landmarks.positions;

  // Eye centres (landmarks 36-41 = left eye, 42-47 = right eye)
  const leftEye = avgPoint(pts.slice(36, 42));
  const rightEye = avgPoint(pts.slice(42, 48));
  const nose = pts[30]; // tip of nose
  const mouthLeft = pts[48];
  const mouthRight = pts[54];

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

  // Transform: translate to target left eye, then rotate+scale around source left eye
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
  const n = points.length;
  let x = 0,
    y = 0;
  for (const p of points) {
    x += p.x;
    y += p.y;
  }
  return { x: x / n, y: y / n };
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
  // Only use faces from included entries
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
    $("#status").textContent = "No included faces to average";
    $("#status").className = "status error";
    return;
  }

  const resultCanvas = $("#average-canvas");
  resultCanvas.width = FACE_SIZE;
  resultCanvas.height = FACE_SIZE;
  const ctx = resultCanvas.getContext("2d");

  // Accumulate pixel values
  const accum = new Float64Array(FACE_SIZE * FACE_SIZE * 4);
  const count = includedFaces.length;

  for (const faceCanvas of includedFaces) {
    const tempCtx = faceCanvas.getContext("2d");
    const data = tempCtx.getImageData(0, 0, FACE_SIZE, FACE_SIZE).data;
    for (let i = 0; i < data.length; i++) {
      accum[i] += data[i];
    }
  }

  // Write average
  const output = ctx.createImageData(FACE_SIZE, FACE_SIZE);
  for (let i = 0; i < accum.length; i++) {
    output.data[i] = Math.round(accum[i] / count);
  }
  // Force full opacity
  for (let i = 3; i < output.data.length; i += 4) {
    output.data[i] = 255;
  }
  ctx.putImageData(output, 0, 0);

  $("#result-section").classList.remove("hidden");
  $("#result-meta").textContent = `Averaged ${count} face(s) from ${allEntries.filter((e) => e.included).length} photo(s)`;
  $("#status").textContent = "Average face generated!";
  $("#status").className = "status ready";
}

// ── Clear all ───────────────────────────────────────────────

function clearAll() {
  allEntries = [];
  alignedFaces = [];
  $("#gallery").innerHTML = "";
  $("#gallery-title").classList.add("hidden");
  $("#face-strip").innerHTML = "";
  $("#faces-section").classList.add("hidden");
  $("#result-section").classList.add("hidden");
  $("#load-preset-btn").disabled = false;
  updateButtons();
  $("#status").textContent = "Models loaded — load photos or drag & drop your own";
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

  $("#load-preset-btn").addEventListener("click", loadPresetImages);
  $("#generate-btn").addEventListener("click", generateAverageFace);
  $("#clear-btn").addEventListener("click", clearAll);
});
