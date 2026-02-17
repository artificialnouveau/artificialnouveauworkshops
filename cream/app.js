const MATCH_THRESHOLD = 0.6;

const $ = (sel) => document.querySelector(sel);

let labeledDescriptors = [];
let modelsLoaded = false;

async function init() {
  const status = $("#status");
  const dropZone = $("#drop-zone");

  try {
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri("./models"),
      faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
      faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
    ]);

    const resp = await fetch("./descriptors.json");
    if (!resp.ok) throw new Error("Could not load descriptors.json");
    const data = await resp.json();

    labeledDescriptors = data.map((d) => ({
      label: d.label,
      descriptor: new Float32Array(d.descriptor),
    }));

    modelsLoaded = true;
    status.textContent = `Ready — ${labeledDescriptors.length} faces loaded`;
    status.classList.add("ready");
    dropZone.classList.remove("hidden");
  } catch (err) {
    status.textContent = "Error loading models: " + err.message;
    status.classList.add("error");
    console.error(err);
  }
}

function euclideanDistance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = a[i] - b[i];
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function findBestMatch(descriptor) {
  let bestDist = Infinity;
  let bestLabel = null;

  for (const entry of labeledDescriptors) {
    const dist = euclideanDistance(descriptor, entry.descriptor);
    if (dist < bestDist) {
      bestDist = dist;
      bestLabel = entry.label;
    }
  }

  return { label: bestLabel, distance: bestDist };
}

async function processImage(file) {
  const status = $("#status");
  const result = $("#result");
  const previewSection = $("#preview-section");
  const previewImg = $("#preview-img");
  const overlayCanvas = $("#overlay-canvas");
  const resetBtn = $("#reset-btn");
  const dropZone = $("#drop-zone");

  dropZone.classList.add("hidden");
  result.classList.add("hidden");
  result.className = "result hidden";
  status.textContent = "Detecting face…";
  status.className = "status";

  const imgURL = URL.createObjectURL(file);
  previewImg.src = imgURL;

  await new Promise((resolve) => {
    previewImg.onload = resolve;
  });

  previewSection.classList.remove("hidden");

  overlayCanvas.width = previewImg.naturalWidth;
  overlayCanvas.height = previewImg.naturalHeight;
  overlayCanvas.style.width = previewImg.width + "px";
  overlayCanvas.style.height = previewImg.height + "px";

  const detection = await faceapi
    .detectSingleFace(previewImg)
    .withFaceLandmarks()
    .withFaceDescriptor();

  if (!detection) {
    status.textContent = "No face detected";
    status.className = "status error";
    result.innerHTML = "Could not detect a face in this photo. Try a clearer image.";
    result.className = "result no-face";
    result.classList.remove("hidden");
    resetBtn.classList.remove("hidden");
    return;
  }

  // Draw bounding box on overlay
  const dims = faceapi.matchDimensions(overlayCanvas, {
    width: previewImg.naturalWidth,
    height: previewImg.naturalHeight,
  });
  const resized = faceapi.resizeResults(detection, dims);
  faceapi.draw.drawDetections(overlayCanvas, resized);

  const { label, distance } = findBestMatch(detection.descriptor);
  const confidence = Math.max(0, ((1 - distance / MATCH_THRESHOLD) * 100).toFixed(1));

  if (distance < MATCH_THRESHOLD) {
    status.textContent = "Match found!";
    status.className = "status ready";
    result.innerHTML =
      `<div class="match-name">${label}</div>` +
      `<div class="confidence">Distance: ${distance.toFixed(4)} — Confidence: ${confidence}%</div>`;
    result.className = "result match";
  } else {
    status.textContent = "No match";
    status.className = "status";
    result.innerHTML =
      `No match found in the database.<br>` +
      `<span class="confidence">Closest: ${label} (distance: ${distance.toFixed(4)})</span>`;
    result.className = "result no-match";
  }

  result.classList.remove("hidden");
  resetBtn.classList.remove("hidden");
}

function reset() {
  $("#preview-section").classList.add("hidden");
  $("#result").classList.add("hidden");
  $("#result").className = "result hidden";
  $("#reset-btn").classList.add("hidden");
  $("#drop-zone").classList.remove("hidden");
  $("#file-input").value = "";
  $("#status").textContent = `Ready — ${labeledDescriptors.length} faces loaded`;
  $("#status").className = "status ready";
}

document.addEventListener("DOMContentLoaded", () => {
  init();

  const dropZone = $("#drop-zone");
  const fileInput = $("#file-input");

  dropZone.addEventListener("click", () => fileInput.click());

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
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) processImage(file);
  });

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) processImage(file);
  });

  $("#reset-btn").addEventListener("click", reset);
});
