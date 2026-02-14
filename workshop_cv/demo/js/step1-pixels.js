/**
 * step1-pixels.js — Pixel grid visualization, MobileNet classification, attention heatmap
 */

(function () {
  const fileInput = document.getElementById('file-input-1');
  const results = document.getElementById('step1-results');
  const uploadArea = document.getElementById('upload-area-1');
  let currentImg = null;

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    uploadArea.classList.add('has-file');
    currentImg = await App.loadImage(file);
    results.classList.remove('hidden');

    drawOriginal();
    drawPixelGrid();
    await runClassification();
    await drawHeatmap();
  });

  function drawOriginal() {
    const canvas = document.getElementById('canvas-original-1');
    App.drawToCanvas(canvas, currentImg, 360);
  }

  function drawPixelGrid() {
    const canvas = document.getElementById('canvas-pixels');
    const pixelInfo = document.getElementById('pixel-info');

    // Downsample the ENTIRE image to a grid that fits
    // Use a grid that preserves aspect ratio
    const maxGridDim = 24; // max cells in either direction
    const aspect = currentImg.width / currentImg.height;
    let gridCols, gridRows;

    if (aspect >= 1) {
      gridCols = maxGridDim;
      gridRows = Math.max(4, Math.round(maxGridDim / aspect));
    } else {
      gridRows = maxGridDim;
      gridCols = Math.max(4, Math.round(maxGridDim * aspect));
    }

    const cellSize = Math.floor(360 / gridCols);
    canvas.width = gridCols * cellSize;
    canvas.height = gridRows * cellSize;
    const ctx = canvas.getContext('2d');

    // Draw the image at grid resolution to sample all pixels
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = gridCols;
    tempCanvas.height = gridRows;
    const tempCtx = tempCanvas.getContext('2d');
    // Disable smoothing so we get nearest-neighbor sampling
    tempCtx.imageSmoothingEnabled = false;
    tempCtx.drawImage(currentImg, 0, 0, gridCols, gridRows);
    const imgData = tempCtx.getImageData(0, 0, gridCols, gridRows);

    // Draw each pixel as a colored cell
    for (let y = 0; y < gridRows; y++) {
      for (let x = 0; x < gridCols; x++) {
        const i = (y * gridCols + x) * 4;
        const r = imgData.data[i];
        const g = imgData.data[i + 1];
        const b = imgData.data[i + 2];
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x * cellSize, y * cellSize, cellSize - 1, cellSize - 1);
      }
    }

    // Tap to inspect any cell
    canvas.onclick = null; // remove old listeners
    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const cx = Math.floor((e.clientX - rect.left) * scaleX / cellSize);
      const cy = Math.floor((e.clientY - rect.top) * scaleY / cellSize);
      if (cx >= 0 && cx < gridCols && cy >= 0 && cy < gridRows) {
        const i = (cy * gridCols + cx) * 4;
        const r = imgData.data[i];
        const g = imgData.data[i + 1];
        const b = imgData.data[i + 2];
        pixelInfo.innerHTML =
          `Pixel (${cx}, ${cy}) &rarr; ` +
          `<span style="color:rgb(${r},100,100)">R:${r}</span> ` +
          `<span style="color:rgb(100,${g},100)">G:${g}</span> ` +
          `<span style="color:rgb(100,100,${b})">B:${b}</span>`;
      }
    });
  }

  async function runClassification() {
    if (!App.models.mobilenet) {
      document.getElementById('predictions').innerHTML =
        '<p style="color:var(--red)">Classification model failed to load. Check console.</p>';
      return;
    }

    const predictions = await App.models.mobilenet.classify(currentImg, 5);
    App.renderPredictions(document.getElementById('predictions'), predictions);
  }

  async function drawHeatmap() {
    const canvas = document.getElementById('canvas-heatmap');
    const statusEl = document.getElementById('heatmap-status');
    const maxW = 360;
    const scale = Math.min(1, maxW / currentImg.width);
    const w = Math.floor(currentImg.width * scale);
    const h = Math.floor(currentImg.height * scale);
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(currentImg, 0, 0, w, h);

    if (!App.models.mobilenet) {
      statusEl.textContent = 'Heatmap unavailable — classification model not loaded.';
      return;
    }

    statusEl.textContent = 'Generating heatmap... (this takes a moment)';

    // Occlusion-based attention: block each region, measure confidence drop
    const gridCols = 8;
    const gridRows = 8;
    const cellW = Math.floor(w / gridCols);
    const cellH = Math.floor(h / gridRows);

    // Baseline prediction
    const baseline = await App.models.mobilenet.classify(currentImg, 1);
    const baseConf = baseline[0].probability;
    const baseLabel = baseline[0].className.split(',')[0];

    const heatValues = [];

    for (let gy = 0; gy < gridRows; gy++) {
      for (let gx = 0; gx < gridCols; gx++) {
        // Create version with this region blocked
        const occ = document.createElement('canvas');
        occ.width = w;
        occ.height = h;
        const octx = occ.getContext('2d');
        octx.drawImage(currentImg, 0, 0, w, h);
        octx.fillStyle = '#808080';
        octx.fillRect(gx * cellW, gy * cellH, cellW, cellH);

        const pred = await App.models.mobilenet.classify(occ, 1);
        const drop = Math.max(0, baseConf - pred[0].probability);
        heatValues.push({ gx, gy, drop });
      }
    }

    // Normalize and draw overlay
    const maxDrop = Math.max(...heatValues.map(h => h.drop), 0.001);

    // Redraw original image first
    ctx.drawImage(currentImg, 0, 0, w, h);

    for (const { gx, gy, drop } of heatValues) {
      const intensity = drop / maxDrop;
      ctx.fillStyle = `rgba(255, ${Math.floor(100 * (1 - intensity))}, 0, ${intensity * 0.55})`;
      ctx.fillRect(gx * cellW, gy * cellH, cellW, cellH);
    }

    statusEl.textContent =
      `Baseline: "${baseLabel}" at ${(baseConf * 100).toFixed(1)}% confidence. ` +
      `Orange regions caused the biggest confidence drop when blocked.`;
  }
})();
