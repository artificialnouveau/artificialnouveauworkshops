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
    const gridSize = 16; // Show a 16x16 grid from center of image
    const cellSize = 22;
    canvas.width = gridSize * cellSize;
    canvas.height = gridSize * cellSize;
    const ctx = canvas.getContext('2d');

    // Sample from center of image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = currentImg.width;
    tempCanvas.height = currentImg.height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(currentImg, 0, 0);

    const startX = Math.floor((currentImg.width - gridSize) / 2);
    const startY = Math.floor((currentImg.height - gridSize) / 2);
    const imgData = tempCtx.getImageData(startX, startY, gridSize, gridSize);

    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        const i = (y * gridSize + x) * 4;
        const r = imgData.data[i];
        const g = imgData.data[i + 1];
        const b = imgData.data[i + 2];
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(x * cellSize, y * cellSize, cellSize - 1, cellSize - 1);
      }
    }

    // Tap to inspect
    const pixelInfo = document.getElementById('pixel-info');
    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const cx = Math.floor((e.clientX - rect.left) * scaleX / cellSize);
      const cy = Math.floor((e.clientY - rect.top) * scaleY / cellSize);
      if (cx >= 0 && cx < gridSize && cy >= 0 && cy < gridSize) {
        const i = (cy * gridSize + cx) * 4;
        const r = imgData.data[i];
        const g = imgData.data[i + 1];
        const b = imgData.data[i + 2];
        pixelInfo.textContent = `Pixel (${cx}, ${cy}) → R:${r} G:${g} B:${b}`;
      }
    });
  }

  async function runClassification() {
    if (!App.models.mobilenet) {
      document.getElementById('predictions').innerHTML = '<p style="color:var(--red)">Model failed to load</p>';
      return;
    }

    const predictions = await App.models.mobilenet.classify(currentImg, 5);
    App.renderPredictions(document.getElementById('predictions'), predictions);
  }

  async function drawHeatmap() {
    const canvas = document.getElementById('canvas-heatmap');
    const maxW = 360;
    const scale = Math.min(1, maxW / currentImg.width);
    const w = Math.floor(currentImg.width * scale);
    const h = Math.floor(currentImg.height * scale);
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(currentImg, 0, 0, w, h);

    if (!App.models.mobilenet) return;

    // Simple occlusion-based attention: block regions and measure confidence drop
    const gridCols = 8;
    const gridRows = 8;
    const cellW = Math.floor(w / gridCols);
    const cellH = Math.floor(h / gridRows);

    // Baseline prediction
    const baseline = await App.models.mobilenet.classify(currentImg, 1);
    const baseConf = baseline[0].probability;

    const heatValues = [];

    for (let gy = 0; gy < gridRows; gy++) {
      for (let gx = 0; gx < gridCols; gx++) {
        // Create occluded version
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

    // Normalize and draw
    const maxDrop = Math.max(...heatValues.map(h => h.drop), 0.001);

    for (const { gx, gy, drop } of heatValues) {
      const intensity = drop / maxDrop;
      ctx.fillStyle = `rgba(255, ${Math.floor(100 * (1 - intensity))}, 0, ${intensity * 0.6})`;
      ctx.fillRect(gx * cellW, gy * cellH, cellW, cellH);
    }
  }
})();
