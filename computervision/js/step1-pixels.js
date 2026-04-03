/**
 * step1-pixels.js — Pixel grid, color channels, edge detection, classification, heatmap
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
    drawChannels();
    drawEdges();
    await runClassification();
    await drawHeatmap();
  });

  function drawOriginal() {
    App.drawToCanvas(document.getElementById('canvas-original-1'), currentImg, 360);
  }

  // ── Pixel grid: downsample entire image ──
  function drawPixelGrid() {
    const canvas = document.getElementById('canvas-pixels');
    const pixelInfo = document.getElementById('pixel-info');
    const maxDim = 24;
    const aspect = currentImg.width / currentImg.height;
    const gridCols = aspect >= 1 ? maxDim : Math.max(4, Math.round(maxDim * aspect));
    const gridRows = aspect >= 1 ? Math.max(4, Math.round(maxDim / aspect)) : maxDim;
    const cellSize = Math.floor(360 / gridCols);

    canvas.width = gridCols * cellSize;
    canvas.height = gridRows * cellSize;
    const ctx = canvas.getContext('2d');

    const tmp = document.createElement('canvas');
    tmp.width = gridCols; tmp.height = gridRows;
    const tctx = tmp.getContext('2d');
    tctx.imageSmoothingEnabled = false;
    tctx.drawImage(currentImg, 0, 0, gridCols, gridRows);
    const imgData = tctx.getImageData(0, 0, gridCols, gridRows);

    for (let y = 0; y < gridRows; y++) {
      for (let x = 0; x < gridCols; x++) {
        const i = (y * gridCols + x) * 4;
        ctx.fillStyle = `rgb(${imgData.data[i]},${imgData.data[i+1]},${imgData.data[i+2]})`;
        ctx.fillRect(x * cellSize, y * cellSize, cellSize - 1, cellSize - 1);
      }
    }

    canvas.addEventListener('click', (e) => {
      const rect = canvas.getBoundingClientRect();
      const cx = Math.floor((e.clientX - rect.left) * (canvas.width / rect.width) / cellSize);
      const cy = Math.floor((e.clientY - rect.top) * (canvas.height / rect.height) / cellSize);
      if (cx >= 0 && cx < gridCols && cy >= 0 && cy < gridRows) {
        const i = (cy * gridCols + cx) * 4;
        const r = imgData.data[i], g = imgData.data[i+1], b = imgData.data[i+2];
        pixelInfo.innerHTML =
          `Pixel (${cx},${cy}) &rarr; ` +
          `<span style="color:rgb(${r},100,100)">R:${r}</span> ` +
          `<span style="color:rgb(100,${g},100)">G:${g}</span> ` +
          `<span style="color:rgb(100,100,${b})">B:${b}</span>`;
      }
    });
  }

  // ── RGB Channels + Grayscale ──
  function drawChannels() {
    const maxW = 180;
    const scale = Math.min(1, maxW / currentImg.width);
    const w = Math.floor(currentImg.width * scale);
    const h = Math.floor(currentImg.height * scale);

    // Get pixel data from image
    const tmp = document.createElement('canvas');
    tmp.width = w; tmp.height = h;
    const tctx = tmp.getContext('2d');
    tctx.drawImage(currentImg, 0, 0, w, h);
    const src = tctx.getImageData(0, 0, w, h);

    const channels = ['canvas-red', 'canvas-green', 'canvas-blue', 'canvas-gray'];
    channels.forEach(id => {
      const c = document.getElementById(id);
      c.width = w; c.height = h;
    });

    // Red channel
    const redData = new ImageData(w, h);
    // Green channel
    const greenData = new ImageData(w, h);
    // Blue channel
    const blueData = new ImageData(w, h);
    // Grayscale
    const grayData = new ImageData(w, h);

    for (let i = 0; i < src.data.length; i += 4) {
      const r = src.data[i], g = src.data[i+1], b = src.data[i+2];
      const gray = Math.round(0.299 * r + 0.587 * g + 0.114 * b);

      redData.data[i] = r; redData.data[i+1] = 0; redData.data[i+2] = 0; redData.data[i+3] = 255;
      greenData.data[i] = 0; greenData.data[i+1] = g; greenData.data[i+2] = 0; greenData.data[i+3] = 255;
      blueData.data[i] = 0; blueData.data[i+1] = 0; blueData.data[i+2] = b; blueData.data[i+3] = 255;
      grayData.data[i] = gray; grayData.data[i+1] = gray; grayData.data[i+2] = gray; grayData.data[i+3] = 255;
    }

    document.getElementById('canvas-red').getContext('2d').putImageData(redData, 0, 0);
    document.getElementById('canvas-green').getContext('2d').putImageData(greenData, 0, 0);
    document.getElementById('canvas-blue').getContext('2d').putImageData(blueData, 0, 0);
    document.getElementById('canvas-gray').getContext('2d').putImageData(grayData, 0, 0);
  }

  // ── Edge Detection (Sobel filter) ──
  function drawEdges() {
    const maxW = 360;
    const scale = Math.min(1, maxW / currentImg.width);
    const w = Math.floor(currentImg.width * scale);
    const h = Math.floor(currentImg.height * scale);

    const tmp = document.createElement('canvas');
    tmp.width = w; tmp.height = h;
    const tctx = tmp.getContext('2d');
    tctx.drawImage(currentImg, 0, 0, w, h);
    const src = tctx.getImageData(0, 0, w, h);

    // Convert to grayscale array
    const gray = new Float32Array(w * h);
    for (let i = 0; i < gray.length; i++) {
      const j = i * 4;
      gray[i] = 0.299 * src.data[j] + 0.587 * src.data[j+1] + 0.114 * src.data[j+2];
    }

    // Sobel kernels
    const edges = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        const gx =
          -gray[(y-1)*w+(x-1)] + gray[(y-1)*w+(x+1)] +
          -2*gray[y*w+(x-1)] + 2*gray[y*w+(x+1)] +
          -gray[(y+1)*w+(x-1)] + gray[(y+1)*w+(x+1)];
        const gy =
          -gray[(y-1)*w+(x-1)] - 2*gray[(y-1)*w+x] - gray[(y-1)*w+(x+1)] +
          gray[(y+1)*w+(x-1)] + 2*gray[(y+1)*w+x] + gray[(y+1)*w+(x+1)];
        edges[idx] = Math.min(255, Math.sqrt(gx * gx + gy * gy));
      }
    }

    // Draw edges
    const edgeCanvas = document.getElementById('canvas-edges');
    edgeCanvas.width = w; edgeCanvas.height = h;
    const edgeCtx = edgeCanvas.getContext('2d');
    const edgeImg = edgeCtx.createImageData(w, h);
    for (let i = 0; i < edges.length; i++) {
      const v = edges[i];
      edgeImg.data[i*4] = v;
      edgeImg.data[i*4+1] = v;
      edgeImg.data[i*4+2] = v;
      edgeImg.data[i*4+3] = 255;
    }
    edgeCtx.putImageData(edgeImg, 0, 0);

    // Draw edges overlay on original
    const overlayCanvas = document.getElementById('canvas-edges-overlay');
    overlayCanvas.width = w; overlayCanvas.height = h;
    const oCtx = overlayCanvas.getContext('2d');
    oCtx.drawImage(currentImg, 0, 0, w, h);
    // Overlay edges in green
    const overlayImg = oCtx.getImageData(0, 0, w, h);
    for (let i = 0; i < edges.length; i++) {
      if (edges[i] > 40) {
        overlayImg.data[i*4] = 0;
        overlayImg.data[i*4+1] = 255;
        overlayImg.data[i*4+2] = 100;
        overlayImg.data[i*4+3] = 200;
      }
    }
    oCtx.putImageData(overlayImg, 0, 0);
  }

  // ── Classification ──
  async function runClassification() {
    if (!App.models.mobilenet) {
      document.getElementById('predictions').innerHTML =
        '<p style="color:var(--red)">Classification model failed to load.</p>';
      return;
    }
    const preds = await App.models.mobilenet.classify(currentImg, 5);
    App.renderPredictions(document.getElementById('predictions'), preds);
  }

  // ── Attention Heatmap ──
  async function drawHeatmap() {
    const canvas = document.getElementById('canvas-heatmap');
    const statusEl = document.getElementById('heatmap-status');
    const maxW = 360;
    const scale = Math.min(1, maxW / currentImg.width);
    const w = Math.floor(currentImg.width * scale);
    const h = Math.floor(currentImg.height * scale);
    canvas.width = w; canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(currentImg, 0, 0, w, h);

    if (!App.models.mobilenet) {
      statusEl.textContent = 'Heatmap unavailable — model not loaded.';
      return;
    }
    statusEl.textContent = 'Generating heatmap... (this takes a moment)';

    const cols = 8, rows = 8;
    const cw = Math.floor(w / cols), ch = Math.floor(h / rows);

    const baseline = await App.models.mobilenet.classify(currentImg, 1);
    const baseConf = baseline[0].probability;
    const baseLabel = baseline[0].className.split(',')[0];

    const heat = [];
    for (let gy = 0; gy < rows; gy++) {
      for (let gx = 0; gx < cols; gx++) {
        const occ = document.createElement('canvas');
        occ.width = w; occ.height = h;
        const octx = occ.getContext('2d');
        octx.drawImage(currentImg, 0, 0, w, h);
        octx.fillStyle = '#808080';
        octx.fillRect(gx * cw, gy * ch, cw, ch);
        const pred = await App.models.mobilenet.classify(occ, 1);
        heat.push({ gx, gy, drop: Math.max(0, baseConf - pred[0].probability) });
      }
    }

    const maxDrop = Math.max(...heat.map(h => h.drop), 0.001);
    ctx.drawImage(currentImg, 0, 0, w, h);
    for (const { gx, gy, drop } of heat) {
      const intensity = drop / maxDrop;
      ctx.fillStyle = `rgba(255, ${Math.floor(100*(1-intensity))}, 0, ${intensity * 0.55})`;
      ctx.fillRect(gx * cw, gy * ch, cw, ch);
    }
    statusEl.textContent =
      `Baseline: "${baseLabel}" at ${(baseConf*100).toFixed(1)}%. Orange = important regions.`;
  }
})();
