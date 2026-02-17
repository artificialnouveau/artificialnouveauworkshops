/**
 * step2-cream-detection.js — In-browser ethnicity classification
 * Uses face-api.js (vladmandic) for face detection + FairFace ONNX for ethnicity
 */

const Step2 = {
  faceApiReady: false,
  fairfaceSession: null,
  modelsReady: false,

  RACE_LABELS: ['White', 'Black', 'Latino_Hispanic', 'East Asian', 'Southeast Asian', 'Indian', 'Middle Eastern'],
  IMAGENET_MEAN: [0.485, 0.456, 0.406],
  IMAGENET_STD:  [0.229, 0.224, 0.225],

  FACE_API_MODEL_URL: 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/',
  FAIRFACE_MODEL_URL: 'https://huggingface.co/facefusion/models-3.0.0/resolve/main/fairface.onnx',

  init() {
    document.getElementById('file-input-2').addEventListener('change', (e) => this.handleUpload(e));
    this.loadModels();
  },

  updateStatus(prefix, model, state, detail) {
    const el = document.getElementById(`${prefix}-${model}-status`);
    if (!el) return;
    const dot = el.querySelector('.model-load-dot');
    const detailEl = el.querySelector('.model-load-detail');
    dot.className = `model-load-dot ${state}`;
    detailEl.textContent = detail;
  },

  /** Broadcast model status to both Tab 2 (cream-) and Tab 3 (pol-) */
  broadcastStatus(model, state, detail) {
    this.updateStatus('cream', model, state, detail);
    this.updateStatus('pol', model, state, detail);
  },

  async loadModels() {
    // Load face-api.js
    this.broadcastStatus('faceapi', 'loading', 'loading...');
    try {
      await faceapi.nets.tinyFaceDetector.loadFromUri(this.FACE_API_MODEL_URL);
      this.faceApiReady = true;
      this.broadcastStatus('faceapi', 'ready', 'ready');
    } catch (err) {
      this.broadcastStatus('faceapi', 'error', 'failed: ' + err.message);
      return;
    }

    // Load FairFace ONNX
    this.broadcastStatus('fairface', 'loading', 'downloading (~85 MB)...');
    try {
      this.fairfaceSession = await ort.InferenceSession.create(this.FAIRFACE_MODEL_URL, {
        executionProviders: ['wasm'],
      });
      this.broadcastStatus('fairface', 'ready', 'ready');
    } catch (err) {
      this.broadcastStatus('fairface', 'error', 'failed: ' + err.message);
      return;
    }

    this.modelsReady = true;
    // Enable the Tab 3 analyze button too
    const btn = document.getElementById('btn-analyze-all');
    if (btn) btn.disabled = false;
  },

  softmax(logits) {
    const max = Math.max(...logits);
    const exps = logits.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
  },

  /**
   * Preprocess a face crop for FairFace: resize to 224x224, normalize with ImageNet stats.
   * Returns an ort.Tensor of shape [1, 3, 224, 224].
   */
  preprocessFace(canvas, box) {
    const { x, y, width, height } = box;
    const cropCanvas = document.createElement('canvas');
    cropCanvas.width = 224;
    cropCanvas.height = 224;
    const ctx = cropCanvas.getContext('2d');
    ctx.drawImage(canvas, x, y, width, height, 0, 0, 224, 224);

    const imageData = ctx.getImageData(0, 0, 224, 224);
    const { data } = imageData;
    const float32 = new Float32Array(3 * 224 * 224);

    for (let i = 0; i < 224 * 224; i++) {
      const r = data[i * 4] / 255;
      const g = data[i * 4 + 1] / 255;
      const b = data[i * 4 + 2] / 255;
      // CHW layout, normalized with ImageNet mean/std
      float32[0 * 224 * 224 + i] = (r - this.IMAGENET_MEAN[0]) / this.IMAGENET_STD[0];
      float32[1 * 224 * 224 + i] = (g - this.IMAGENET_MEAN[1]) / this.IMAGENET_STD[1];
      float32[2 * 224 * 224 + i] = (b - this.IMAGENET_MEAN[2]) / this.IMAGENET_STD[2];
    }

    return new ort.Tensor('float32', float32, [1, 3, 224, 224]);
  },

  /**
   * Run FairFace on a single face tensor. Returns { race: {label: pct}, dominant_race }.
   */
  async classifyFace(tensor) {
    const inputName = this.fairfaceSession.inputNames[0];
    const feeds = { [inputName]: tensor };
    const results = await this.fairfaceSession.run(feeds);

    // Inspect all outputs and find the one with race logits
    const outputNames = this.fairfaceSession.outputNames;
    let raceProbs = null;

    for (const name of outputNames) {
      const output = results[name];
      // Convert BigInt64Array / BigUint64Array to regular numbers
      const raw = Array.from(output.data).map(x => typeof x === 'bigint' ? Number(x) : x);

      if (raw.length >= 7) {
        // Check if these look like logits (floats) or class indices (ints)
        const hasDecimals = raw.some(v => v !== Math.floor(v));
        if (hasDecimals && raw.length >= 7) {
          // Float logits — take first 7 for race, apply softmax
          raceProbs = this.softmax(raw.slice(0, 7));
        } else if (!hasDecimals && raw.length <= 3) {
          // Likely class indices [race_idx, gender_idx, age_idx]
          const raceIdx = Math.min(Math.max(0, raw[0]), 6);
          raceProbs = this.RACE_LABELS.map((_, i) => i === raceIdx ? 0.95 : 0.05 / 6);
        } else {
          // Treat as logits anyway
          raceProbs = this.softmax(raw.slice(0, 7));
        }
        break;
      }
    }

    // Fallback: if no suitable output found, try first output raw
    if (!raceProbs) {
      const raw = Array.from(results[outputNames[0]].data).map(x => typeof x === 'bigint' ? Number(x) : x);
      raceProbs = this.softmax(raw.slice(0, Math.min(7, raw.length)));
      while (raceProbs.length < 7) raceProbs.push(0);
    }

    const race = {};
    let maxIdx = 0;
    for (let i = 0; i < this.RACE_LABELS.length; i++) {
      race[this.RACE_LABELS[i]] = raceProbs[i] * 100;
      if (raceProbs[i] > raceProbs[maxIdx]) maxIdx = i;
    }

    return { race, dominant_race: this.RACE_LABELS[maxIdx] };
  },

  /**
   * Detect faces + classify ethnicity for an image element.
   * Returns { faces: [{ region, race, dominant_race }] }
   */
  async analyzeImage(imgElement) {
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = imgElement.naturalWidth || imgElement.width;
    tempCanvas.height = imgElement.naturalHeight || imgElement.height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(imgElement, 0, 0, tempCanvas.width, tempCanvas.height);

    const detections = await faceapi.detectAllFaces(tempCanvas, new faceapi.TinyFaceDetectorOptions({
      inputSize: 416,
      scoreThreshold: 0.4,
    }));

    const faces = [];
    for (const det of detections) {
      const box = det.box;
      const pad = Math.round(Math.max(box.width, box.height) * 0.1);
      const px = Math.max(0, Math.round(box.x - pad));
      const py = Math.max(0, Math.round(box.y - pad));
      const pw = Math.min(tempCanvas.width - px, Math.round(box.width + pad * 2));
      const ph = Math.min(tempCanvas.height - py, Math.round(box.height + pad * 2));

      const tensor = this.preprocessFace(tempCanvas, { x: px, y: py, width: pw, height: ph });
      const result = await this.classifyFace(tensor);

      faces.push({
        region: { x: Math.round(box.x), y: Math.round(box.y), w: Math.round(box.width), h: Math.round(box.height) },
        race: result.race,
        dominant_race: result.dominant_race,
      });
    }

    return { faces };
  },

  async handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!this.modelsReady) {
      alert('Models are still loading. Please wait for both indicators to show "ready".');
      return;
    }

    document.getElementById('upload-area-2').classList.add('has-file');
    document.getElementById('cream-loading').classList.add('visible');
    document.getElementById('cream-results').classList.add('hidden');

    try {
      const img = await App.loadImage(file);
      const data = await this.analyzeImage(img);
      this.renderResults(img, data);
    } catch (err) {
      alert('Analysis failed: ' + err.message);
    } finally {
      document.getElementById('cream-loading').classList.remove('visible');
    }
  },

  renderResults(img, data) {
    const results = document.getElementById('cream-results');
    const canvas = document.getElementById('cream-canvas');
    const facesContainer = document.getElementById('cream-faces');

    const ctx = App.drawToCanvas(canvas, img);

    ctx.strokeStyle = '#4a9eff';
    ctx.lineWidth = 2;
    ctx.font = '14px monospace';
    ctx.fillStyle = '#4a9eff';

    const scaleX = canvas.width / img.width;
    const scaleY = canvas.height / img.height;

    data.faces.forEach((face, i) => {
      const r = face.region;
      const x = r.x * scaleX;
      const y = r.y * scaleY;
      const w = r.w * scaleX;
      const h = r.h * scaleY;
      ctx.strokeRect(x, y, w, h);
      ctx.fillText(`Face ${i + 1}`, x, y - 6);
    });

    facesContainer.innerHTML = data.faces.map((face, i) => {
      const sorted = Object.entries(face.race).sort((a, b) => b[1] - a[1]);
      const bars = sorted.map(([label, pct]) => `
        <div class="prediction-bar">
          <span class="prediction-label">${label}</span>
          <div class="prediction-track">
            <div class="prediction-fill" style="width:${pct.toFixed(1)}%"></div>
          </div>
          <span class="prediction-value">${pct.toFixed(1)}%</span>
        </div>
      `).join('');

      return `
        <div class="face-card">
          <h4>Face ${i + 1} — Dominant: ${face.dominant_race}</h4>
          ${bars}
        </div>
      `;
    }).join('');

    if (data.faces.length === 0) {
      facesContainer.innerHTML = '<div class="face-card"><h4>No faces detected</h4><p style="color:var(--text-dim)">Try uploading a clearer photo with visible faces.</p></div>';
    }

    results.classList.remove('hidden');
  },
};

document.addEventListener('DOMContentLoaded', () => Step2.init());
