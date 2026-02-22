/**
 * app.js — Main orchestration: step navigation, webcam lifecycle, and 5 ML modules
 */

/* ================================================================
   STEP 1 — Neural Style Transfer (Magenta)
   ================================================================ */
const Step1 = {
  model: null,
  contentImg: null,
  styleImg: null,

  init() {
    document.getElementById('file-content').addEventListener('change', e => this.handleUpload(e, 'content'));
    document.getElementById('file-style').addEventListener('change', e => this.handleUpload(e, 'style'));
    document.getElementById('btn-stylize').addEventListener('click', () => this.stylize());
  },

  async handleUpload(e, type) {
    const file = e.target.files[0];
    if (!file) return;
    const img = await App.loadImage(file);
    const canvasId = type === 'content' ? 'canvas-content' : 'canvas-style';
    const canvas = document.getElementById(canvasId);
    App.drawToCanvas(canvas, img, 300);
    canvas.style.display = 'block';

    if (type === 'content') {
      this.contentImg = img;
      document.getElementById('upload-area-content').classList.add('has-file');
    } else {
      this.styleImg = img;
      document.getElementById('upload-area-style').classList.add('has-file');
    }

    if (this.contentImg && this.styleImg) {
      document.getElementById('btn-stylize').disabled = false;
    }
  },

  async loadModel() {
    if (this.model) return this.model;
    const loading = document.getElementById('step1-loading');
    const fill = document.getElementById('step1-loading-fill');
    const status = document.getElementById('step1-loading-status');
    loading.classList.add('visible');
    status.textContent = 'Downloading style transfer model...';
    fill.style.width = '30%';

    try {
      this.model = new mi.ArbitraryStyleTransferNetwork();
      fill.style.width = '60%';
      status.textContent = 'Initializing model...';
      await this.model.initialize();
      fill.style.width = '100%';
      status.textContent = 'Model ready!';
      setTimeout(() => loading.classList.remove('visible'), 1000);
      return this.model;
    } catch (err) {
      status.textContent = 'Error loading model: ' + err.message;
      fill.style.width = '0%';
      throw err;
    }
  },

  async stylize() {
    const btn = document.getElementById('btn-stylize');
    btn.disabled = true;
    btn.textContent = 'Stylizing...';

    try {
      const model = await this.loadModel();
      const result = await model.stylize(this.contentImg, this.styleImg);
      const canvas = document.getElementById('canvas-result');
      canvas.width = result.width;
      canvas.height = result.height;
      canvas.getContext('2d').drawImage(result, 0, 0);
      document.getElementById('step1-results').classList.remove('hidden');
    } catch (err) {
      console.error('Style transfer error:', err);
      alert('Style transfer failed: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Stylize';
    }
  }
};

/* ================================================================
   STEP 2 — Teachable Machine (MobileNet + Transfer Learning)
   ================================================================ */
const Step2 = {
  mobilenet: null,
  classifier: null,
  stream: null,
  video: null,
  samples: [[], [], []],
  classNames: ['Class 1', 'Class 2', 'Class 3'],
  predicting: false,
  animFrameId: null,

  init() {
    this.video = document.getElementById('step2-video');

    document.querySelectorAll('.btn-sample').forEach(btn => {
      btn.addEventListener('click', () => this.addSample(parseInt(btn.dataset.class)));
    });
    document.getElementById('btn-train').addEventListener('click', () => this.train());

    // Track class name changes
    for (let i = 0; i < 3; i++) {
      document.getElementById(`class-name-${i}`).addEventListener('input', e => {
        this.classNames[i] = e.target.value || `Class ${i + 1}`;
      });
    }
  },

  async start() {
    const loading = document.getElementById('step2-loading');
    const fill = document.getElementById('step2-loading-fill');
    const status = document.getElementById('step2-loading-status');

    // Start webcam
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 224, height: 224, facingMode: 'user' }
      });
      this.video.srcObject = this.stream;
      await this.video.play();
    } catch (err) {
      alert('Could not access webcam: ' + err.message);
      return;
    }

    // Load MobileNet for feature extraction
    if (!this.mobilenet) {
      loading.classList.add('visible');
      status.textContent = 'Loading MobileNet feature extractor...';
      fill.style.width = '30%';

      try {
        const mobilenetModel = await tf.loadLayersModel(
          'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
        );
        // Use intermediate layer output as feature extractor
        const layer = mobilenetModel.getLayer('conv_pw_13_relu');
        this.mobilenet = tf.model({
          inputs: mobilenetModel.inputs,
          outputs: layer.output
        });
        fill.style.width = '100%';
        status.textContent = 'MobileNet loaded!';
        setTimeout(() => loading.classList.remove('visible'), 800);
      } catch (err) {
        status.textContent = 'Error loading MobileNet: ' + err.message;
        console.error(err);
        return;
      }
    }
  },

  stop() {
    this.predicting = false;
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
  },

  getFeatures(imgElement) {
    return tf.tidy(() => {
      let tensor;
      if (imgElement instanceof HTMLVideoElement) {
        tensor = tf.browser.fromPixels(imgElement).resizeBilinear([224, 224]);
      } else {
        tensor = tf.browser.fromPixels(imgElement).resizeBilinear([224, 224]);
      }
      const normalized = tensor.toFloat().div(127.5).sub(1);
      const batched = normalized.expandDims(0);
      return this.mobilenet.predict(batched);
    });
  },

  addSample(classIndex) {
    if (!this.mobilenet || !this.video.srcObject) return;

    const features = this.getFeatures(this.video);
    this.samples[classIndex].push(features);

    const countEl = document.getElementById(`count-${classIndex}`);
    countEl.textContent = this.samples[classIndex].length;

    // Enable train button if at least 2 classes have samples
    const classesWithSamples = this.samples.filter(s => s.length > 0).length;
    document.getElementById('btn-train').disabled = classesWithSamples < 2;
  },

  async train() {
    const btn = document.getElementById('btn-train');
    const loading = document.getElementById('step2-loading');
    const fill = document.getElementById('step2-loading-fill');
    const status = document.getElementById('step2-loading-status');
    const loadText = document.getElementById('step2-loading-text');

    btn.disabled = true;
    loading.classList.add('visible');
    loadText.innerHTML = 'Training<span class="loading-dots"></span>';
    status.textContent = 'Building training data...';
    fill.style.width = '10%';

    // Determine active classes
    const activeClasses = [];
    const xs = [];
    const ys = [];
    for (let i = 0; i < 3; i++) {
      if (this.samples[i].length > 0) {
        activeClasses.push(i);
      }
    }
    const numClasses = activeClasses.length;

    // Build training tensors
    for (const ci of activeClasses) {
      for (const feat of this.samples[ci]) {
        xs.push(feat);
        const oneHot = new Array(numClasses).fill(0);
        oneHot[activeClasses.indexOf(ci)] = 1;
        ys.push(oneHot);
      }
    }

    const xConcat = tf.concat(xs);
    const flatShape = [xConcat.shape[0], xConcat.shape[1] * xConcat.shape[2] * xConcat.shape[3]];
    const xFlat = xConcat.reshape(flatShape);
    const yTensor = tf.tensor2d(ys);

    fill.style.width = '30%';
    status.textContent = 'Creating classifier...';

    // Build simple dense classifier
    if (this.classifier) this.classifier.dispose();
    this.classifier = tf.sequential();
    this.classifier.add(tf.layers.dense({
      inputShape: [flatShape[1]],
      units: 64,
      activation: 'relu'
    }));
    this.classifier.add(tf.layers.dense({
      units: numClasses,
      activation: 'softmax'
    }));
    this.classifier.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });

    fill.style.width = '50%';
    status.textContent = 'Training...';

    await this.classifier.fit(xFlat, yTensor, {
      epochs: 20,
      batchSize: 16,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const pct = 50 + (epoch / 20) * 45;
          fill.style.width = pct + '%';
          status.textContent = `Epoch ${epoch + 1}/20 — accuracy: ${(logs.acc * 100).toFixed(1)}%`;
        }
      }
    });

    xFlat.dispose();
    yTensor.dispose();

    fill.style.width = '100%';
    status.textContent = 'Training complete! Running live predictions...';
    setTimeout(() => loading.classList.remove('visible'), 1000);

    // Start live prediction
    this.activeClasses = activeClasses;
    this.predicting = true;
    document.getElementById('step2-predictions').classList.remove('hidden');
    this.predictLoop();
  },

  predictLoop() {
    if (!this.predicting) return;

    const features = this.getFeatures(this.video);
    const flatShape = [1, features.shape[1] * features.shape[2] * features.shape[3]];
    const flat = features.reshape(flatShape);
    const preds = this.classifier.predict(flat);
    const data = preds.dataSync();

    const predictions = this.activeClasses.map((ci, idx) => ({
      className: this.classNames[ci],
      probability: data[idx]
    }));
    predictions.sort((a, b) => b.probability - a.probability);

    App.renderPredictions(document.getElementById('step2-pred-bars'), predictions);

    features.dispose();
    flat.dispose();
    preds.dispose();

    this.animFrameId = requestAnimationFrame(() => this.predictLoop());
  }
};

/* ================================================================
   STEP 3 — Hand & Face Tracking (MediaPipe)
   ================================================================ */
const Step3 = {
  handLandmarker: null,
  faceLandmarker: null,
  stream: null,
  video: null,
  canvas: null,
  ctx: null,
  running: false,
  showHands: true,
  showFace: true,
  animFrameId: null,

  init() {
    this.video = document.getElementById('step3-video');
    this.canvas = document.getElementById('step3-overlay');

    document.getElementById('toggle-hands').addEventListener('click', e => {
      this.showHands = !this.showHands;
      e.target.textContent = `Hands: ${this.showHands ? 'ON' : 'OFF'}`;
      e.target.classList.toggle('active', this.showHands);
    });
    document.getElementById('toggle-face').addEventListener('click', e => {
      this.showFace = !this.showFace;
      e.target.textContent = `Face: ${this.showFace ? 'ON' : 'OFF'}`;
      e.target.classList.toggle('active', this.showFace);
    });
  },

  async start() {
    const loading = document.getElementById('step3-loading');
    const fill = document.getElementById('step3-loading-fill');
    const status = document.getElementById('step3-loading-status');
    const info = document.getElementById('step3-info');

    // Start webcam
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });
      this.video.srcObject = this.stream;
      await this.video.play();

      this.canvas.width = this.video.videoWidth || 640;
      this.canvas.height = this.video.videoHeight || 480;
      this.ctx = this.canvas.getContext('2d');
    } catch (err) {
      alert('Could not access webcam: ' + err.message);
      return;
    }

    // Load MediaPipe models
    if (!this.handLandmarker || !this.faceLandmarker) {
      loading.classList.add('visible');
      status.textContent = 'Loading MediaPipe vision models...';
      fill.style.width = '10%';

      try {
        const vision = await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs');
        const { HandLandmarker, FaceLandmarker, FilesetResolver } = vision;

        fill.style.width = '30%';
        status.textContent = 'Loading vision WASM runtime...';
        const wasmFileset = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
        );

        fill.style.width = '50%';
        status.textContent = 'Loading hand landmarker...';
        this.handLandmarker = await HandLandmarker.createFromOptions(wasmFileset, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numHands: 2
        });

        fill.style.width = '75%';
        status.textContent = 'Loading face landmarker...';
        this.faceLandmarker = await FaceLandmarker.createFromOptions(wasmFileset, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numFaces: 1
        });

        fill.style.width = '100%';
        status.textContent = 'Models loaded!';
        setTimeout(() => loading.classList.remove('visible'), 800);
      } catch (err) {
        status.textContent = 'Error loading models: ' + err.message;
        console.error(err);
        return;
      }
    }

    info.classList.remove('hidden');
    document.getElementById('step3-status-text').textContent = 'Tracking active';
    this.running = true;
    this.detectLoop();
  },

  stop() {
    this.running = false;
    if (this.animFrameId) {
      cancelAnimationFrame(this.animFrameId);
      this.animFrameId = null;
    }
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
  },

  detectLoop() {
    if (!this.running) return;

    const now = performance.now();

    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // Hand detection
    if (this.showHands && this.handLandmarker) {
      try {
        const handResults = this.handLandmarker.detectForVideo(this.video, now);
        if (handResults.landmarks) {
          for (const hand of handResults.landmarks) {
            this.drawHandLandmarks(hand);
          }
        }
      } catch (e) { /* skip frame */ }
    }

    // Face detection
    if (this.showFace && this.faceLandmarker) {
      try {
        const faceResults = this.faceLandmarker.detectForVideo(this.video, now);
        if (faceResults.faceLandmarks) {
          for (const face of faceResults.faceLandmarks) {
            this.drawFaceLandmarks(face);
          }
        }
      } catch (e) { /* skip frame */ }
    }

    this.animFrameId = requestAnimationFrame(() => this.detectLoop());
  },

  drawHandLandmarks(landmarks) {
    const w = this.canvas.width;
    const h = this.canvas.height;

    // Draw connections
    const connections = [
      [0,1],[1,2],[2,3],[3,4],       // thumb
      [0,5],[5,6],[6,7],[7,8],       // index
      [0,9],[9,10],[10,11],[11,12],  // middle
      [0,13],[13,14],[14,15],[15,16],// ring
      [0,17],[17,18],[18,19],[19,20],// pinky
      [5,9],[9,13],[13,17]           // palm
    ];

    this.ctx.strokeStyle = '#00ff66';
    this.ctx.lineWidth = 2;
    for (const [a, b] of connections) {
      this.ctx.beginPath();
      this.ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
      this.ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
      this.ctx.stroke();
    }

    // Draw keypoints
    this.ctx.fillStyle = '#4a9eff';
    for (const pt of landmarks) {
      this.ctx.beginPath();
      this.ctx.arc(pt.x * w, pt.y * h, 4, 0, 2 * Math.PI);
      this.ctx.fill();
    }
  },

  drawFaceLandmarks(landmarks) {
    const w = this.canvas.width;
    const h = this.canvas.height;

    this.ctx.fillStyle = 'rgba(255, 217, 74, 0.5)';
    for (const pt of landmarks) {
      this.ctx.beginPath();
      this.ctx.arc(pt.x * w, pt.y * h, 1.2, 0, 2 * Math.PI);
      this.ctx.fill();
    }
  }
};

/* ================================================================
   STEP 4 — Image Captioning (Transformers.js / BLIP)
   ================================================================ */
const Step4 = {
  pipeline: null,
  loading: false,

  init() {
    document.getElementById('file-caption').addEventListener('change', e => this.handleUpload(e));
  },

  async handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const img = await App.loadImage(file);
    const canvas = document.getElementById('canvas-caption');
    App.drawToCanvas(canvas, img, 500);
    canvas.style.display = 'block';
    document.getElementById('upload-area-caption').classList.add('has-file');

    document.getElementById('step4-results').classList.add('hidden');
    document.getElementById('step4-error').classList.add('hidden');

    await this.caption(file);
  },

  async caption(file) {
    const loadingEl = document.getElementById('step4-loading');
    const fill = document.getElementById('step4-loading-fill');
    const status = document.getElementById('step4-loading-status');
    const pctEl = document.getElementById('step4-loading-pct');

    try {
      if (!this.pipeline) {
        if (this.loading) return;
        this.loading = true;
        loadingEl.classList.add('visible');
        status.textContent = 'Starting download...';
        fill.style.width = '5%';

        this.pipeline = await window._loadCaptionPipeline((progress) => {
          if (progress.status === 'progress' && progress.total) {
            const pct = Math.round((progress.loaded / progress.total) * 100);
            fill.style.width = pct + '%';
            pctEl.textContent = pct + '%';
            const mb = (progress.loaded / 1e6).toFixed(1);
            const totalMb = (progress.total / 1e6).toFixed(1);
            status.textContent = `${progress.file || 'model'}: ${mb} / ${totalMb} MB`;
          } else if (progress.status === 'done') {
            status.textContent = 'Model loaded!';
          }
        });

        fill.style.width = '100%';
        this.loading = false;
        setTimeout(() => loadingEl.classList.remove('visible'), 800);
      }

      // Run captioning
      loadingEl.classList.add('visible');
      fill.style.width = '100%';
      status.textContent = 'Generating caption...';

      const imageUrl = URL.createObjectURL(file);
      const output = await this.pipeline(imageUrl);
      URL.revokeObjectURL(imageUrl);

      loadingEl.classList.remove('visible');

      const caption = output[0]?.generated_text || 'No caption generated.';
      document.getElementById('step4-caption').textContent = caption;
      document.getElementById('step4-results').classList.remove('hidden');

    } catch (err) {
      console.error('Captioning error:', err);
      this.loading = false;
      loadingEl.classList.remove('visible');
      document.getElementById('step4-error-text').textContent = 'Error: ' + err.message;
      document.getElementById('step4-error').classList.remove('hidden');
    }
  }
};

/* ================================================================
   STEP 5 — Text to Image (Stable Diffusion via WebGPU)
   ================================================================ */
const Step5 = {
  pipeline: null,
  loading: false,

  init() {
    document.getElementById('btn-generate').addEventListener('click', () => this.generate());

    // Check WebGPU support
    if (!navigator.gpu) {
      document.getElementById('step5-webgpu-warning').classList.remove('hidden');
      document.getElementById('btn-generate').disabled = true;
      document.getElementById('step5-prompt').disabled = true;
    } else {
      document.getElementById('step5-prompt').disabled = false;
      document.getElementById('btn-generate').disabled = false;
    }
  },

  async loadPipeline() {
    if (this.pipeline) return this.pipeline;
    if (this.loading) return null;
    this.loading = true;

    const loadingEl = document.getElementById('step5-loading');
    const fill = document.getElementById('step5-loading-fill');
    const status = document.getElementById('step5-loading-status');
    const pctEl = document.getElementById('step5-loading-pct');
    const loadText = document.getElementById('step5-loading-text');

    loadingEl.classList.add('visible');
    loadText.innerHTML = 'Downloading Stable Diffusion<span class="loading-dots"></span>';
    status.textContent = 'This may take a few minutes on first use (~1.7 GB)...';
    fill.style.width = '5%';

    try {
      this.pipeline = await window._loadTextToImagePipeline((progress) => {
        if (progress.status === 'progress' && progress.total) {
          const pct = Math.round((progress.loaded / progress.total) * 100);
          fill.style.width = Math.min(pct, 95) + '%';
          pctEl.textContent = pct + '%';
          const mb = (progress.loaded / 1e6).toFixed(1);
          const totalMb = (progress.total / 1e6).toFixed(1);
          status.textContent = `${progress.file || progress.name || 'component'}: ${mb} / ${totalMb} MB`;
        } else if (progress.status === 'loading') {
          status.textContent = `Loading ${progress.name}...`;
        } else if (progress.status === 'done') {
          status.textContent = 'Component loaded!';
        }
      });

      fill.style.width = '100%';
      status.textContent = 'All components loaded!';
      this.loading = false;
      setTimeout(() => loadingEl.classList.remove('visible'), 800);
      return this.pipeline;
    } catch (err) {
      this.loading = false;
      loadingEl.classList.remove('visible');
      throw err;
    }
  },

  async generate() {
    const btn = document.getElementById('btn-generate');
    const prompt = document.getElementById('step5-prompt').value.trim();
    if (!prompt) return;

    btn.disabled = true;
    btn.textContent = 'Generating...';

    document.getElementById('step5-results').classList.add('hidden');
    document.getElementById('step5-error').classList.add('hidden');

    const loadingEl = document.getElementById('step5-loading');
    const fill = document.getElementById('step5-loading-fill');
    const status = document.getElementById('step5-loading-status');
    const loadText = document.getElementById('step5-loading-text');

    try {
      const pipeline = await this.loadPipeline();
      if (!pipeline) {
        btn.disabled = false;
        btn.textContent = 'Generate Image';
        return;
      }

      loadingEl.classList.add('visible');
      loadText.innerHTML = 'Generating image<span class="loading-dots"></span>';
      status.textContent = 'Running diffusion steps...';
      fill.style.width = '50%';

      const { tokenizer, text_encoder, vae_decoder, unetSession, OrtTensor } = pipeline;

      // Tokenize
      const inputs = tokenizer(prompt, {
        padding: 'max_length',
        max_length: 77,
        truncation: true,
        return_tensors: 'pt'
      });

      // Text encoding
      status.textContent = 'Encoding text prompt...';
      fill.style.width = '60%';
      const textOutput = await text_encoder({ input_ids: inputs.input_ids });
      const textEmbeddings = textOutput.text_embeds || textOutput.last_hidden_state;

      // Unconditional embeddings (empty prompt for classifier-free guidance)
      const uncondInputs = tokenizer('', {
        padding: 'max_length',
        max_length: 77,
        truncation: true,
        return_tensors: 'pt'
      });
      const uncondOutput = await text_encoder({ input_ids: uncondInputs.input_ids });
      const uncondEmbeddings = uncondOutput.text_embeds || uncondOutput.last_hidden_state;

      // Generate random latent
      status.textContent = 'Running UNet denoising...';
      fill.style.width = '70%';
      const latentShape = [1, 4, 64, 64];
      const latentData = new Float32Array(1 * 4 * 64 * 64);
      for (let i = 0; i < latentData.length; i++) {
        // Box-Muller transform for normal distribution
        const u1 = Math.random();
        const u2 = Math.random();
        latentData[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }

      // SD-Turbo: single-step inference
      const timestep = new Float32Array([999]);
      const latentTensor = new OrtTensor('float16', new Uint16Array(latentData.length), latentShape);
      // Convert float32 to float16 (simplified)
      const f32 = latentData;
      const f16 = latentTensor.data;
      for (let i = 0; i < f32.length; i++) {
        // Simple float32 to float16 conversion
        const val = f32[i];
        const floatView = new Float32Array([val]);
        const intView = new Uint32Array(floatView.buffer);
        const bits = intView[0];
        const sign = (bits >> 16) & 0x8000;
        const exponent = ((bits >> 23) & 0xff) - 127 + 15;
        const mantissa = (bits >> 13) & 0x3ff;
        if (exponent <= 0) {
          f16[i] = sign;
        } else if (exponent >= 31) {
          f16[i] = sign | 0x7c00;
        } else {
          f16[i] = sign | (exponent << 10) | mantissa;
        }
      }

      // Run UNet
      const unetOutput = await unetSession.run({
        sample: latentTensor,
        timestep: new OrtTensor('float32', timestep, [1]),
        encoder_hidden_states: textEmbeddings.ort_tensor || textEmbeddings
      });

      // Decode with VAE
      status.textContent = 'Decoding image...';
      fill.style.width = '90%';
      const denoised = unetOutput.out_sample || Object.values(unetOutput)[0];
      const decoded = await vae_decoder({ latent_sample: denoised });
      const imageData = decoded.sample || Object.values(decoded)[0];

      // Convert to canvas
      const canvas = document.getElementById('canvas-generated');
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext('2d');
      const imgData = ctx.createImageData(512, 512);
      const pixels = imageData.data || imageData.cpuData;

      // Denormalize from [-1,1] to [0,255]
      for (let y = 0; y < 512; y++) {
        for (let x = 0; x < 512; x++) {
          const idx = (y * 512 + x) * 4;
          const srcIdx = y * 512 + x;
          imgData.data[idx]     = Math.max(0, Math.min(255, (pixels[srcIdx] + 1) * 127.5));               // R
          imgData.data[idx + 1] = Math.max(0, Math.min(255, (pixels[srcIdx + 512*512] + 1) * 127.5));     // G
          imgData.data[idx + 2] = Math.max(0, Math.min(255, (pixels[srcIdx + 512*512*2] + 1) * 127.5));   // B
          imgData.data[idx + 3] = 255;                                                                      // A
        }
      }
      ctx.putImageData(imgData, 0, 0);

      fill.style.width = '100%';
      status.textContent = 'Done!';
      setTimeout(() => loadingEl.classList.remove('visible'), 800);
      document.getElementById('step5-results').classList.remove('hidden');

    } catch (err) {
      console.error('Text-to-image error:', err);
      loadingEl.classList.remove('visible');
      document.getElementById('step5-error-text').textContent =
        'Error: ' + err.message + '\n\nNote: Text-to-image requires Chrome 113+ with WebGPU enabled and significant GPU memory (~4GB VRAM). This feature is experimental.';
      document.getElementById('step5-error').classList.remove('hidden');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Generate Image';
    }
  }
};

/* ================================================================
   STEP 6 — Depth Estimation (Depth Anything)
   ================================================================ */
const Step6 = {
  pipeline: null,
  loading: false,

  init() {
    document.getElementById('file-depth').addEventListener('change', e => this.handleUpload(e));
  },

  async handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const img = await App.loadImage(file);
    const inputCanvas = document.getElementById('canvas-depth-input');
    App.drawToCanvas(inputCanvas, img, 400);
    document.getElementById('upload-area-depth').classList.add('has-file');
    document.getElementById('step6-results').classList.remove('hidden');
    document.getElementById('step6-error').classList.add('hidden');

    await this.estimateDepth(file, img);
  },

  async estimateDepth(file, img) {
    const loadingEl = document.getElementById('step6-loading');
    const fill = document.getElementById('step6-loading-fill');
    const status = document.getElementById('step6-loading-status');
    const pctEl = document.getElementById('step6-loading-pct');

    try {
      if (!this.pipeline) {
        if (this.loading) return;
        this.loading = true;
        loadingEl.classList.add('visible');
        status.textContent = 'Starting download...';
        fill.style.width = '5%';

        this.pipeline = await window._loadDepthPipeline((progress) => {
          if (progress.status === 'progress' && progress.total) {
            const pct = Math.round((progress.loaded / progress.total) * 100);
            fill.style.width = pct + '%';
            pctEl.textContent = pct + '%';
            const mb = (progress.loaded / 1e6).toFixed(1);
            const totalMb = (progress.total / 1e6).toFixed(1);
            status.textContent = `${progress.file || 'model'}: ${mb} / ${totalMb} MB`;
          } else if (progress.status === 'done') {
            status.textContent = 'Model loaded!';
          }
        });

        fill.style.width = '100%';
        this.loading = false;
        setTimeout(() => loadingEl.classList.remove('visible'), 800);
      }

      // Run depth estimation
      loadingEl.classList.add('visible');
      fill.style.width = '100%';
      status.textContent = 'Estimating depth...';

      const imageUrl = URL.createObjectURL(file);
      const output = await this.pipeline(imageUrl);
      URL.revokeObjectURL(imageUrl);

      loadingEl.classList.remove('visible');

      // Render depth map to canvas
      const depthCanvas = document.getElementById('canvas-depth-output');
      const depthData = output.depth || output.predicted_depth;

      if (depthData && depthData.toCanvas) {
        // transformers.js RawImage has toCanvas()
        const tempCanvas = depthData.toCanvas();
        depthCanvas.width = tempCanvas.width;
        depthCanvas.height = tempCanvas.height;
        depthCanvas.getContext('2d').drawImage(tempCanvas, 0, 0);
      } else if (depthData && depthData.data) {
        // Manual rendering from tensor data
        const w = depthData.width || img.width;
        const h = depthData.height || img.height;
        depthCanvas.width = w;
        depthCanvas.height = h;
        const ctx = depthCanvas.getContext('2d');
        const imgData = ctx.createImageData(w, h);
        const data = depthData.data;

        // Find min/max for normalization
        let min = Infinity, max = -Infinity;
        for (let i = 0; i < data.length; i++) {
          if (data[i] < min) min = data[i];
          if (data[i] > max) max = data[i];
        }
        const range = max - min || 1;

        for (let i = 0; i < data.length; i++) {
          const val = Math.round(((data[i] - min) / range) * 255);
          imgData.data[i * 4]     = val;
          imgData.data[i * 4 + 1] = val;
          imgData.data[i * 4 + 2] = val;
          imgData.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
      }

      document.getElementById('step6-results').classList.remove('hidden');

    } catch (err) {
      console.error('Depth estimation error:', err);
      this.loading = false;
      loadingEl.classList.remove('visible');
      document.getElementById('step6-error-text').textContent = 'Error: ' + err.message;
      document.getElementById('step6-error').classList.remove('hidden');
    }
  }
};

/* ================================================================
   STEP 7 — Segment Anything (SAM)
   ================================================================ */
const Step7 = {
  model: null,
  processor: null,
  RawImage: null,
  imageEmbeddings: null,
  imageProcessed: null,
  originalImg: null,
  loading: false,

  init() {
    document.getElementById('file-sam').addEventListener('change', e => this.handleUpload(e));
    document.getElementById('canvas-sam-overlay').addEventListener('click', e => this.handleClick(e));
  },

  async handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    const img = await App.loadImage(file);
    this.originalImg = img;
    const canvas = document.getElementById('canvas-sam');
    App.drawToCanvas(canvas, img, 500);
    document.getElementById('upload-area-sam').classList.add('has-file');
    document.getElementById('step7-results').classList.remove('hidden');
    document.getElementById('step7-error').classList.add('hidden');

    // Sync overlay canvas size
    const overlay = document.getElementById('canvas-sam-overlay');
    overlay.width = canvas.width;
    overlay.height = canvas.height;

    await this.encodeImage(file);
  },

  async encodeImage(file) {
    const loadingEl = document.getElementById('step7-loading');
    const fill = document.getElementById('step7-loading-fill');
    const status = document.getElementById('step7-loading-status');
    const pctEl = document.getElementById('step7-loading-pct');

    try {
      // Load model if needed
      if (!this.model) {
        if (this.loading) return;
        this.loading = true;
        loadingEl.classList.add('visible');
        status.textContent = 'Downloading SAM model...';
        fill.style.width = '5%';

        const sam = await window._loadSAM((progress) => {
          if (progress.status === 'progress' && progress.total) {
            const pct = Math.round((progress.loaded / progress.total) * 100);
            fill.style.width = pct + '%';
            pctEl.textContent = pct + '%';
            const mb = (progress.loaded / 1e6).toFixed(1);
            const totalMb = (progress.total / 1e6).toFixed(1);
            status.textContent = `${progress.file || 'model'}: ${mb} / ${totalMb} MB`;
          } else if (progress.status === 'done') {
            status.textContent = 'Component loaded!';
          }
        });

        this.model = sam.model;
        this.processor = sam.processor;
        this.RawImage = sam.RawImage;
        this.loading = false;
      }

      // Encode the image
      loadingEl.classList.add('visible');
      fill.style.width = '80%';
      status.textContent = 'Encoding image (this takes a moment)...';

      const imageUrl = URL.createObjectURL(file);
      const rawImage = await this.RawImage.fromURL(imageUrl);
      URL.revokeObjectURL(imageUrl);

      const processed = await this.processor(rawImage);
      this.imageProcessed = processed;
      this.imageEmbeddings = await this.model.get_image_embeddings(processed);

      fill.style.width = '100%';
      status.textContent = 'Ready — click on the image to segment!';
      setTimeout(() => loadingEl.classList.remove('visible'), 800);

      document.getElementById('step7-hint').classList.remove('hidden');

    } catch (err) {
      console.error('SAM encoding error:', err);
      this.loading = false;
      loadingEl.classList.remove('visible');
      document.getElementById('step7-error-text').textContent = 'Error: ' + err.message;
      document.getElementById('step7-error').classList.remove('hidden');
    }
  },

  async handleClick(e) {
    if (!this.imageEmbeddings || !this.model) return;

    const overlay = document.getElementById('canvas-sam-overlay');
    const rect = overlay.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Convert to image coordinates (0-1 normalized)
    const canvas = document.getElementById('canvas-sam');
    const normX = x / canvas.width;
    const normY = y / canvas.height;

    // Draw click point
    const ctx = overlay.getContext('2d');
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    ctx.fillStyle = '#4a9eff';
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 6, 0, 2 * Math.PI);
    ctx.stroke();

    try {
      // Get the original image dimensions for proper coordinate mapping
      const imgW = this.originalImg.width;
      const imgH = this.originalImg.height;

      const input_points = [[[normX * imgW, normY * imgH]]];
      const input_labels = [[1]]; // 1 = foreground point

      const outputs = await this.model({
        ...this.imageEmbeddings,
        input_points,
        input_labels,
      });

      const masks = await this.processor.post_process_masks(
        outputs.pred_masks,
        this.imageProcessed.original_sizes,
        this.imageProcessed.reshaped_input_sizes,
      );

      // Draw the best mask as overlay
      const maskData = masks[0][0][0]; // first image, first mask set, best mask
      this.drawMask(ctx, canvas.width, canvas.height, maskData, imgW, imgH);

      // Redraw click point on top
      ctx.fillStyle = '#4a9eff';
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.stroke();

    } catch (err) {
      console.error('SAM segmentation error:', err);
    }
  },

  drawMask(ctx, canvasW, canvasH, maskData, imgW, imgH) {
    const imgData = ctx.createImageData(canvasW, canvasH);
    const data = maskData.data || maskData;
    const maskW = maskData.dims ? maskData.dims[1] : imgW;
    const maskH = maskData.dims ? maskData.dims[0] : imgH;

    for (let y = 0; y < canvasH; y++) {
      for (let x = 0; x < canvasW; x++) {
        // Map canvas coords to mask coords
        const mx = Math.floor((x / canvasW) * maskW);
        const my = Math.floor((y / canvasH) * maskH);
        const maskVal = data[my * maskW + mx];

        const idx = (y * canvasW + x) * 4;
        if (maskVal > 0) {
          imgData.data[idx]     = 74;   // R (accent blue)
          imgData.data[idx + 1] = 158;  // G
          imgData.data[idx + 2] = 255;  // B
          imgData.data[idx + 3] = 100;  // A (semi-transparent)
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }
};

/* ================================================================
   APP — Main orchestration
   ================================================================ */
const App = {
  currentStep: 1,
  previousStep: null,

  async init() {
    Step1.init();
    Step2.init();
    Step3.init();
    Step4.init();
    Step5.init();
    Step6.init();
    Step7.init();
    this.setupNav();
    this.hideLoader();
  },

  hideLoader() {
    const overlay = document.getElementById('loading-overlay');
    overlay.classList.add('hidden');
    setTimeout(() => overlay.remove(), 500);
  },

  setupNav() {
    document.querySelectorAll('.step-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const step = parseInt(tab.dataset.step);
        this.goToStep(step);
      });
    });
  },

  goToStep(n) {
    const prev = this.currentStep;
    this.currentStep = n;

    // Stop webcams when leaving steps 2 or 3
    if (prev === 2 && n !== 2) Step2.stop();
    if (prev === 3 && n !== 3) Step3.stop();

    // Show/hide steps
    document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.step-tab').forEach(t => t.classList.remove('active'));
    document.getElementById(`step-${n}`).classList.add('active');
    document.querySelector(`.step-tab[data-step="${n}"]`).classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });

    // Lazy-start webcam steps
    if (n === 2) Step2.start();
    if (n === 3) Step3.start();
  },

  /**
   * Load an image file into an HTMLImageElement.
   */
  loadImage(file) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  },

  /**
   * Draw an image onto a canvas, fitting to maxWidth.
   */
  drawToCanvas(canvas, img, maxWidth = 400) {
    const scale = Math.min(1, maxWidth / img.width);
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return ctx;
  },

  /**
   * Render prediction bars into a container.
   */
  renderPredictions(container, predictions) {
    container.innerHTML = predictions.map(p => {
      const pct = (p.probability * 100).toFixed(1);
      return `
        <div class="prediction-bar">
          <span class="prediction-label">${p.className}</span>
          <div class="prediction-track">
            <div class="prediction-fill" style="width:${pct}%"></div>
          </div>
          <span class="prediction-value">${pct}%</span>
        </div>
      `;
    }).join('');
  },
};

// Boot
document.addEventListener('DOMContentLoaded', () => App.init());
