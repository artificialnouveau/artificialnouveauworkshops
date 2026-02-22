/**
 * app.js — 3-tab Generative AI workshop:
 *   Tab 1: Text to Image (AI Horde — free, no auth)
 *   Tab 2: Image to Image (Style Transfer via Magenta, runs locally)
 *   Tab 3: Image to Text (Captioning via BLIP, runs locally)
 */

/* ================================================================
   TAB 1 — Text to Image (AI Horde — free, no login)
   ================================================================ */
const TextToImage = {
  init() {
    document.getElementById('btn-generate').addEventListener('click', () => this.generate());
  },

  async generate() {
    const btn = document.getElementById('btn-generate');
    const prompt = document.getElementById('txt2img-prompt').value.trim();
    if (!prompt) return;

    btn.disabled = true;
    btn.textContent = 'Generating...';

    document.getElementById('txt2img-results').classList.add('hidden');
    document.getElementById('txt2img-error').classList.add('hidden');

    const loadingEl = document.getElementById('txt2img-loading');
    const fill = document.getElementById('txt2img-loading-fill');
    const status = document.getElementById('txt2img-loading-status');

    loadingEl.classList.add('visible');
    status.textContent = 'Submitting to AI Horde...';
    fill.style.width = '10%';

    try {
      // Step 1: Submit async generation request
      const submitRes = await fetch('https://aihorde.net/api/v2/generate/async', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'apikey': '0000000000'
        },
        body: JSON.stringify({
          prompt: prompt,
          params: { width: 512, height: 512, steps: 25 },
          nsfw: false,
          models: ['stable_diffusion']
        })
      });

      if (!submitRes.ok) {
        const errData = await submitRes.json().catch(() => ({}));
        throw new Error(errData.message || `Submission failed (status ${submitRes.status})`);
      }

      const { id } = await submitRes.json();
      fill.style.width = '20%';
      status.textContent = 'Queued — waiting for a worker to pick this up...';

      // Step 2: Poll for completion
      let done = false;
      let attempts = 0;
      const maxAttempts = 60; // 2 minutes max

      while (!done && attempts < maxAttempts) {
        await new Promise(r => setTimeout(r, 2000));
        attempts++;

        const checkRes = await fetch(`https://aihorde.net/api/v2/generate/check/${id}`);
        const checkData = await checkRes.json();

        if (checkData.done) {
          done = true;
          break;
        }

        if (checkData.faulted) {
          throw new Error('Generation failed on the worker. Please try again.');
        }

        const waitTime = checkData.wait_time || 0;
        const queuePos = checkData.queue_position || 0;
        const pct = checkData.processing ? 60 : Math.min(20 + attempts, 50);
        fill.style.width = pct + '%';

        if (checkData.processing) {
          status.textContent = 'A worker is generating your image...';
        } else if (queuePos > 0) {
          status.textContent = `Queue position: ${queuePos} — estimated wait: ${waitTime}s`;
        } else {
          status.textContent = 'Waiting for worker...';
        }
      }

      if (!done) {
        throw new Error('Generation timed out after 2 minutes. The queue may be busy — try again.');
      }

      // Step 3: Fetch the result
      fill.style.width = '80%';
      status.textContent = 'Downloading generated image...';

      const resultRes = await fetch(`https://aihorde.net/api/v2/generate/status/${id}`);
      const resultData = await resultRes.json();

      if (!resultData.generations || resultData.generations.length === 0) {
        throw new Error('No image was returned. Please try again.');
      }

      const imgBase64 = resultData.generations[0].img;
      const imgEl = document.getElementById('img-generated');
      imgEl.src = `data:image/webp;base64,${imgBase64}`;

      fill.style.width = '100%';
      status.textContent = 'Done!';

      imgEl.onload = () => {
        setTimeout(() => loadingEl.classList.remove('visible'), 600);
        document.getElementById('txt2img-results').classList.remove('hidden');
      };

    } catch (err) {
      console.error('Text-to-image error:', err);
      loadingEl.classList.remove('visible');
      document.getElementById('txt2img-error-text').textContent = 'Error: ' + err.message;
      document.getElementById('txt2img-error').classList.remove('hidden');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Generate Image';
    }
  }
};

/* ================================================================
   TAB 2 — Image to Image (Style Transfer via Magenta)
   ================================================================ */
const ImageToImage = {
  model: null,
  contentImg: null,
  styleImg: null,

  init() {
    const contentInput = document.getElementById('file-content');
    const styleInput = document.getElementById('file-style');
    contentInput.addEventListener('change', e => this.handleUpload(e, 'content'));
    styleInput.addEventListener('change', e => this.handleUpload(e, 'style'));
    document.getElementById('upload-area-content').addEventListener('click', () => contentInput.click());
    document.getElementById('upload-area-style').addEventListener('click', () => styleInput.click());
    document.getElementById('btn-stylize').addEventListener('click', () => this.stylize());
  },

  async handleUpload(e, type) {
    const file = e.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      this.showError(`"${file.name}" is not an image file. Please upload a JPG, PNG, or WebP image.`);
      return;
    }

    // Validate file size (max 20MB)
    if (file.size > 20 * 1024 * 1024) {
      this.showError(`"${file.name}" is too large (${(file.size / 1e6).toFixed(1)} MB). Please use an image under 20 MB.`);
      return;
    }

    try {
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

      this.hideError();
      if (this.contentImg && this.styleImg) {
        document.getElementById('btn-stylize').disabled = false;
      }
    } catch (err) {
      this.showError(`Could not load "${file.name}". The file may be corrupted or in an unsupported format. Try a JPG or PNG instead.`);
    }
  },

  showError(msg) {
    let el = document.getElementById('img2img-error');
    if (!el) {
      el = document.createElement('div');
      el.id = 'img2img-error';
      el.className = 'output-block mono';
      el.style.cssText = 'border-left-color:var(--red); color:var(--red); margin-bottom:24px;';
      document.getElementById('btn-stylize').before(el);
    }
    el.textContent = msg;
    el.classList.remove('hidden');
  },

  hideError() {
    const el = document.getElementById('img2img-error');
    if (el) el.classList.add('hidden');
  },

  async loadModel() {
    if (this.model) return this.model;
    const loading = document.getElementById('img2img-loading');
    const fill = document.getElementById('img2img-loading-fill');
    const status = document.getElementById('img2img-loading-status');
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

      // Magenta stylize() returns an HTMLCanvasElement
      const resultCanvas = await model.stylize(this.contentImg, this.styleImg);

      const outputCanvas = document.getElementById('canvas-result');

      // Handle both HTMLCanvasElement and ImageData returns
      if (resultCanvas instanceof HTMLCanvasElement) {
        outputCanvas.width = resultCanvas.width;
        outputCanvas.height = resultCanvas.height;
        outputCanvas.getContext('2d').drawImage(resultCanvas, 0, 0);
      } else if (resultCanvas instanceof ImageData) {
        outputCanvas.width = resultCanvas.width;
        outputCanvas.height = resultCanvas.height;
        outputCanvas.getContext('2d').putImageData(resultCanvas, 0, 0);
      } else if (resultCanvas && resultCanvas.data) {
        // Tensor-like object — convert to ImageData
        const w = resultCanvas.width || this.contentImg.width;
        const h = resultCanvas.height || this.contentImg.height;
        outputCanvas.width = w;
        outputCanvas.height = h;
        const ctx = outputCanvas.getContext('2d');
        const imgData = ctx.createImageData(w, h);
        const src = resultCanvas.data;
        for (let i = 0; i < w * h; i++) {
          imgData.data[i * 4]     = Math.round(src[i * 3] * 255);
          imgData.data[i * 4 + 1] = Math.round(src[i * 3 + 1] * 255);
          imgData.data[i * 4 + 2] = Math.round(src[i * 3 + 2] * 255);
          imgData.data[i * 4 + 3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
      } else {
        throw new Error('Unexpected output format from style transfer model.');
      }

      document.getElementById('img2img-results').classList.remove('hidden');
    } catch (err) {
      console.error('Style transfer error:', err);
      this.showError('Style transfer failed: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Stylize';
    }
  }
};

/* ================================================================
   TAB 3 — Image to Text (Captioning via BLIP)
   ================================================================ */
const ImageToText = {
  pipeline: null,
  loading: false,

  init() {
    const captionInput = document.getElementById('file-caption');
    captionInput.addEventListener('change', e => this.handleUpload(e));
    document.getElementById('upload-area-caption').addEventListener('click', () => captionInput.click());
  },

  async handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      document.getElementById('img2txt-error-text').textContent =
        `"${file.name}" is not an image file. Please upload a JPG, PNG, or WebP image.`;
      document.getElementById('img2txt-error').classList.remove('hidden');
      return;
    }

    try {
      const img = await App.loadImage(file);
      const canvas = document.getElementById('canvas-caption');
      App.drawToCanvas(canvas, img, 500);
      canvas.style.display = 'block';
      document.getElementById('upload-area-caption').classList.add('has-file');

      document.getElementById('img2txt-results').classList.add('hidden');
      document.getElementById('img2txt-error').classList.add('hidden');

      await this.caption(file);
    } catch (err) {
      document.getElementById('img2txt-error-text').textContent =
        `Could not load "${file.name}". The file may be corrupted or unsupported. Try a JPG or PNG instead.`;
      document.getElementById('img2txt-error').classList.remove('hidden');
    }
  },

  async caption(file) {
    const loadingEl = document.getElementById('img2txt-loading');
    const fill = document.getElementById('img2txt-loading-fill');
    const status = document.getElementById('img2txt-loading-status');
    const pctEl = document.getElementById('img2txt-loading-pct');

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
      document.getElementById('img2txt-caption').textContent = caption;
      document.getElementById('img2txt-results').classList.remove('hidden');

    } catch (err) {
      console.error('Captioning error:', err);
      this.loading = false;
      loadingEl.classList.remove('visible');
      document.getElementById('img2txt-error-text').textContent = 'Error: ' + err.message;
      document.getElementById('img2txt-error').classList.remove('hidden');
    }
  }
};

/* ================================================================
   APP — Main orchestration
   ================================================================ */
const App = {
  currentStep: 1,

  async init() {
    TextToImage.init();
    ImageToImage.init();
    ImageToText.init();
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
    this.currentStep = n;
    document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.step-tab').forEach(t => t.classList.remove('active'));
    document.getElementById(`step-${n}`).classList.add('active');
    document.querySelector(`.step-tab[data-step="${n}"]`).classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  },

  loadImage(file) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error('Failed to load image'));
      img.src = URL.createObjectURL(file);
    });
  },

  drawToCanvas(canvas, img, maxWidth = 400) {
    const scale = Math.min(1, maxWidth / img.width);
    canvas.width = img.width * scale;
    canvas.height = img.height * scale;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    return ctx;
  },
};

// Boot
document.addEventListener('DOMContentLoaded', () => App.init());
