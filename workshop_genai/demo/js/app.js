/**
 * app.js — 3-tab Generative AI workshop:
 *   Tab 1: Text to Image (HuggingFace Inference API)
 *   Tab 2: Image to Image (Style Transfer via Magenta, runs locally)
 *   Tab 3: Image to Text (Captioning via BLIP, runs locally)
 */

/* ================================================================
   TAB 1 — Text to Image (HuggingFace Inference API)
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
    status.textContent = 'Sending prompt to Stable Diffusion...';
    fill.style.width = '30%';

    try {
      const token = document.getElementById('hf-token').value.trim();
      const headers = { 'Content-Type': 'application/json' };
      if (token) headers['Authorization'] = `Bearer ${token}`;

      // Try SDXL first, fall back to SD 1.5
      const models = [
        'stabilityai/stable-diffusion-xl-base-1.0',
        'runwayml/stable-diffusion-v1-5'
      ];

      let response = null;
      let lastError = null;

      for (const model of models) {
        status.textContent = `Trying ${model.split('/')[1]}...`;
        fill.style.width = '50%';

        try {
          response = await fetch(`https://api-inference.huggingface.co/models/${model}`, {
            method: 'POST',
            headers,
            body: JSON.stringify({ inputs: prompt })
          });

          if (response.ok) {
            const contentType = response.headers.get('content-type') || '';
            if (contentType.includes('image')) {
              break; // Success
            }
          }

          // Model might be loading — check for retry
          if (response.status === 503) {
            const data = await response.json();
            if (data.estimated_time) {
              status.textContent = `Model is loading, estimated wait: ${Math.ceil(data.estimated_time)}s...`;
              fill.style.width = '40%';
              // Wait and retry once
              await new Promise(r => setTimeout(r, Math.min(data.estimated_time * 1000, 30000)));
              response = await fetch(`https://api-inference.huggingface.co/models/${model}`, {
                method: 'POST',
                headers,
                body: JSON.stringify({ inputs: prompt })
              });
              if (response.ok) break;
            }
          }

          // Auth required or rate limited
          if (response.status === 401 || response.status === 403) {
            lastError = 'Authentication required. Please add a free HuggingFace token above.';
          } else if (response.status === 429) {
            lastError = 'Rate limited. Please wait a moment and try again, or add a HuggingFace token for higher limits.';
          } else {
            const text = await response.text();
            lastError = `Model ${model.split('/')[1]} returned ${response.status}: ${text}`;
          }
          response = null;
        } catch (fetchErr) {
          lastError = fetchErr.message;
          response = null;
        }
      }

      if (!response || !response.ok) {
        throw new Error(lastError || 'All models failed. Try adding a HuggingFace token.');
      }

      fill.style.width = '90%';
      status.textContent = 'Rendering image...';

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      const imgEl = document.getElementById('img-generated');
      imgEl.src = imageUrl;
      imgEl.onload = () => {
        fill.style.width = '100%';
        status.textContent = 'Done!';
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
      const result = await model.stylize(this.contentImg, this.styleImg);
      const canvas = document.getElementById('canvas-result');
      canvas.width = result.width;
      canvas.height = result.height;
      canvas.getContext('2d').drawImage(result, 0, 0);
      document.getElementById('img2img-results').classList.remove('hidden');
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

    const img = await App.loadImage(file);
    const canvas = document.getElementById('canvas-caption');
    App.drawToCanvas(canvas, img, 500);
    canvas.style.display = 'block';
    document.getElementById('upload-area-caption').classList.add('has-file');

    document.getElementById('img2txt-results').classList.add('hidden');
    document.getElementById('img2txt-error').classList.add('hidden');

    await this.caption(file);
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
      img.onerror = reject;
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
