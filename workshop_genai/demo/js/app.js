/**
 * app.js — 3-tab Generative AI workshop:
 *   Tab 1: Text to Image (Stable Diffusion via WebGPU)
 *   Tab 2: Image to Image (Style Transfer via Magenta)
 *   Tab 3: Image to Text (Captioning via BLIP)
 */

/* ================================================================
   TAB 1 — Text to Image (Stable Diffusion via WebGPU)
   ================================================================ */
const TextToImage = {
  pipeline: null,
  loading: false,

  init() {
    document.getElementById('btn-generate').addEventListener('click', () => this.generate());
    document.getElementById('btn-generate').disabled = false;
  },

  async loadPipeline() {
    if (this.pipeline) return this.pipeline;
    if (this.loading) return null;
    this.loading = true;

    const loadingEl = document.getElementById('txt2img-loading');
    const fill = document.getElementById('txt2img-loading-fill');
    const status = document.getElementById('txt2img-loading-status');
    const pctEl = document.getElementById('txt2img-loading-pct');
    const loadText = document.getElementById('txt2img-loading-text');

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
    const prompt = document.getElementById('txt2img-prompt').value.trim();
    if (!prompt) return;

    if (!navigator.gpu) {
      document.getElementById('txt2img-webgpu-warning').classList.remove('hidden');
      document.getElementById('txt2img-error-text').textContent =
        'WebGPU is not available in this browser. Please use Chrome 113+ or Edge 113+ to generate images.';
      document.getElementById('txt2img-error').classList.remove('hidden');
      return;
    }

    btn.disabled = true;
    btn.textContent = 'Generating...';

    document.getElementById('txt2img-results').classList.add('hidden');
    document.getElementById('txt2img-error').classList.add('hidden');

    const loadingEl = document.getElementById('txt2img-loading');
    const fill = document.getElementById('txt2img-loading-fill');
    const status = document.getElementById('txt2img-loading-status');
    const loadText = document.getElementById('txt2img-loading-text');

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

      // Generate random latent (Box-Muller for normal distribution)
      status.textContent = 'Running UNet denoising...';
      fill.style.width = '70%';
      const latentShape = [1, 4, 64, 64];
      const latentData = new Float32Array(1 * 4 * 64 * 64);
      for (let i = 0; i < latentData.length; i++) {
        const u1 = Math.random();
        const u2 = Math.random();
        latentData[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }

      // Float32 to Float16 conversion
      const latentTensor = new OrtTensor('float16', new Uint16Array(latentData.length), latentShape);
      const f32 = latentData;
      const f16 = latentTensor.data;
      for (let i = 0; i < f32.length; i++) {
        const floatView = new Float32Array([f32[i]]);
        const intView = new Uint32Array(floatView.buffer);
        const bits = intView[0];
        const sign = (bits >> 16) & 0x8000;
        const exponent = ((bits >> 23) & 0xff) - 127 + 15;
        const mantissa = (bits >> 13) & 0x3ff;
        if (exponent <= 0) f16[i] = sign;
        else if (exponent >= 31) f16[i] = sign | 0x7c00;
        else f16[i] = sign | (exponent << 10) | mantissa;
      }

      // SD-Turbo: single-step inference
      const timestep = new Float32Array([999]);
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

      // Render to canvas
      const canvas = document.getElementById('canvas-generated');
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext('2d');
      const imgData = ctx.createImageData(512, 512);
      const pixels = imageData.data || imageData.cpuData;

      for (let y = 0; y < 512; y++) {
        for (let x = 0; x < 512; x++) {
          const idx = (y * 512 + x) * 4;
          const srcIdx = y * 512 + x;
          imgData.data[idx]     = Math.max(0, Math.min(255, (pixels[srcIdx] + 1) * 127.5));
          imgData.data[idx + 1] = Math.max(0, Math.min(255, (pixels[srcIdx + 512*512] + 1) * 127.5));
          imgData.data[idx + 2] = Math.max(0, Math.min(255, (pixels[srcIdx + 512*512*2] + 1) * 127.5));
          imgData.data[idx + 3] = 255;
        }
      }
      ctx.putImageData(imgData, 0, 0);

      fill.style.width = '100%';
      status.textContent = 'Done!';
      setTimeout(() => loadingEl.classList.remove('visible'), 800);
      document.getElementById('txt2img-results').classList.remove('hidden');

    } catch (err) {
      console.error('Text-to-image error:', err);
      loadingEl.classList.remove('visible');
      document.getElementById('txt2img-error-text').textContent =
        'Error: ' + err.message + '\n\nText-to-image requires Chrome 113+ with WebGPU enabled and ~4 GB GPU memory. This feature is experimental.';
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
