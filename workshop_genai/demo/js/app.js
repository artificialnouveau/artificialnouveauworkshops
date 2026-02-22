/**
 * app.js — 4-tab Generative AI Workshop (Replicate API via local proxy)
 *   Tab 1: Text to Image (FLUX-schnell)
 *   Tab 2: Image to Image (SDXL)
 *   Tab 3: Image to Text (BLIP)
 *   Tab 4: PhotoMaker (consistent characters)
 */

/* ================================================================
   Utility — convert File to data URI
   ================================================================ */
function fileToDataURI(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

/* ================================================================
   TAB 1 — Text to Image (FLUX-schnell via Replicate)
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
    status.textContent = 'Sending request to FLUX...';
    fill.style.width = '30%';

    try {
      const res = await fetch('/api/txt2img', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });

      fill.style.width = '80%';
      status.textContent = 'Processing response...';

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Generation failed');
      if (!data.images || data.images.length === 0) throw new Error('No image returned');

      fill.style.width = '100%';
      status.textContent = 'Done!';

      const imgEl = document.getElementById('txt2img-output');
      imgEl.src = data.images[0];
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
   TAB 2 — Image to Image (SDXL via Replicate)
   ================================================================ */
const ImageToImage = {
  imageDataURI: null,

  init() {
    const fileInput = document.getElementById('file-img2img');
    const uploadArea = document.getElementById('upload-area-img2img');
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => this.handleUpload(e));

    const slider = document.getElementById('img2img-strength');
    const valLabel = document.getElementById('img2img-strength-val');
    slider.addEventListener('input', () => { valLabel.textContent = slider.value; });

    document.getElementById('btn-img2img').addEventListener('click', () => this.transform());
  },

  async handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      this.showError(`"${file.name}" is not an image file. Please upload a JPG, PNG, or WebP image.`);
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      this.showError(`"${file.name}" is too large (${(file.size / 1e6).toFixed(1)} MB). Please use an image under 20 MB.`);
      return;
    }

    try {
      this.imageDataURI = await fileToDataURI(file);

      const img = new Image();
      img.onload = () => {
        const canvas = document.getElementById('canvas-img2img-preview');
        App.drawToCanvas(canvas, img, 400);
        canvas.style.display = 'block';
        document.getElementById('upload-area-img2img').classList.add('has-file');
        document.getElementById('btn-img2img').disabled = false;
      };
      img.src = this.imageDataURI;

      this.hideError();
    } catch (err) {
      this.showError(`Could not load "${file.name}". Try a JPG or PNG instead.`);
    }
  },

  showError(msg) {
    document.getElementById('img2img-error-text').textContent = msg;
    document.getElementById('img2img-error').classList.remove('hidden');
  },

  hideError() {
    document.getElementById('img2img-error').classList.add('hidden');
  },

  async transform() {
    const btn = document.getElementById('btn-img2img');
    const prompt = document.getElementById('img2img-prompt').value.trim();
    const strength = parseFloat(document.getElementById('img2img-strength').value);

    if (!this.imageDataURI) return;
    if (!prompt) { this.showError('Please enter a prompt describing the transformation.'); return; }

    btn.disabled = true;
    btn.textContent = 'Transforming...';

    document.getElementById('img2img-results').classList.add('hidden');
    this.hideError();

    const loadingEl = document.getElementById('img2img-loading');
    const fill = document.getElementById('img2img-loading-fill');
    const status = document.getElementById('img2img-loading-status');

    loadingEl.classList.add('visible');
    status.textContent = 'Sending image to SDXL...';
    fill.style.width = '30%';

    try {
      const res = await fetch('/api/img2img', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, image: this.imageDataURI, strength })
      });

      fill.style.width = '80%';
      status.textContent = 'Processing response...';

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Transformation failed');
      if (!data.images || data.images.length === 0) throw new Error('No image returned');

      fill.style.width = '100%';
      status.textContent = 'Done!';

      document.getElementById('img2img-original').src = this.imageDataURI;
      const outputEl = document.getElementById('img2img-output');
      outputEl.src = data.images[0];
      outputEl.onload = () => {
        setTimeout(() => loadingEl.classList.remove('visible'), 600);
        document.getElementById('img2img-results').classList.remove('hidden');
      };

    } catch (err) {
      console.error('Image-to-image error:', err);
      loadingEl.classList.remove('visible');
      this.showError('Error: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Transform';
    }
  }
};

/* ================================================================
   TAB 3 — Image to Text (BLIP via Replicate)
   ================================================================ */
const ImageToText = {
  imageDataURI: null,

  init() {
    const fileInput = document.getElementById('file-img2txt');
    const uploadArea = document.getElementById('upload-area-img2txt');
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => this.handleUpload(e));
  },

  async handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      this.showError(`"${file.name}" is not an image file. Please upload a JPG, PNG, or WebP image.`);
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      this.showError(`"${file.name}" is too large. Please use an image under 20 MB.`);
      return;
    }

    try {
      this.imageDataURI = await fileToDataURI(file);

      const img = new Image();
      img.onload = () => {
        const canvas = document.getElementById('canvas-img2txt-preview');
        App.drawToCanvas(canvas, img, 500);
        canvas.style.display = 'block';
        document.getElementById('upload-area-img2txt').classList.add('has-file');
      };
      img.src = this.imageDataURI;

      this.hideError();
      document.getElementById('img2txt-results').classList.add('hidden');

      // Auto-caption on upload
      await this.caption();
    } catch (err) {
      this.showError(`Could not load "${file.name}". Try a JPG or PNG instead.`);
    }
  },

  showError(msg) {
    document.getElementById('img2txt-error-text').textContent = msg;
    document.getElementById('img2txt-error').classList.remove('hidden');
  },

  hideError() {
    document.getElementById('img2txt-error').classList.add('hidden');
  },

  async caption() {
    if (!this.imageDataURI) return;

    const loadingEl = document.getElementById('img2txt-loading');
    const fill = document.getElementById('img2txt-loading-fill');
    const status = document.getElementById('img2txt-loading-status');

    loadingEl.classList.add('visible');
    status.textContent = 'Sending image to BLIP...';
    fill.style.width = '30%';

    try {
      const res = await fetch('/api/img2txt', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: this.imageDataURI })
      });

      fill.style.width = '80%';
      status.textContent = 'Processing response...';

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Captioning failed');

      fill.style.width = '100%';
      status.textContent = 'Done!';

      document.getElementById('img2txt-caption').textContent = data.caption || 'No caption generated.';
      setTimeout(() => loadingEl.classList.remove('visible'), 600);
      document.getElementById('img2txt-results').classList.remove('hidden');

    } catch (err) {
      console.error('Image-to-text error:', err);
      loadingEl.classList.remove('visible');
      this.showError('Error: ' + err.message);
    }
  }
};

/* ================================================================
   TAB 4 — PhotoMaker (consistent characters via Replicate)
   ================================================================ */
const PhotoMaker = {
  imageDataURI: null,

  init() {
    const fileInput = document.getElementById('file-photomaker');
    const uploadArea = document.getElementById('upload-area-photomaker');
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => this.handleUpload(e));

    document.getElementById('btn-photomaker').addEventListener('click', () => this.generate());
  },

  async handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      this.showError(`"${file.name}" is not an image file. Please upload a JPG, PNG, or WebP image.`);
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      this.showError(`"${file.name}" is too large. Please use an image under 20 MB.`);
      return;
    }

    try {
      this.imageDataURI = await fileToDataURI(file);

      const img = new Image();
      img.onload = () => {
        const canvas = document.getElementById('canvas-photomaker-preview');
        App.drawToCanvas(canvas, img, 300);
        canvas.style.display = 'block';
        document.getElementById('upload-area-photomaker').classList.add('has-file');
        document.getElementById('btn-photomaker').disabled = false;
      };
      img.src = this.imageDataURI;

      this.hideError();
    } catch (err) {
      this.showError(`Could not load "${file.name}". Try a JPG or PNG instead.`);
    }
  },

  showError(msg) {
    document.getElementById('photomaker-error-text').textContent = msg;
    document.getElementById('photomaker-error').classList.remove('hidden');
  },

  hideError() {
    document.getElementById('photomaker-error').classList.add('hidden');
  },

  async generate() {
    const btn = document.getElementById('btn-photomaker');
    const prompt = document.getElementById('photomaker-prompt').value.trim();
    const style = document.getElementById('photomaker-style').value;

    if (!this.imageDataURI) return;
    if (!prompt) { this.showError('Please enter a prompt describing the scene.'); return; }

    btn.disabled = true;
    btn.textContent = 'Generating...';

    document.getElementById('photomaker-results').classList.add('hidden');
    this.hideError();

    const loadingEl = document.getElementById('photomaker-loading');
    const fill = document.getElementById('photomaker-loading-fill');
    const status = document.getElementById('photomaker-loading-status');

    loadingEl.classList.add('visible');
    status.textContent = 'Sending to PhotoMaker...';
    fill.style.width = '20%';

    try {
      const res = await fetch('/api/photomaker', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          image: this.imageDataURI,
          style,
          num_outputs: 2
        })
      });

      fill.style.width = '80%';
      status.textContent = 'Processing response...';

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Generation failed');
      if (!data.images || data.images.length === 0) throw new Error('No images returned');

      fill.style.width = '100%';
      status.textContent = 'Done!';

      const gallery = document.getElementById('photomaker-gallery');
      gallery.innerHTML = '';
      data.images.forEach(url => {
        const img = document.createElement('img');
        img.src = url;
        img.className = 'result-image';
        gallery.appendChild(img);
      });

      setTimeout(() => loadingEl.classList.remove('visible'), 600);
      document.getElementById('photomaker-results').classList.remove('hidden');

    } catch (err) {
      console.error('PhotoMaker error:', err);
      loadingEl.classList.remove('visible');
      this.showError('Error: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Generate Character';
    }
  }
};

/* ================================================================
   TAB 5 — Image to 3D (Hunyuan3D-2 via Replicate)
   ================================================================ */
const ImageTo3D = {
  imageDataURI: null,

  init() {
    const fileInput = document.getElementById('file-img3d');
    const uploadArea = document.getElementById('upload-area-img3d');
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', e => this.handleUpload(e));

    document.getElementById('btn-img3d').addEventListener('click', () => this.generate());
  },

  async handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
      this.showError(`"${file.name}" is not an image file. Please upload a JPG, PNG, or WebP image.`);
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      this.showError(`"${file.name}" is too large. Please use an image under 20 MB.`);
      return;
    }

    try {
      this.imageDataURI = await fileToDataURI(file);

      const img = new Image();
      img.onload = () => {
        const canvas = document.getElementById('canvas-img3d-preview');
        App.drawToCanvas(canvas, img, 400);
        canvas.style.display = 'block';
        document.getElementById('upload-area-img3d').classList.add('has-file');
        document.getElementById('btn-img3d').disabled = false;
      };
      img.src = this.imageDataURI;

      this.hideError();
    } catch (err) {
      this.showError(`Could not load "${file.name}". Try a JPG or PNG instead.`);
    }
  },

  showError(msg) {
    document.getElementById('img3d-error-text').textContent = msg;
    document.getElementById('img3d-error').classList.remove('hidden');
  },

  hideError() {
    document.getElementById('img3d-error').classList.add('hidden');
  },

  async generate() {
    const btn = document.getElementById('btn-img3d');

    if (!this.imageDataURI) return;

    btn.disabled = true;
    btn.textContent = 'Generating...';

    document.getElementById('img3d-results').classList.add('hidden');
    this.hideError();

    const loadingEl = document.getElementById('img3d-loading');
    const fill = document.getElementById('img3d-loading-fill');
    const status = document.getElementById('img3d-loading-status');

    loadingEl.classList.add('visible');
    status.textContent = 'Sending image to Hunyuan3D-2 (this may take 1-2 minutes)...';
    fill.style.width = '20%';

    try {
      const res = await fetch('/api/img3d', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: this.imageDataURI })
      });

      fill.style.width = '80%';
      status.textContent = 'Processing response...';

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || '3D generation failed');
      if (!data.mesh) throw new Error('No 3D model returned');

      fill.style.width = '100%';
      status.textContent = 'Done!';

      const viewer = document.getElementById('img3d-viewer');
      viewer.setAttribute('src', data.mesh);

      const downloadLink = document.getElementById('img3d-download');
      downloadLink.href = data.mesh;

      setTimeout(() => loadingEl.classList.remove('visible'), 600);
      document.getElementById('img3d-results').classList.remove('hidden');

    } catch (err) {
      console.error('Image-to-3D error:', err);
      loadingEl.classList.remove('visible');
      this.showError('Error: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Generate 3D Model';
    }
  }
};

/* ================================================================
   TAB 6 — Text to 3D (Shap-E via Replicate)
   ================================================================ */
const TextTo3D = {
  init() {
    document.getElementById('btn-txt3d').addEventListener('click', () => this.generate());
  },

  showError(msg) {
    document.getElementById('txt3d-error-text').textContent = msg;
    document.getElementById('txt3d-error').classList.remove('hidden');
  },

  hideError() {
    document.getElementById('txt3d-error').classList.add('hidden');
  },

  async generate() {
    const btn = document.getElementById('btn-txt3d');
    const prompt = document.getElementById('txt3d-prompt').value.trim();
    if (!prompt) return;

    btn.disabled = true;
    btn.textContent = 'Generating...';

    document.getElementById('txt3d-results').classList.add('hidden');
    this.hideError();

    const loadingEl = document.getElementById('txt3d-loading');
    const fill = document.getElementById('txt3d-loading-fill');
    const status = document.getElementById('txt3d-loading-status');

    loadingEl.classList.add('visible');
    status.textContent = 'Sending prompt to Shap-E (this may take a few minutes)...';
    fill.style.width = '20%';

    try {
      const res = await fetch('/api/txt3d', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt })
      });

      fill.style.width = '80%';
      status.textContent = 'Processing response...';

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || '3D generation failed');
      if (!data.files || data.files.length === 0) throw new Error('No output returned');

      fill.style.width = '100%';
      status.textContent = 'Done!';

      const gallery = document.getElementById('txt3d-gallery');
      gallery.innerHTML = '';

      data.files.forEach(url => {
        if (url.endsWith('.glb') || url.endsWith('.obj') || url.endsWith('.ply')) {
          // 3D mesh file — show with model-viewer + download link
          const container = document.createElement('div');
          container.innerHTML = `
            <model-viewer src="${url}" alt="Generated 3D model" auto-rotate camera-controls shadow-intensity="1" style="width:100%; height:350px; background:#111; border-radius:8px;"></model-viewer>
            <a href="${url}" class="btn-secondary" style="display:inline-block; margin-top:8px;" download>Download Model</a>
          `;
          gallery.appendChild(container);
        } else {
          // GIF/image preview
          const img = document.createElement('img');
          img.src = url;
          img.className = 'result-image';
          gallery.appendChild(img);
        }
      });

      setTimeout(() => loadingEl.classList.remove('visible'), 600);
      document.getElementById('txt3d-results').classList.remove('hidden');

    } catch (err) {
      console.error('Text-to-3D error:', err);
      loadingEl.classList.remove('visible');
      this.showError('Error: ' + err.message);
    } finally {
      btn.disabled = false;
      btn.textContent = 'Generate 3D Model';
    }
  }
};

/* ================================================================
   APP — Main orchestration
   ================================================================ */
const App = {
  currentStep: 1,

  init() {
    TextToImage.init();
    ImageToImage.init();
    ImageToText.init();
    PhotoMaker.init();
    ImageTo3D.init();
    TextTo3D.init();
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
