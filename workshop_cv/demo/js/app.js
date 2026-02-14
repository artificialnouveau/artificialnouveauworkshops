/**
 * app.js — Main orchestration: model loading, step navigation, shared utilities
 */

const App = {
  models: {
    mobilenet: null,
    blazeface: null,
    facemesh: null,
  },

  currentStep: 1,

  async init() {
    this.setupNav();
    await this.loadModels();
    this.hideLoader();
  },

  async loadModels() {
    try {
      // Load models in parallel
      const [mnet, bface] = await Promise.all([
        mobilenet.load({ version: 2, alpha: 1.0 }),
        blazeface.load(),
      ]);
      this.models.mobilenet = mnet;
      this.models.blazeface = bface;

      // Face landmarks model (heavier, load separately)
      this.models.facemesh = await faceLandmarksDetection.load(
        faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
        { maxFaces: 1 }
      );
    } catch (err) {
      console.error('Model loading error:', err);
      // Still hide loader — steps that need missing models will show errors
      this.hideLoader();
    }
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

  /**
   * Load an image file into an HTMLImageElement, returning a promise.
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
   * Draw an image onto a canvas, fitting it to maxWidth while maintaining aspect ratio.
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
   * Render a list of prediction bars into a container.
   */
  renderPredictions(container, predictions) {
    container.innerHTML = predictions.map(p => {
      const pct = (p.probability * 100).toFixed(1);
      return `
        <div class="prediction-bar">
          <span class="prediction-label">${p.className.split(',')[0]}</span>
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
