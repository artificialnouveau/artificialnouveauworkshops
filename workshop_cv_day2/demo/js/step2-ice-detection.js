/**
 * step2-ice-detection.js — Image upload + DeepFace backend for ethnicity classification
 */

const Step2 = {
  backendUrl: 'http://localhost:8000',
  online: false,

  init() {
    this.checkBackend();
    document.getElementById('file-input-2').addEventListener('change', (e) => this.handleUpload(e));
  },

  async checkBackend() {
    const dot = document.getElementById('backend-dot');
    const label = document.getElementById('backend-label');
    try {
      const res = await fetch(`${this.backendUrl}/health`, { signal: AbortSignal.timeout(3000) });
      const data = await res.json();
      if (data.status === 'ok') {
        dot.className = 'status-dot online';
        label.textContent = 'Backend: connected';
        this.online = true;
      } else {
        throw new Error('bad status');
      }
    } catch {
      dot.className = 'status-dot offline';
      label.textContent = 'Backend: offline — start the server first';
      this.online = false;
    }
  },

  async handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!this.online) {
      await this.checkBackend();
      if (!this.online) {
        alert('Backend is offline. Start the server first:\ncd backend && python server.py');
        return;
      }
    }

    document.getElementById('upload-area-2').classList.add('has-file');
    document.getElementById('ice-loading').classList.add('visible');
    document.getElementById('ice-results').classList.add('hidden');

    try {
      // Send image to backend
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${this.backendUrl}/analyze`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      this.renderResults(file, data);
    } catch (err) {
      alert('Analysis failed: ' + err.message);
    } finally {
      document.getElementById('ice-loading').classList.remove('visible');
    }
  },

  async renderResults(file, data) {
    const results = document.getElementById('ice-results');
    const canvas = document.getElementById('ice-canvas');
    const facesContainer = document.getElementById('ice-faces');

    // Draw the original image with face bounding boxes
    const img = await App.loadImage(file);
    const ctx = App.drawToCanvas(canvas, img);

    // Draw face boxes
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

    // Render ethnicity predictions per face
    facesContainer.innerHTML = data.faces.map((face, i) => {
      const race = face.race;
      const sorted = Object.entries(race).sort((a, b) => b[1] - a[1]);
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

    results.classList.remove('hidden');
  },
};

document.addEventListener('DOMContentLoaded', () => Step2.init());
