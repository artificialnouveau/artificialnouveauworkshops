/**
 * step1-nazi-detection.js — Image + video upload with backend-based detection
 */

const Step1 = {
  video: null,
  frames: [],
  backendUrl: 'http://localhost:8000',
  online: false,

  init() {
    this.video = document.getElementById('local-video');

    // Check backend
    this.checkBackend();

    // Image upload
    document.getElementById('file-input-1-img').addEventListener('change', (e) => this.handleImageUpload(e));

    // YouTube URL loading
    document.getElementById('btn-load-youtube').addEventListener('click', () => this.loadYouTube());

    // Local video upload
    document.getElementById('file-input-1').addEventListener('change', (e) => this.loadLocalVideo(e));

    // Scan frames button
    document.getElementById('btn-scan-frames').addEventListener('click', () => this.scanFrames());
  },

  async checkBackend() {
    const dot = document.getElementById('nazi-backend-dot');
    const label = document.getElementById('nazi-backend-label');
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
      label.textContent = 'Backend: offline — start server (cd backend && python server.py)';
      this.online = false;
    }
  },

  // ——— Image upload ———

  async handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!this.online) {
      await this.checkBackend();
      if (!this.online) {
        alert('Backend is offline. Start the server first:\ncd backend && python server.py');
        return;
      }
    }

    document.getElementById('upload-area-1-img').classList.add('has-file');

    // Show image on canvas
    const img = await App.loadImage(file);
    const canvas = document.getElementById('nazi-image-canvas');
    const ctx = App.drawToCanvas(canvas, img);

    const container = document.getElementById('image-result-container');
    container.classList.remove('hidden');

    const detectionsDiv = document.getElementById('nazi-image-detections');
    detectionsDiv.innerHTML = '<div class="loading-bar visible"><div class="loading-label"><span>Detecting<span class="loading-dots"></span></span></div><div class="loading-track"><div class="loading-fill" style="width:100%;animation:pulse 1.5s ease infinite"></div></div></div>';

    // Send to backend
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch(`${this.backendUrl}/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Server error ${res.status}`);
      }

      const data = await res.json();
      this.drawDetections(canvas, ctx, img, data.detections);

      if (data.detections.length > 0) {
        detectionsDiv.innerHTML = `<div class="face-card"><h4 style="color:var(--red)">${data.detections.length} detection(s) found</h4>${data.detections.map((d, i) => `<p class="description">Detection ${i + 1}: ${d.label || 'symbol'} — ${(d.confidence * 100).toFixed(1)}% confidence</p>`).join('')}</div>`;
      } else {
        detectionsDiv.innerHTML = '<p class="description">No detections found in this image.</p>';
      }
    } catch (err) {
      detectionsDiv.innerHTML = `<p class="description" style="color:var(--red)">Detection failed: ${err.message}</p>`;
    }
  },

  drawDetections(canvas, ctx, img, detections) {
    // Redraw original
    App.drawToCanvas(canvas, img);
    const newCtx = canvas.getContext('2d');

    const scaleX = canvas.width / img.width;
    const scaleY = canvas.height / img.height;

    newCtx.strokeStyle = '#ff4a4a';
    newCtx.lineWidth = 3;
    newCtx.font = '14px monospace';
    newCtx.fillStyle = '#ff4a4a';

    for (const det of detections) {
      const x = det.x * scaleX;
      const y = det.y * scaleY;
      const w = det.w * scaleX;
      const h = det.h * scaleY;
      newCtx.strokeRect(x, y, w, h);
      const label = `${det.label || 'symbol'} ${(det.confidence * 100).toFixed(0)}%`;
      newCtx.fillText(label, x, y - 6);
    }
  },

  // ——— YouTube ———

  loadYouTube() {
    const url = document.getElementById('youtube-url').value.trim();
    if (!url) return;

    const videoId = this.extractYouTubeId(url);
    if (!videoId) {
      alert('Could not parse YouTube URL. Try a standard youtube.com/watch?v=... link.');
      return;
    }

    const iframe = document.getElementById('youtube-iframe');
    iframe.src = `https://www.youtube-nocookie.com/embed/${videoId}`;
    document.getElementById('youtube-container').classList.remove('hidden');
    document.getElementById('local-video-container').classList.add('hidden');
  },

  extractYouTubeId(url) {
    const patterns = [
      /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})/,
      /youtube\.com\/shorts\/([a-zA-Z0-9_-]{11})/,
    ];
    for (const p of patterns) {
      const m = url.match(p);
      if (m) return m[1];
    }
    return null;
  },

  // ——— Video upload + frame scanning ———

  loadLocalVideo(e) {
    const file = e.target.files[0];
    if (!file) return;

    this.video.src = URL.createObjectURL(file);
    document.getElementById('local-video-container').classList.remove('hidden');
    document.getElementById('youtube-container').classList.add('hidden');
    document.getElementById('upload-area-1').classList.add('has-file');
  },

  async scanFrames() {
    const video = this.video;
    if (!video.src) return;

    if (!this.online) {
      await this.checkBackend();
      if (!this.online) {
        alert('Backend is offline. Start the server first:\ncd backend && python server.py');
        return;
      }
    }

    // Wait for metadata
    if (video.readyState < 1) {
      await new Promise(resolve => {
        video.addEventListener('loadedmetadata', resolve, { once: true });
      });
    }

    const duration = video.duration;
    if (!duration || !isFinite(duration)) {
      alert('Cannot read video duration. Make sure the file is a valid video.');
      return;
    }

    const totalFrames = Math.floor(duration); // 1 fps
    const progressBar = document.getElementById('scan-progress');
    const progressFill = document.getElementById('scan-progress-fill');
    const progressPct = document.getElementById('scan-progress-pct');
    const progressSteps = document.getElementById('scan-progress-steps');
    const grid = document.getElementById('detection-grid');

    progressBar.classList.add('visible');
    grid.innerHTML = '';
    this.frames = [];

    for (let i = 0; i < totalFrames; i++) {
      // Seek to time
      video.currentTime = i;
      await new Promise(resolve => {
        video.addEventListener('seeked', resolve, { once: true });
      });

      // Extract frame to canvas
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      // Send frame to backend for detection
      const detections = await this.detectFrame(canvas);

      // Draw bounding boxes if any
      if (detections.length > 0) {
        ctx.strokeStyle = '#ff4a4a';
        ctx.lineWidth = 3;
        ctx.font = '14px monospace';
        ctx.fillStyle = '#ff4a4a';
        for (const det of detections) {
          ctx.strokeRect(det.x, det.y, det.w, det.h);
          ctx.fillText(`${det.label || ''} ${(det.confidence * 100).toFixed(0)}%`, det.x, det.y - 4);
        }
      }

      // Create thumbnail in grid
      const thumb = document.createElement('canvas');
      const thumbWidth = 300;
      const scale = thumbWidth / canvas.width;
      thumb.width = thumbWidth;
      thumb.height = canvas.height * scale;
      thumb.getContext('2d').drawImage(canvas, 0, 0, thumb.width, thumb.height);

      const frameDiv = document.createElement('div');
      frameDiv.className = 'detection-frame' + (detections.length > 0 ? ' has-detection' : '');
      frameDiv.appendChild(thumb);

      const label = document.createElement('div');
      label.className = 'frame-label';
      label.textContent = `Frame ${i + 1} — ${this.formatTime(i)}` +
        (detections.length > 0 ? ` — ${detections.length} detection(s)` : '');
      frameDiv.appendChild(label);

      grid.appendChild(frameDiv);
      this.frames.push({ time: i, canvas, detections });

      // Update progress
      const pct = Math.round(((i + 1) / totalFrames) * 100);
      progressFill.style.width = pct + '%';
      progressPct.textContent = pct + '%';
      progressSteps.textContent = `Frame ${i + 1} / ${totalFrames}`;
    }

    // Done
    progressBar.classList.remove('visible');
  },

  /**
   * Send a canvas frame to the backend /detect endpoint as a JPEG blob.
   */
  async detectFrame(canvas) {
    try {
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
      const formData = new FormData();
      formData.append('file', blob, 'frame.jpg');

      const res = await fetch(`${this.backendUrl}/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) return [];
      const data = await res.json();
      return data.detections || [];
    } catch {
      return [];
    }
  },

  formatTime(seconds) {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  },
};

document.addEventListener('DOMContentLoaded', () => Step1.init());
