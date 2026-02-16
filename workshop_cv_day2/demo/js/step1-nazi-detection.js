/**
 * step1-nazi-detection.js — Video frame extraction + placeholder detection model
 */

const Step1 = {
  video: null,
  frames: [],

  init() {
    this.video = document.getElementById('local-video');

    // YouTube URL loading
    document.getElementById('btn-load-youtube').addEventListener('click', () => this.loadYouTube());

    // Local video upload
    document.getElementById('file-input-1').addEventListener('change', (e) => this.loadLocalVideo(e));

    // Scan frames button
    document.getElementById('btn-scan-frames').addEventListener('click', () => this.scanFrames());
  },

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

      // Run detection (placeholder)
      const detections = await this.runDetection(canvas);

      // Draw bounding boxes if any
      if (detections.length > 0) {
        ctx.strokeStyle = '#ff4a4a';
        ctx.lineWidth = 3;
        for (const det of detections) {
          ctx.strokeRect(det.x, det.y, det.w, det.h);
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
   * Placeholder detection function.
   * Replace this with your actual TF.js / YOLO model inference.
   * Should return an array of { x, y, w, h, label, confidence }.
   */
  async runDetection(canvas) {
    // TODO: Load your model and run inference on the canvas ImageData
    // Example:
    //   const model = await tf.loadGraphModel('model/model.json');
    //   const tensor = tf.browser.fromPixels(canvas);
    //   const predictions = await model.predict(tensor);
    //   return parsePredictions(predictions);
    return [];
  },

  formatTime(seconds) {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  },
};

document.addEventListener('DOMContentLoaded', () => Step1.init());
