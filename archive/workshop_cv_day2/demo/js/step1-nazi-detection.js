/**
 * step1-nazi-detection.js — Swastika detection via OpenCV.js
 * Uses edge detection + contour analysis (no trained model).
 *
 * Detection heuristic for swastika-like shapes:
 *  - Polygon approximation has 12–20 vertices (the cross + 4 arm hooks)
 *  - Exactly ~4 significant convexity defects (the 4 concavities between arms)
 *  - Roughly square bounding box (aspect ratio 0.6–1.7)
 *  - Minimum area threshold to ignore noise
 */

const Step1 = {
  video: null,
  frames: [],
  cvReady: false,

  init() {
    this.video = document.getElementById('local-video');

    // Image upload
    document.getElementById('file-input-1-img').addEventListener('change', (e) => this.handleImageUpload(e));

    // YouTube URL loading
    document.getElementById('btn-load-youtube').addEventListener('click', () => this.loadYouTube());

    // Local video upload
    document.getElementById('file-input-1').addEventListener('change', (e) => this.loadLocalVideo(e));

    // Scan frames button
    document.getElementById('btn-scan-frames').addEventListener('click', () => this.scanFrames());

    // Wait for OpenCV.js
    this.waitForOpenCV();
  },

  waitForOpenCV() {
    const dot = document.getElementById('opencv-dot');
    const detail = document.getElementById('opencv-detail');

    // OpenCV.js sets cv.Mat when ready
    const check = () => {
      if (typeof cv !== 'undefined' && cv.Mat) {
        this.cvReady = true;
        dot.className = 'model-load-dot ready';
        detail.textContent = 'ready';
        return;
      }
      setTimeout(check, 200);
    };

    // Also listen for the onRuntimeInitialized callback
    if (typeof cv !== 'undefined' && cv.onRuntimeInitialized === undefined) {
      // cv object exists but might not be initialized
      window._opencvReady = () => setTimeout(check, 500);
    }

    // If cv has a promise-based loading
    if (typeof cv === 'object' && typeof cv.then === 'function') {
      cv.then(() => {
        this.cvReady = true;
        dot.className = 'model-load-dot ready';
        detail.textContent = 'ready';
      });
    } else {
      check();
    }
  },

  /**
   * Run swastika detection on a canvas using OpenCV.js.
   * Returns array of { x, y, w, h, label, confidence, vertices, defects }.
   */
  detectSwastikas(canvas) {
    const src = cv.imread(canvas);
    const gray = new cv.Mat();
    const blurred = new cv.Mat();
    const edges = new cv.Mat();
    const contours = new cv.MatVector();
    const hierarchy = new cv.Mat();

    // Preprocessing
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, blurred, new cv.Size(5, 5), 0);

    // Adaptive threshold to handle varying backgrounds
    const binary = new cv.Mat();
    cv.adaptiveThreshold(blurred, binary, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2);

    // Canny edge detection
    cv.Canny(blurred, edges, 50, 150);

    // Combine binary + edges for better contour detection
    const combined = new cv.Mat();
    cv.bitwise_or(binary, edges, combined);

    // Morphological close to connect broken edges
    const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
    cv.morphologyEx(combined, combined, cv.MORPH_CLOSE, kernel);

    // Find contours
    cv.findContours(combined, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    const detections = [];
    const imgArea = canvas.width * canvas.height;
    const minArea = imgArea * 0.001;  // Minimum 0.1% of image
    const maxArea = imgArea * 0.8;    // Maximum 80% of image

    for (let i = 0; i < contours.size(); i++) {
      const contour = contours.get(i);
      const area = cv.contourArea(contour);

      // Skip too small or too large
      if (area < minArea || area > maxArea) continue;

      // Bounding rect
      const rect = cv.boundingRect(contour);
      const aspect = rect.width / rect.height;

      // Swastikas are roughly square (aspect 0.5–2.0)
      if (aspect < 0.5 || aspect > 2.0) continue;

      // Polygon approximation
      const approx = new cv.Mat();
      const peri = cv.arcLength(contour, true);
      cv.approxPolyDP(contour, approx, 0.02 * peri, true);
      const vertices = approx.rows;

      // Convex hull + convexity defects
      const hull = new cv.Mat();
      const hullIndices = new cv.Mat();
      const defects = new cv.Mat();
      cv.convexHull(contour, hull);
      cv.convexHull(contour, hullIndices, false, false);

      let significantDefects = 0;
      try {
        cv.convexityDefects(contour, hullIndices, defects);
        // Count significant defects (depth > threshold)
        const depthThreshold = peri * 0.02;
        for (let j = 0; j < defects.rows; j++) {
          const depth = defects.intPtr(j, 0)[3] / 256.0;
          if (depth > depthThreshold) significantDefects++;
        }
      } catch {
        // convexityDefects can fail on simple shapes
      }

      // Solidity: area / convex hull area
      const hullArea = cv.contourArea(hull);
      const solidity = hullArea > 0 ? area / hullArea : 0;

      // Score the shape — how swastika-like is it?
      const score = this.scoreSwastika(vertices, significantDefects, aspect, solidity, area, imgArea);

      if (score > 0.4) {
        detections.push({
          x: rect.x,
          y: rect.y,
          w: rect.width,
          h: rect.height,
          label: 'swastika',
          confidence: Math.min(score, 1.0),
          vertices,
          defects: significantDefects,
        });
      }

      approx.delete();
      hull.delete();
      hullIndices.delete();
      defects.delete();
    }

    // Cleanup
    src.delete();
    gray.delete();
    blurred.delete();
    edges.delete();
    binary.delete();
    combined.delete();
    kernel.delete();
    contours.delete();
    hierarchy.delete();

    return detections;
  },

  /**
   * Score how likely a contour represents a swastika.
   * Returns 0–1 confidence score.
   */
  scoreSwastika(vertices, defects, aspect, solidity, area, imgArea) {
    let score = 0;

    // Vertices: swastikas typically approximate to 12–20 vertices
    if (vertices >= 10 && vertices <= 24) {
      const vertexScore = 1 - Math.abs(vertices - 16) / 10;
      score += vertexScore * 0.3;
    }

    // Convexity defects: swastikas have ~4 significant concavities
    if (defects >= 3 && defects <= 6) {
      const defectScore = 1 - Math.abs(defects - 4) / 3;
      score += defectScore * 0.35;
    }

    // Aspect ratio: should be roughly square
    const aspectScore = 1 - Math.abs(aspect - 1.0);
    score += Math.max(0, aspectScore) * 0.15;

    // Solidity: swastika has gaps between arms, so solidity ~0.4–0.75
    if (solidity >= 0.3 && solidity <= 0.85) {
      const solidityScore = 1 - Math.abs(solidity - 0.55) / 0.3;
      score += Math.max(0, solidityScore) * 0.2;
    }

    return score;
  },

  // ——— Image upload ———

  async handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    if (!this.cvReady) {
      alert('OpenCV.js is still loading. Please wait for the indicator to show "ready".');
      return;
    }

    document.getElementById('upload-area-1-img').classList.add('has-file');

    const img = await App.loadImage(file);
    const canvas = document.getElementById('nazi-image-canvas');
    App.drawToCanvas(canvas, img);

    const container = document.getElementById('image-result-container');
    container.classList.remove('hidden');

    const detectionsDiv = document.getElementById('nazi-image-detections');
    detectionsDiv.innerHTML = '<div class="loading-bar visible"><div class="loading-label"><span>Detecting<span class="loading-dots"></span></span></div><div class="loading-track"><div class="loading-fill" style="width:100%;animation:pulse 1.5s ease infinite"></div></div></div>';

    // Allow UI to update before heavy processing
    await new Promise(r => setTimeout(r, 50));

    try {
      const detections = this.detectSwastikas(canvas);
      this.drawDetections(canvas, img, detections);

      if (detections.length > 0) {
        detectionsDiv.innerHTML = `<div class="face-card"><h4 style="color:var(--red)">${detections.length} detection(s) found</h4>${detections.map((d, i) => `<p class="description">Detection ${i + 1}: ${d.label} — ${(d.confidence * 100).toFixed(0)}% confidence (${d.vertices} vertices, ${d.defects} defects)</p>`).join('')}</div>`;
      } else {
        detectionsDiv.innerHTML = '<p class="description">No swastika-like shapes detected in this image.</p>';
      }
    } catch (err) {
      detectionsDiv.innerHTML = `<p class="description" style="color:var(--red)">Detection failed: ${err.message}</p>`;
    }
  },

  drawDetections(canvas, img, detections) {
    App.drawToCanvas(canvas, img);
    const ctx = canvas.getContext('2d');

    ctx.strokeStyle = '#ff4a4a';
    ctx.lineWidth = 3;
    ctx.font = '14px monospace';
    ctx.fillStyle = '#ff4a4a';

    const scaleX = canvas.width / img.width;
    const scaleY = canvas.height / img.height;

    for (const det of detections) {
      // Coordinates are already in canvas space (from cv.imread of the canvas)
      ctx.strokeRect(det.x, det.y, det.w, det.h);
      const label = `${det.label} ${(det.confidence * 100).toFixed(0)}%`;
      ctx.fillText(label, det.x, det.y - 6);
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

    if (!this.cvReady) {
      alert('OpenCV.js is still loading. Please wait.');
      return;
    }

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

    const totalFrames = Math.floor(duration);
    const progressBar = document.getElementById('scan-progress');
    const progressFill = document.getElementById('scan-progress-fill');
    const progressPct = document.getElementById('scan-progress-pct');
    const progressSteps = document.getElementById('scan-progress-steps');
    const grid = document.getElementById('detection-grid');

    progressBar.classList.add('visible');
    grid.innerHTML = '';
    this.frames = [];

    for (let i = 0; i < totalFrames; i++) {
      video.currentTime = i;
      await new Promise(resolve => {
        video.addEventListener('seeked', resolve, { once: true });
      });

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      let detections = [];
      try {
        detections = this.detectSwastikas(canvas);
      } catch {
        // Continue on error
      }

      if (detections.length > 0) {
        ctx.strokeStyle = '#ff4a4a';
        ctx.lineWidth = 3;
        ctx.font = '14px monospace';
        ctx.fillStyle = '#ff4a4a';
        for (const det of detections) {
          ctx.strokeRect(det.x, det.y, det.w, det.h);
          ctx.fillText(`${det.label} ${(det.confidence * 100).toFixed(0)}%`, det.x, det.y - 4);
        }
      }

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

      const pct = Math.round(((i + 1) / totalFrames) * 100);
      progressFill.style.width = pct + '%';
      progressPct.textContent = pct + '%';
      progressSteps.textContent = `Frame ${i + 1} / ${totalFrames}`;

      // Yield for UI
      await new Promise(r => setTimeout(r, 10));
    }

    progressBar.classList.remove('visible');
  },

  formatTime(seconds) {
    const m = Math.floor(seconds / 60).toString().padStart(2, '0');
    const s = (seconds % 60).toString().padStart(2, '0');
    return `${m}:${s}`;
  },
};

document.addEventListener('DOMContentLoaded', () => Step1.init());
