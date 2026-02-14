/**
 * step2-surveillance.js — Full surveillance stack with toggleable layers
 * Layers: demographics (face-api.js), pose (BlazeFace), objects (COCO-SSD), content (NSFWJS)
 * Sources: image upload or webcam
 */

(function () {
  const fileInput = document.getElementById('file-input-2');
  const uploadArea = document.getElementById('upload-area-2');
  const webcamBtn = document.getElementById('webcam-btn');
  const results = document.getElementById('step2-results');
  const layerToggles = document.getElementById('layer-toggles');
  const dataDiv = document.getElementById('surveillance-data');
  const statsDiv = document.getElementById('face-stats');
  const canvas = document.getElementById('canvas-surveillance');
  const video = document.getElementById('webcam-video');
  const webcamOverlay = document.getElementById('canvas-webcam-overlay');

  const FACE_API_MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model/';

  // State
  let currentImg = null;
  let webcamActive = false;
  let webcamLoop = null;

  // Lazy-loaded model flags
  const loaded = {
    faceapi: false,
    cocoSsd: null,
    nsfw: null,
  };

  // Active layers
  const activeLayers = {
    demographics: true,
    pose: false,
    objects: false,
    nsfw: false,
  };

  // ── Toggle setup ──
  document.querySelectorAll('.toggle-chip').forEach(chip => {
    chip.addEventListener('click', async () => {
      const layer = chip.dataset.layer;
      const checkbox = chip.querySelector('input');
      checkbox.checked = !checkbox.checked;
      activeLayers[layer] = checkbox.checked;
      chip.classList.toggle('active', checkbox.checked);

      if (checkbox.checked) {
        chip.classList.add('loading');
        await ensureModel(layer);
        chip.classList.remove('loading');
      }

      if (currentImg && !webcamActive) {
        await analyzeImage(currentImg);
      }
    });
  });

  // ── Image upload ──
  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    stopWebcam();
    uploadArea.classList.add('has-file');
    layerToggles.classList.remove('hidden');
    results.classList.remove('hidden');
    canvas.classList.remove('hidden');
    video.classList.add('hidden');
    webcamOverlay.classList.add('hidden');

    currentImg = await App.loadImage(file);
    updateTimestamp();
    await ensureActiveModels();
    await analyzeImage(currentImg);
  });

  // ── Webcam ──
  webcamBtn.addEventListener('click', async () => {
    if (webcamActive) {
      stopWebcam();
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }
      });
      video.srcObject = stream;
      await video.play();

      webcamActive = true;
      webcamBtn.classList.add('active');
      webcamBtn.querySelector('span:last-child').textContent = 'Stop Webcam';
      layerToggles.classList.remove('hidden');
      results.classList.remove('hidden');
      canvas.classList.add('hidden');
      video.classList.remove('hidden');
      webcamOverlay.classList.remove('hidden');

      webcamOverlay.width = video.videoWidth || 640;
      webcamOverlay.height = video.videoHeight || 480;

      await ensureActiveModels();
      webcamAnalysisLoop();
    } catch (err) {
      console.error('Webcam error:', err);
      statsDiv.innerHTML = 'Webcam access denied or unavailable.';
    }
  });

  function stopWebcam() {
    webcamActive = false;
    if (webcamLoop) {
      clearTimeout(webcamLoop);
      webcamLoop = null;
    }
    if (video.srcObject) {
      video.srcObject.getTracks().forEach(t => t.stop());
      video.srcObject = null;
    }
    webcamBtn.classList.remove('active');
    webcamBtn.querySelector('span:last-child').textContent = 'Use Webcam';
    video.classList.add('hidden');
    webcamOverlay.classList.add('hidden');
  }

  async function webcamAnalysisLoop() {
    if (!webcamActive) return;

    if (video.videoWidth) {
      webcamOverlay.width = video.videoWidth;
      webcamOverlay.height = video.videoHeight;
    }

    const octx = webcamOverlay.getContext('2d');
    octx.clearRect(0, 0, webcamOverlay.width, webcamOverlay.height);
    updateTimestamp();

    const allData = [];

    if (activeLayers.demographics) await runDemographics(video, octx, allData);
    if (activeLayers.pose) await runPose(video, octx, allData);
    if (activeLayers.objects) await runObjectDetection(video, octx, allData);
    if (activeLayers.nsfw) await runNSFW(video, allData);

    renderDataReadout(allData);

    webcamLoop = setTimeout(() => {
      if (webcamActive) webcamAnalysisLoop();
    }, 300);
  }

  // ── Analyze static image ──
  async function analyzeImage(img) {
    const ctx = App.drawToCanvas(canvas, img, 600);
    updateTimestamp();
    drawScanLines(ctx, canvas.width, canvas.height);

    const allData = [];

    if (activeLayers.demographics) await runDemographics(canvas, ctx, allData);
    if (activeLayers.pose) await runPose(canvas, ctx, allData);
    if (activeLayers.objects) await runObjectDetection(canvas, ctx, allData);
    if (activeLayers.nsfw) await runNSFW(canvas, allData);

    renderDataReadout(allData);

    const activeList = Object.entries(activeLayers)
      .filter(([, v]) => v)
      .map(([k]) => k.toUpperCase())
      .join(', ') || 'NONE';

    statsDiv.innerHTML =
      `FRAME RESOLUTION: ${canvas.width}x${canvas.height}<br>` +
      `ACTIVE LAYERS: ${activeList}<br>` +
      `PROCESSING: CLIENT-SIDE (no data transmitted)`;
  }

  // ── Demographics via face-api.js ──
  async function runDemographics(source, ctx, allData) {
    if (!loaded.faceapi) return;

    let detections;
    try {
      detections = await faceapi
        .detectAllFaces(source, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.4 }))
        .withFaceLandmarks(true)
        .withAgeAndGender()
        .withFaceExpressions();
    } catch (err) {
      console.error('face-api error:', err);
      allData.push({ section: 'DEMOGRAPHICS' });
      allData.push({ key: 'ERROR', value: err.message });
      allData.push({ spacer: true });
      return;
    }

    if (!detections || detections.length === 0) {
      allData.push({ section: 'DEMOGRAPHICS' });
      allData.push({ key: 'STATUS', value: 'No faces detected' });
      allData.push({ spacer: true });
      return;
    }

    detections.forEach((det, idx) => {
      const box = det.detection.box;
      const x = box.x, y = box.y, w = box.width, h = box.height;
      const conf = (det.detection.score * 100).toFixed(0);

      // Green bounding box
      ctx.strokeStyle = '#00ff66';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
      drawBrackets(ctx, x, y, w, h, Math.min(w, h) * 0.15);

      // Label
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(x, y - 18, Math.max(w, 140), 18);
      ctx.fillStyle = '#00ff66';
      ctx.font = '11px monospace';
      ctx.fillText(`SUBJ-${String(idx + 1).padStart(2, '0')}  ${conf}%`, x + 4, y - 5);

      // Draw landmark points
      if (det.landmarks) {
        ctx.fillStyle = 'rgba(255, 74, 74, 0.5)';
        det.landmarks.positions.forEach(pt => {
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, 1.5, 0, Math.PI * 2);
          ctx.fill();
        });
      }

      // Data readout
      const age = Math.round(det.age);
      const gender = det.gender;
      const genderConf = (det.genderProbability * 100).toFixed(0);

      // Top expression
      const expressions = det.expressions;
      let topExpr = 'N/A', topExprConf = 0;
      if (expressions) {
        for (const [expr, score] of Object.entries(expressions)) {
          if (score > topExprConf) {
            topExpr = expr;
            topExprConf = score;
          }
        }
      }

      allData.push({ section: `SUBJECT ${idx + 1} — DEMOGRAPHICS` });
      allData.push({ key: 'AGE (estimated)', value: `${age} years` });
      allData.push({ key: 'GENDER', value: `${gender} (${genderConf}%)` });
      allData.push({ key: 'EXPRESSION', value: `${topExpr} (${(topExprConf * 100).toFixed(0)}%)` });
      allData.push({ key: 'FACE CONFIDENCE', value: `${conf}%` });
      allData.push({ key: 'LANDMARKS', value: `${det.landmarks ? det.landmarks.positions.length : 0} points` });
      allData.push({ key: 'BBOX', value: `${Math.round(w)}x${Math.round(h)}px at (${Math.round(x)},${Math.round(y)})` });
      allData.push({ spacer: true });
    });
  }

  // ── Pose via BlazeFace (face keypoints as lightweight pose proxy) ──
  async function runPose(source, ctx, allData) {
    if (!App.models.blazeface) return;

    let predictions;
    try {
      predictions = await App.models.blazeface.estimateFaces(source, false);
    } catch (err) {
      console.error('BlazeFace pose error:', err);
      return;
    }

    allData.push({ section: 'POSE / KEYPOINTS' });

    if (!predictions || predictions.length === 0) {
      allData.push({ key: 'STATUS', value: 'No faces/poses detected' });
      allData.push({ spacer: true });
      return;
    }

    ctx.fillStyle = '#ff4a4a';
    ctx.strokeStyle = '#ff4a4a';
    ctx.lineWidth = 2;

    const landmarkNames = ['right eye', 'left eye', 'nose', 'mouth', 'right ear', 'left ear'];

    predictions.forEach((face, idx) => {
      if (face.landmarks) {
        // Draw keypoints
        face.landmarks.forEach((point, pi) => {
          ctx.beginPath();
          ctx.arc(point[0], point[1], 5, 0, Math.PI * 2);
          ctx.fill();

          // Label
          ctx.fillStyle = 'rgba(0,0,0,0.7)';
          ctx.fillRect(point[0] + 7, point[1] - 6, 60, 14);
          ctx.fillStyle = '#ff4a4a';
          ctx.font = '10px monospace';
          ctx.fillText(landmarkNames[pi] || `pt${pi}`, point[0] + 9, point[1] + 5);
          ctx.fillStyle = '#ff4a4a';
        });

        // Connect eyes
        if (face.landmarks.length >= 2) {
          ctx.beginPath();
          ctx.moveTo(face.landmarks[0][0], face.landmarks[0][1]);
          ctx.lineTo(face.landmarks[1][0], face.landmarks[1][1]);
          ctx.stroke();
        }
        // Connect to nose
        if (face.landmarks.length >= 3) {
          ctx.beginPath();
          ctx.moveTo(face.landmarks[0][0], face.landmarks[0][1]);
          ctx.lineTo(face.landmarks[2][0], face.landmarks[2][1]);
          ctx.lineTo(face.landmarks[1][0], face.landmarks[1][1]);
          ctx.stroke();
        }
        // Connect to mouth
        if (face.landmarks.length >= 4) {
          ctx.beginPath();
          ctx.moveTo(face.landmarks[2][0], face.landmarks[2][1]);
          ctx.lineTo(face.landmarks[3][0], face.landmarks[3][1]);
          ctx.stroke();
        }

        allData.push({ key: `FACE ${idx + 1} KEYPOINTS`, value: `${face.landmarks.length} landmarks` });
        face.landmarks.forEach((pt, pi) => {
          allData.push({ key: `  ${(landmarkNames[pi] || 'pt' + pi).toUpperCase()}`, value: `(${Math.round(pt[0])}, ${Math.round(pt[1])})` });
        });
      }
    });
    allData.push({ spacer: true });
  }

  // ── COCO-SSD: object detection ──
  async function runObjectDetection(source, ctx, allData) {
    if (!loaded.cocoSsd) return;

    let predictions;
    try {
      predictions = await loaded.cocoSsd.detect(source);
    } catch (err) {
      console.error('COCO-SSD error:', err);
      return;
    }

    allData.push({ section: 'OBJECT DETECTION' });

    if (predictions.length === 0) {
      allData.push({ key: 'STATUS', value: 'No objects detected' });
      allData.push({ spacer: true });
      return;
    }

    ctx.lineWidth = 2;
    ctx.font = '11px monospace';
    predictions.forEach(pred => {
      const [x, y, w, h] = pred.bbox;
      const label = pred.class;
      const conf = (pred.score * 100).toFixed(0);

      ctx.strokeStyle = '#ffd94a';
      ctx.strokeRect(x, y, w, h);
      const textW = ctx.measureText(`${label} ${conf}%`).width + 10;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x, y - 16, textW, 16);
      ctx.fillStyle = '#ffd94a';
      ctx.fillText(`${label} ${conf}%`, x + 4, y - 4);

      allData.push({ key: label.toUpperCase(), value: `${conf}% — ${Math.round(w)}x${Math.round(h)}px` });
    });
    allData.push({ spacer: true });
  }

  // ── NSFWJS: content classification ──
  async function runNSFW(source, allData) {
    if (!loaded.nsfw) return;

    let inputEl = source;
    if (source instanceof HTMLVideoElement) {
      const tmp = document.createElement('canvas');
      tmp.width = source.videoWidth;
      tmp.height = source.videoHeight;
      tmp.getContext('2d').drawImage(source, 0, 0);
      inputEl = tmp;
    }

    let predictions;
    try {
      predictions = await loaded.nsfw.classify(inputEl);
    } catch (err) {
      console.error('NSFW error:', err);
      return;
    }

    allData.push({ section: 'CONTENT CLASSIFICATION' });
    predictions.forEach(p => {
      allData.push({ key: p.className, value: `${(p.probability * 100).toFixed(1)}%` });
    });
    allData.push({ spacer: true });
  }

  // ── Model loading ──
  async function ensureModel(layer) {
    if (layer === 'demographics' && !loaded.faceapi) {
      try {
        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(FACE_API_MODEL_URL),
          faceapi.nets.faceLandmark68TinyNet.loadFromUri(FACE_API_MODEL_URL),
          faceapi.nets.ageGenderNet.loadFromUri(FACE_API_MODEL_URL),
          faceapi.nets.faceExpressionNet.loadFromUri(FACE_API_MODEL_URL),
        ]);
        loaded.faceapi = true;
        console.log('face-api.js models loaded');
      } catch (err) {
        console.error('face-api.js failed to load:', err);
      }
    }

    if (layer === 'pose') {
      // Uses BlazeFace from App.models (already loaded at startup)
      if (!App.models.blazeface) {
        console.warn('BlazeFace not available for pose');
      }
    }

    if (layer === 'objects' && !loaded.cocoSsd) {
      try {
        loaded.cocoSsd = await cocoSsd.load();
        console.log('COCO-SSD loaded');
      } catch (err) {
        console.error('COCO-SSD failed:', err);
      }
    }

    if (layer === 'nsfw' && !loaded.nsfw) {
      try {
        loaded.nsfw = await nsfwjs.load();
        console.log('NSFWJS loaded');
      } catch (err) {
        console.error('NSFWJS failed:', err);
      }
    }
  }

  async function ensureActiveModels() {
    const promises = [];
    for (const [layer, active] of Object.entries(activeLayers)) {
      if (active) promises.push(ensureModel(layer));
    }
    await Promise.all(promises);
  }

  // ── Rendering ──
  function renderDataReadout(allData) {
    let html = '';
    for (const item of allData) {
      if (item.section) {
        html += `<div class="data-row" style="color:var(--accent);margin-top:8px">${item.section} ━━━━━━━━━━</div>`;
      } else if (item.spacer) {
        html += '<div class="data-row">&nbsp;</div>';
      } else {
        html += `<div class="data-row">${item.key}: ${item.value}</div>`;
      }
    }
    if (allData.length === 0) {
      html = '<div class="data-row">NO DETECTIONS — Upload a photo or enable layers above.</div>';
    }
    dataDiv.innerHTML = html;
  }

  function updateTimestamp() {
    const now = new Date();
    const el = document.getElementById('timestamp');
    if (el) el.textContent = now.toISOString().replace('T', ' ').slice(0, 19);
  }

  // ── Drawing helpers ──
  function drawBrackets(ctx, x, y, w, h, len) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#00ff66';
    ctx.beginPath(); ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x, y + h - len); ctx.lineTo(x, y + h); ctx.lineTo(x + len, y + h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x + w - len, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - len); ctx.stroke();
  }

  function drawScanLines(ctx, w, h) {
    ctx.strokeStyle = 'rgba(0, 255, 102, 0.03)';
    ctx.lineWidth = 1;
    for (let y = 0; y < h; y += 3) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }
  }
})();
