/**
 * step2-surveillance.js — Full surveillance stack with toggleable layers
 * Uses: BlazeFace (face boxes), face-api.js nobundle (age/gender/expression),
 *       COCO-SSD (objects), NSFWJS (content classification)
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

  let currentImg = null;
  let webcamActive = false;
  let webcamLoop = null;

  const models = { cocoSsd: null, nsfw: null, faceApiLoaded: false };

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
      const cb = chip.querySelector('input');
      cb.checked = !cb.checked;
      activeLayers[layer] = cb.checked;
      chip.classList.toggle('active', cb.checked);

      if (cb.checked) {
        chip.classList.add('loading');
        await ensureModel(layer);
        chip.classList.remove('loading');
      }

      if (currentImg && !webcamActive) await analyzeImage(currentImg);
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
    if (webcamActive) { stopWebcam(); return; }
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
    if (webcamLoop) { clearTimeout(webcamLoop); webcamLoop = null; }
    if (video.srcObject) { video.srcObject.getTracks().forEach(t => t.stop()); video.srcObject = null; }
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
    if (activeLayers.demographics) await runFaceDetection(video, octx, allData);
    if (activeLayers.pose) await runPose(video, octx, allData);
    if (activeLayers.objects) await runObjects(video, octx, allData);
    if (activeLayers.nsfw) await runNSFW(video, allData);
    renderData(allData);

    webcamLoop = setTimeout(() => { if (webcamActive) webcamAnalysisLoop(); }, 300);
  }

  // ── Analyze static image ──
  async function analyzeImage(img) {
    const ctx = App.drawToCanvas(canvas, img, 600);
    updateTimestamp();
    drawScanLines(ctx, canvas.width, canvas.height);

    const allData = [];
    if (activeLayers.demographics) await runFaceDetection(canvas, ctx, allData);
    if (activeLayers.pose) await runPose(canvas, ctx, allData);
    if (activeLayers.objects) await runObjects(canvas, ctx, allData);
    if (activeLayers.nsfw) await runNSFW(canvas, allData);
    renderData(allData);

    const active = Object.entries(activeLayers).filter(([,v]) => v).map(([k]) => k.toUpperCase()).join(', ') || 'NONE';
    statsDiv.innerHTML =
      `FRAME RESOLUTION: ${canvas.width}x${canvas.height}<br>` +
      `ACTIVE LAYERS: ${active}<br>` +
      `PROCESSING: CLIENT-SIDE (no data transmitted)`;
  }

  // ── Face Detection via BlazeFace + face-api.js for age/gender/expression ──
  async function runFaceDetection(source, ctx, allData) {
    if (!App.models.blazeface) {
      allData.push({ section: 'FACE DETECTION' });
      allData.push({ key: 'ERROR', value: 'BlazeFace model not loaded — reload the page' });
      allData.push({ spacer: true });
      return;
    }

    let faces;
    try {
      faces = await App.models.blazeface.estimateFaces(source, false);
    } catch (err) {
      console.error('BlazeFace error:', err);
      allData.push({ section: 'FACE DETECTION' });
      allData.push({ key: 'ERROR', value: err.message });
      allData.push({ spacer: true });
      return;
    }

    if (!faces || faces.length === 0) {
      allData.push({ section: 'FACE DETECTION' });
      allData.push({ key: 'STATUS', value: 'No faces detected' });
      allData.push({ spacer: true });
      return;
    }

    // Run face-api.js for age/gender/expression if available
    let faceApiResults = null;
    if (models.faceApiLoaded && typeof faceapi !== 'undefined') {
      try {
        // face-api.js needs a canvas or image element
        let inputEl = source;
        if (source instanceof HTMLVideoElement) {
          const tmp = document.createElement('canvas');
          tmp.width = source.videoWidth;
          tmp.height = source.videoHeight;
          tmp.getContext('2d').drawImage(source, 0, 0);
          inputEl = tmp;
        }
        faceApiResults = await faceapi
          .detectAllFaces(inputEl, new faceapi.TinyFaceDetectorOptions())
          .withAgeAndGender()
          .withFaceExpressions();
      } catch (err) {
        console.error('face-api.js analysis error:', err);
      }
    }

    const landmarkNames = ['right eye', 'left eye', 'nose', 'mouth', 'right ear', 'left ear'];

    faces.forEach((face, idx) => {
      const x1 = face.topLeft[0];
      const y1 = face.topLeft[1];
      const x2 = face.bottomRight[0];
      const y2 = face.bottomRight[1];
      const w = x2 - x1;
      const h = y2 - y1;
      const conf = (face.probability[0] * 100).toFixed(0);

      // Try to match this face with a face-api.js result by bounding box overlap
      let ageGenderResult = null;
      if (faceApiResults && faceApiResults.length > 0) {
        ageGenderResult = matchFaceApiResult(faceApiResults, x1, y1, w, h, idx);
      }

      // Green bounding box with brackets
      ctx.strokeStyle = '#00ff66';
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, w, h);
      drawBrackets(ctx, x1, y1, w, h, Math.min(w, h) * 0.15);

      // Label — include age/gender if available
      let labelText = `SUBJ-${String(idx + 1).padStart(2, '0')}  CONF:${conf}%`;
      if (ageGenderResult) {
        const age = Math.round(ageGenderResult.age);
        const gender = ageGenderResult.gender.toUpperCase();
        const genderConf = (ageGenderResult.genderProbability * 100).toFixed(0);
        labelText = `SUBJ-${String(idx + 1).padStart(2, '0')}  AGE:${age}  ${gender}(${genderConf}%)`;
      }

      const labelW = Math.max(ctx.measureText(labelText).width + 10, w);
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(x1, y1 - 18, labelW, 18);
      ctx.fillStyle = '#00ff66';
      ctx.font = '11px monospace';
      ctx.fillText(labelText, x1 + 4, y1 - 5);

      // Draw landmark points
      if (face.landmarks) {
        face.landmarks.forEach((pt, pi) => {
          ctx.beginPath();
          ctx.arc(pt[0], pt[1], 4, 0, Math.PI * 2);
          ctx.fillStyle = '#ff4a4a';
          ctx.fill();

          ctx.fillStyle = 'rgba(0,0,0,0.7)';
          ctx.fillRect(pt[0] + 7, pt[1] - 7, 62, 14);
          ctx.fillStyle = '#ff4a4a';
          ctx.font = '10px monospace';
          ctx.fillText(landmarkNames[pi] || `pt${pi}`, pt[0] + 9, pt[1] + 4);
        });
      }

      // Data readout
      allData.push({ section: `SUBJECT ${idx + 1} — FACE ANALYSIS` });
      allData.push({ key: 'CONFIDENCE', value: `${conf}%` });
      allData.push({ key: 'BBOX', value: `${Math.round(w)}x${Math.round(h)}px at (${Math.round(x1)},${Math.round(y1)})` });
      allData.push({ key: 'FACE AREA', value: `${(w * h).toFixed(0)}px²` });

      // Age/Gender/Expression from face-api.js
      if (ageGenderResult) {
        allData.push({ key: 'AGE (estimated)', value: `${Math.round(ageGenderResult.age)} years` });
        allData.push({ key: 'GENDER', value: `${ageGenderResult.gender.toUpperCase()} (${(ageGenderResult.genderProbability * 100).toFixed(0)}%)` });

        if (ageGenderResult.expressions) {
          const sorted = Object.entries(ageGenderResult.expressions)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3);
          sorted.forEach(([expr, prob]) => {
            allData.push({ key: `  ${expr.toUpperCase()}`, value: `${(prob * 100).toFixed(1)}%` });
          });
        }
      } else if (models.faceApiLoaded) {
        allData.push({ key: 'AGE/GENDER', value: 'Could not match face for analysis' });
      } else {
        allData.push({ key: 'AGE/GENDER', value: 'Loading age/gender model...' });
      }

      if (face.landmarks) {
        allData.push({ key: 'LANDMARKS', value: `${face.landmarks.length} keypoints` });

        if (face.landmarks.length >= 2) {
          const eyeR = face.landmarks[0];
          const eyeL = face.landmarks[1];
          const eyeDist = Math.sqrt(Math.pow(eyeR[0] - eyeL[0], 2) + Math.pow(eyeR[1] - eyeL[1], 2));
          const tiltDeg = Math.atan2(eyeR[1] - eyeL[1], eyeR[0] - eyeL[0]) * (180 / Math.PI);
          allData.push({ key: 'INTER-EYE DIST', value: `${eyeDist.toFixed(1)}px` });
          allData.push({ key: 'HEAD TILT', value: `${tiltDeg.toFixed(1)}°` });
        }
      }
      allData.push({ spacer: true });
    });
  }

  // Match a BlazeFace detection to the closest face-api.js result by center distance
  function matchFaceApiResult(faceApiResults, bx, by, bw, bh, idx) {
    if (idx < faceApiResults.length) {
      return faceApiResults[idx];
    }
    // Fallback: match by closest center
    const bcx = bx + bw / 2;
    const bcy = by + bh / 2;
    let best = null;
    let bestDist = Infinity;
    for (const r of faceApiResults) {
      const box = r.detection.box;
      const cx = box.x + box.width / 2;
      const cy = box.y + box.height / 2;
      const dist = Math.sqrt(Math.pow(bcx - cx, 2) + Math.pow(bcy - cy, 2));
      if (dist < bestDist) {
        bestDist = dist;
        best = r;
      }
    }
    return best;
  }

  // ── Pose: BlazeFace landmarks with connections ──
  async function runPose(source, ctx, allData) {
    if (!App.models.blazeface) return;

    let faces;
    try {
      faces = await App.models.blazeface.estimateFaces(source, false);
    } catch (err) {
      return;
    }

    allData.push({ section: 'POSE / FACIAL GEOMETRY' });
    if (!faces || faces.length === 0) {
      allData.push({ key: 'STATUS', value: 'No faces detected for pose' });
      allData.push({ spacer: true });
      return;
    }

    ctx.strokeStyle = '#4a9eff';
    ctx.lineWidth = 2;

    faces.forEach((face, idx) => {
      if (!face.landmarks || face.landmarks.length < 4) return;

      const pts = face.landmarks;
      const connections = [[0, 1], [0, 2], [1, 2], [2, 3], [0, 4], [1, 5]];

      for (const [a, b] of connections) {
        if (pts[a] && pts[b]) {
          ctx.beginPath();
          ctx.moveTo(pts[a][0], pts[a][1]);
          ctx.lineTo(pts[b][0], pts[b][1]);
          ctx.stroke();
        }
      }

      const eyeMidX = (pts[0][0] + pts[1][0]) / 2;
      const noseX = pts[2][0];
      const noseOffset = noseX - eyeMidX;
      let direction = 'FORWARD';
      if (noseOffset > 5) direction = 'LOOKING LEFT';
      if (noseOffset < -5) direction = 'LOOKING RIGHT';

      allData.push({ key: `FACE ${idx + 1} DIRECTION`, value: direction });
      allData.push({ key: 'NOSE OFFSET', value: `${noseOffset.toFixed(1)}px from center` });
    });
    allData.push({ spacer: true });
  }

  // ── COCO-SSD: object detection ──
  async function runObjects(source, ctx, allData) {
    if (!models.cocoSsd) {
      allData.push({ section: 'OBJECT DETECTION' });
      allData.push({ key: 'STATUS', value: 'Model loading... toggle off and on to retry' });
      allData.push({ spacer: true });
      return;
    }

    let preds;
    try {
      preds = await models.cocoSsd.detect(source);
    } catch (err) {
      console.error('COCO-SSD error:', err);
      return;
    }

    allData.push({ section: 'OBJECT DETECTION' });
    if (preds.length === 0) {
      allData.push({ key: 'STATUS', value: 'No objects detected' });
      allData.push({ spacer: true });
      return;
    }

    ctx.lineWidth = 2;
    ctx.font = '11px monospace';
    preds.forEach(pred => {
      const [x, y, w, h] = pred.bbox;
      const label = pred.class;
      const conf = (pred.score * 100).toFixed(0);

      ctx.strokeStyle = '#ffd94a';
      ctx.strokeRect(x, y, w, h);
      const tw = ctx.measureText(`${label} ${conf}%`).width + 10;
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x, y - 16, tw, 16);
      ctx.fillStyle = '#ffd94a';
      ctx.fillText(`${label} ${conf}%`, x + 4, y - 4);

      allData.push({ key: label.toUpperCase(), value: `${conf}% — ${Math.round(w)}x${Math.round(h)}px` });
    });
    allData.push({ spacer: true });
  }

  // ── NSFWJS: content classification ──
  async function runNSFW(source, allData) {
    if (!models.nsfw) {
      allData.push({ section: 'CONTENT CLASSIFICATION' });
      allData.push({ key: 'STATUS', value: 'Model loading... toggle off and on to retry' });
      allData.push({ spacer: true });
      return;
    }

    let inputEl = source;
    if (source instanceof HTMLVideoElement) {
      const tmp = document.createElement('canvas');
      tmp.width = source.videoWidth;
      tmp.height = source.videoHeight;
      tmp.getContext('2d').drawImage(source, 0, 0);
      inputEl = tmp;
    }

    let preds;
    try {
      preds = await models.nsfw.classify(inputEl);
    } catch (err) {
      console.error('NSFW error:', err);
      return;
    }

    allData.push({ section: 'CONTENT CLASSIFICATION' });
    preds.forEach(p => {
      allData.push({ key: p.className, value: `${(p.probability * 100).toFixed(1)}%` });
    });
    allData.push({ spacer: true });
  }

  // ── Model loading ──
  async function ensureModel(layer) {
    if (layer === 'demographics') {
      // BlazeFace from App.models — loaded at startup
      // Also load face-api.js models for age/gender/expression
      if (!models.faceApiLoaded && typeof faceapi !== 'undefined') {
        try {
          const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model';
          console.log('Loading face-api.js models...');
          await Promise.all([
            faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
            faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
            faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
          ]);
          models.faceApiLoaded = true;
          console.log('face-api.js models loaded (age/gender/expression)');
        } catch (err) {
          console.error('face-api.js model loading failed:', err);
        }
      }
      return;
    }
    if (layer === 'pose') {
      return;
    }
    if (layer === 'objects' && !models.cocoSsd) {
      try {
        console.log('Loading COCO-SSD...');
        models.cocoSsd = await cocoSsd.load();
        console.log('COCO-SSD loaded');
      } catch (err) {
        console.error('COCO-SSD failed:', err);
      }
    }
    if (layer === 'nsfw' && !models.nsfw) {
      try {
        console.log('Loading NSFWJS...');
        models.nsfw = await nsfwjs.load();
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

  // ── Render ──
  function renderData(allData) {
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
      html = '<div class="data-row">Enable a detection layer above, then upload a photo.</div>';
    }
    dataDiv.innerHTML = html;
  }

  function updateTimestamp() {
    const el = document.getElementById('timestamp');
    if (el) el.textContent = new Date().toISOString().replace('T', ' ').slice(0, 19);
  }

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
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
  }
})();
