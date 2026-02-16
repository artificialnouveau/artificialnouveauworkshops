/**
 * step3-surveillance.js — Full surveillance stack with toggleable layers
 * Uses: BlazeFace (face boxes), face-api.js bundled (age/gender/expression),
 *       COCO-SSD (objects), MobileNet (scene classification)
 * Sources: image upload or webcam
 */

(function () {
  const fileInput = document.getElementById('file-input-3');
  const uploadArea = document.getElementById('upload-area-3');
  const webcamBtn = document.getElementById('webcam-btn');
  const results = document.getElementById('step3-results');
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
  const loadingBar = document.getElementById('model-loading-bar');
  const loadingFill = document.getElementById('loading-fill');
  const loadingPercent = document.getElementById('loading-percent');
  const loadingSteps = document.getElementById('loading-steps');

  const NUDENET_LABELS = [
    'exposed anus', 'exposed armpits', 'belly', 'exposed belly',
    'buttocks', 'exposed buttocks', 'female face', 'male face',
    'feet', 'exposed feet', 'breast', 'exposed breast',
    'vagina', 'exposed vagina', 'male breast', 'exposed male breast',
  ];
  // Indices that should be censored (black boxes)
  const NSFW_INDICES = new Set([0, 3, 5, 11, 13, 15]); // exposed: anus, belly, buttocks, breast, vagina, male breast

  function showLoading(stepText, percent) {
    loadingBar.classList.add('visible');
    loadingFill.style.width = percent + '%';
    loadingPercent.textContent = percent + '%';
    loadingSteps.textContent = stepText;
  }

  function hideLoading() {
    loadingFill.style.width = '100%';
    loadingPercent.textContent = '100%';
    loadingSteps.textContent = 'All models ready';
    setTimeout(() => { loadingBar.classList.remove('visible'); }, 600);
  }

  const activeLayers = {
    demographics: true,
    pose: false,
    objects: false,
    nsfw: false,
    scene: false,
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
        showLoading(`Loading ${layer} model...`, 10);
        await ensureModel(layer);
        chip.classList.remove('loading');
        hideLoading();
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
    if (activeLayers.nsfw) await runNSFW(video, octx, allData);
    if (activeLayers.scene) await runScene(video, allData);
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
    if (activeLayers.nsfw) await runNSFW(canvas, ctx, allData);
    if (activeLayers.scene) await runScene(canvas, allData);
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

    // Run face-api.js for age/gender/expression if loaded
    let faceApiResults = null;
    if (models.faceApiLoaded && typeof faceapi !== 'undefined') {
      try {
        let inputEl = source;
        if (source instanceof HTMLVideoElement) {
          const tmp = document.createElement('canvas');
          tmp.width = source.videoWidth;
          tmp.height = source.videoHeight;
          tmp.getContext('2d').drawImage(source, 0, 0);
          inputEl = tmp;
        }
        faceApiResults = await faceapi
          .detectAllFaces(inputEl, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.3 }))
          .withAgeAndGender()
          .withFaceExpressions();
        console.log('face-api.js results:', faceApiResults?.length || 0, 'faces');
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

      // Match with face-api.js result
      let ageGenderResult = null;
      if (faceApiResults && faceApiResults.length > 0) {
        ageGenderResult = matchFaceApiResult(faceApiResults, x1, y1, w, h, idx);
      }

      // Green bounding box with brackets
      ctx.strokeStyle = '#00ff66';
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, w, h);
      drawBrackets(ctx, x1, y1, w, h, Math.min(w, h) * 0.15);

      // Label
      let labelText = `SUBJ-${String(idx + 1).padStart(2, '0')}  CONF:${conf}%`;
      if (ageGenderResult) {
        const age = Math.round(ageGenderResult.age);
        const gender = ageGenderResult.gender.toUpperCase();
        const genderConf = (ageGenderResult.genderProbability * 100).toFixed(0);
        labelText = `SUBJ-${String(idx + 1).padStart(2, '0')}  AGE:${age}  ${gender}(${genderConf}%)`;
      }

      ctx.font = '11px monospace';
      const labelW = Math.max(ctx.measureText(labelText).width + 10, w);
      ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
      ctx.fillRect(x1, y1 - 18, labelW, 18);
      ctx.fillStyle = '#00ff66';
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

      if (ageGenderResult) {
        allData.push({ key: 'AGE (estimated)', value: `${Math.round(ageGenderResult.age)} years` });
        allData.push({ key: 'GENDER', value: `${ageGenderResult.gender.toUpperCase()} (${(ageGenderResult.genderProbability * 100).toFixed(0)}%)` });

        if (ageGenderResult.expressions) {
          const sorted = Object.entries(ageGenderResult.expressions)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3);
          allData.push({ key: 'EXPRESSION', value: '' });
          sorted.forEach(([expr, prob]) => {
            allData.push({ key: `  ${expr.toUpperCase()}`, value: `${(prob * 100).toFixed(1)}%` });
          });
        }
      } else if (!models.faceApiLoaded) {
        allData.push({ key: 'AGE/GENDER', value: 'Model loading... toggle Face Analysis off and on to retry' });
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

      // BMI estimation from face width-to-height ratio
      const faceWHR = w / h;
      const estimatedBMI = 12.0 + (faceWHR * 22.0);
      let bmiCategory = 'Normal';
      if (estimatedBMI < 18.5) bmiCategory = 'Underweight';
      else if (estimatedBMI < 25) bmiCategory = 'Normal';
      else if (estimatedBMI < 30) bmiCategory = 'Overweight';
      else bmiCategory = 'Obese';
      allData.push({ key: 'FACE WIDTH/HEIGHT', value: `${faceWHR.toFixed(3)}` });
      allData.push({ key: 'BMI ESTIMATE', value: `${estimatedBMI.toFixed(1)} (${bmiCategory})` });

      allData.push({ spacer: true });
    });
  }

  function matchFaceApiResult(faceApiResults, bx, by, bw, bh, idx) {
    if (idx < faceApiResults.length) {
      return faceApiResults[idx];
    }
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

  // ── Pose ──
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

  // ── COCO-SSD ──
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

  // ── NSFW object detection via NudeNet (self-hosted TFJS graph model) ──
  async function runNSFW(source, ctx, allData) {
    if (!models.nsfw) {
      allData.push({ section: 'CONTENT DETECTION' });
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

    const sourceW = inputEl.width || inputEl.videoWidth;
    const sourceH = inputEl.height || inputEl.videoHeight;

    try {
      // Preprocess: resize to 320x320, cast to float32 (keep 0-255 range, do NOT normalize)
      const inputTensor = tf.tidy(() => {
        const img = tf.browser.fromPixels(inputEl);
        const resized = tf.image.resizeBilinear(img, [320, 320]);
        return resized.toFloat().expandDims(0);
      });

      // Run model — outputs: boxes [1,300,4], scores [1,300], classes [1,300]
      const output = await models.nsfw.executeAsync(inputTensor, ['output1', 'output2', 'output3']);
      const boxes = await output[0].array();   // [1, 300, 4] — [y0, x0, y1, x1] in 320x320 pixel coords
      const scores = await output[1].data();   // [300]
      const classes = await output[2].data();  // [300] int

      inputTensor.dispose();
      output.forEach(t => t.dispose());

      const minScore = 0.3;
      const detections = [];
      for (let i = 0; i < scores.length; i++) {
        if (scores[i] < minScore) continue;
        const classIdx = classes[i];
        const label = NUDENET_LABELS[classIdx] || `class-${classIdx}`;
        // Box format: [y0, x0, y1, x1] in 320x320 pixel space
        const [y0, x0, y1, x1] = boxes[0][i];
        // Scale from 320x320 to source image dimensions
        detections.push({
          label, classIdx,
          score: scores[i],
          x: (x0 / 320) * sourceW, y: (y0 / 320) * sourceH,
          w: ((x1 - x0) / 320) * sourceW, h: ((y1 - y0) / 320) * sourceH,
        });
      }

      console.log('NudeNet raw detections:', detections.length, detections.slice(0, 5));

      // NMS: remove overlapping boxes for same class
      const filtered = simpleNMS(detections, 0.4);
      console.log('NudeNet after NMS:', filtered.length, filtered);

      allData.push({ section: 'CONTENT DETECTION (NudeNet)' });
      if (filtered.length === 0) {
        allData.push({ key: 'STATUS', value: 'No exposed body parts detected' });
      }

      // Scale factor if canvas is smaller than source
      const scaleX = (ctx.canvas.width || sourceW) / sourceW;
      const scaleY = (ctx.canvas.height || sourceH) / sourceH;

      for (const det of filtered) {
        const dx = det.x * scaleX;
        const dy = det.y * scaleY;
        const dw = det.w * scaleX;
        const dh = det.h * scaleY;
        const conf = (det.score * 100).toFixed(0);

        if (NSFW_INDICES.has(det.classIdx)) {
          // Black box over exposed region
          ctx.fillStyle = '#000000';
          ctx.fillRect(dx, dy, dw, dh);

          // Label on black box
          ctx.font = '11px monospace';
          ctx.fillStyle = '#ff4a4a';
          const labelText = `CENSORED: ${det.label} (${conf}%)`;
          ctx.fillText(labelText, dx + 4, dy + 14);
        } else {
          // Non-exposed: just outline
          ctx.strokeStyle = 'rgba(255, 217, 74, 0.6)';
          ctx.lineWidth = 1;
          ctx.strokeRect(dx, dy, dw, dh);
          ctx.font = '10px monospace';
          ctx.fillStyle = '#ffd94a';
          ctx.fillText(`${det.label} ${conf}%`, dx + 2, dy - 3);
        }

        allData.push({ key: det.label.toUpperCase(), value: `${conf}% — ${Math.round(det.w)}x${Math.round(det.h)}px` });
      }
      allData.push({ spacer: true });
    } catch (err) {
      console.error('NudeNet detection error:', err);
      allData.push({ section: 'CONTENT DETECTION' });
      allData.push({ key: 'ERROR', value: err.message });
      allData.push({ spacer: true });
    }
  }

  // Simple NMS: suppress overlapping boxes of same class
  function simpleNMS(detections, iouThresh) {
    detections.sort((a, b) => b.score - a.score);
    const keep = [];
    const used = new Set();
    for (let i = 0; i < detections.length; i++) {
      if (used.has(i)) continue;
      keep.push(detections[i]);
      for (let j = i + 1; j < detections.length; j++) {
        if (used.has(j)) continue;
        if (detections[i].classIdx === detections[j].classIdx && iou(detections[i], detections[j]) > iouThresh) {
          used.add(j);
        }
      }
    }
    return keep;
  }

  function iou(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.w, b.x + b.w);
    const y2 = Math.min(a.y + a.h, b.y + b.h);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = a.w * a.h + b.w * b.h - inter;
    return union > 0 ? inter / union : 0;
  }

  // ── Scene classification via MobileNet ──
  async function runScene(source, allData) {
    if (!App.models.mobilenet) {
      allData.push({ section: 'SCENE CLASSIFICATION' });
      allData.push({ key: 'STATUS', value: 'MobileNet not loaded — reload the page' });
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
      preds = await App.models.mobilenet.classify(inputEl, 10);
    } catch (err) {
      console.error('Scene classification error:', err);
      return;
    }

    allData.push({ section: 'SCENE CLASSIFICATION' });
    preds.forEach(p => {
      const label = p.className.split(',')[0];
      allData.push({ key: label.toUpperCase(), value: `${(p.probability * 100).toFixed(1)}%` });
    });
    allData.push({ spacer: true });
  }

  // ── Model loading ──
  async function ensureModel(layer) {
    if (layer === 'demographics') {
      // BlazeFace loaded at startup via App.models
      // Also load face-api.js models for age/gender/expression
      if (!models.faceApiLoaded && typeof faceapi !== 'undefined') {
        try {
          const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model';
          console.log('Loading face-api.js models from:', MODEL_URL);

          showLoading('Loading face detector...', 15);
          await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);

          showLoading('Loading age & gender model...', 40);
          await faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL);

          showLoading('Loading expression model...', 65);
          await faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL);

          models.faceApiLoaded = true;
          showLoading('Face analysis models ready', 80);
          console.log('face-api.js models loaded successfully (age/gender/expression)');
        } catch (err) {
          console.error('face-api.js model loading failed:', err);
          showLoading('Face analysis model failed — check console', 0);
        }
      }
      return;
    }
    if (layer === 'pose') {
      return;
    }
    if (layer === 'objects' && !models.cocoSsd) {
      try {
        showLoading('Loading object detection model...', 50);
        console.log('Loading COCO-SSD...');
        models.cocoSsd = await cocoSsd.load();
        console.log('COCO-SSD loaded');
      } catch (err) {
        console.error('COCO-SSD failed:', err);
      }
    }
    if (layer === 'nsfw' && !models.nsfw) {
      try {
        showLoading('Loading NudeNet detection model...', 30);
        console.log('Loading NudeNet (self-hosted graph model)...');
        models.nsfw = await tf.loadGraphModel('models/nudenet/model.json');
        showLoading('Warming up NudeNet...', 70);
        const dummy = tf.zeros([1, 320, 320, 3], 'float32');
        const warmup = await models.nsfw.executeAsync(dummy, ['output1', 'output2', 'output3']);
        warmup.forEach(t => t.dispose());
        dummy.dispose();
        console.log('NudeNet loaded successfully');
      } catch (err) {
        console.error('NudeNet failed:', err);
      }
    }
    if (layer === 'scene') {
      // Uses App.models.mobilenet — already loaded at startup
      return;
    }
  }

  async function ensureActiveModels() {
    const activelist = Object.entries(activeLayers).filter(([,v]) => v).map(([k]) => k);
    if (activelist.length === 0) return;

    // Check if any models actually need loading
    const needsLoading = activelist.some(layer => {
      if (layer === 'demographics') return !models.faceApiLoaded && typeof faceapi !== 'undefined';
      if (layer === 'objects') return !models.cocoSsd;
      if (layer === 'nsfw') return !models.nsfw;
      return false;
    });

    if (needsLoading) {
      showLoading('Preparing models...', 5);
    }

    let step = 0;
    const total = activelist.length;
    for (const layer of activelist) {
      step++;
      const pct = Math.round((step / total) * 90) + 5;
      showLoading(`Loading ${layer}...`, pct);
      await ensureModel(layer);
    }

    if (needsLoading) {
      hideLoading();
    }
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
