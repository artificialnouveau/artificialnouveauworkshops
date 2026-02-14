/**
 * step2-surveillance.js — Full surveillance stack with toggleable layers
 * Layers: demographics (Human), pose (Human), objects (COCO-SSD), content/nudity (NSFWJS)
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

  // State
  let currentImg = null;
  let webcamActive = false;
  let webcamLoop = null;

  // Lazy-loaded models
  const models = {
    human: null,
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

  // Human library config
  const humanConfig = {
    backend: 'webgl',
    modelBasePath: 'https://cdn.jsdelivr.net/npm/@vladmandic/human/models/',
    face: {
      enabled: true,
      detector: { enabled: true, rotation: false },
      mesh: { enabled: false },
      iris: { enabled: false },
      description: { enabled: true },
      emotion: { enabled: true },
    },
    body: { enabled: false },
    hand: { enabled: false },
    gesture: { enabled: false },
    segmentation: { enabled: false },
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

      // Re-run on current image if not webcam
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

    if (activeLayers.demographics || activeLayers.pose) {
      await runHuman(video, octx, allData);
    }
    if (activeLayers.objects) {
      await runObjectDetection(video, octx, allData);
    }
    if (activeLayers.nsfw) {
      await runNSFW(video, allData);
    }

    renderDataReadout(allData);

    // Loop at ~4fps (models are heavy)
    webcamLoop = setTimeout(() => {
      if (webcamActive) webcamAnalysisLoop();
    }, 250);
  }

  // ── Analyze static image ──
  async function analyzeImage(img) {
    const ctx = App.drawToCanvas(canvas, img, 600);
    updateTimestamp();
    drawScanLines(ctx, canvas.width, canvas.height);

    const allData = [];

    if (activeLayers.demographics || activeLayers.pose) {
      await runHuman(canvas, ctx, allData);
    }
    if (activeLayers.objects) {
      await runObjectDetection(canvas, ctx, allData);
    }
    if (activeLayers.nsfw) {
      await runNSFW(canvas, allData);
    }

    renderDataReadout(allData);

    statsDiv.innerHTML = `
      FRAME RESOLUTION: ${canvas.width}x${canvas.height}<br>
      ACTIVE LAYERS: ${Object.entries(activeLayers).filter(([,v]) => v).map(([k]) => k.toUpperCase()).join(', ') || 'NONE'}<br>
      PROCESSING: CLIENT-SIDE (no data transmitted)<br>
      <br>
      In a real surveillance system, this data would be stored,<br>
      cross-referenced, and used without your knowledge or consent.
    `;
  }

  // ── Human library: demographics + pose ──
  async function runHuman(source, ctx, allData) {
    if (!models.human) return;

    models.human.config.body.enabled = activeLayers.pose;
    models.human.config.face.enabled = activeLayers.demographics;

    let result;
    try {
      result = await models.human.detect(source);
    } catch (err) {
      console.error('Human detection error:', err);
      return;
    }

    // Demographics
    if (activeLayers.demographics && result.face) {
      result.face.forEach((face, idx) => {
        const box = face.box || [0, 0, 0, 0];
        const [x, y, w, h] = box;

        // Bounding box
        ctx.strokeStyle = '#00ff66';
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        drawBrackets(ctx, x, y, w, h, Math.min(w, h) * 0.15);

        // Label
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(x, y - 18, w, 18);
        ctx.fillStyle = '#00ff66';
        ctx.font = '11px monospace';
        const conf = face.score ? (face.score * 100).toFixed(0) : '?';
        ctx.fillText(`SUBJ-${String(idx + 1).padStart(2, '0')}  ${conf}%`, x + 4, y - 5);

        // Data readout
        const age = face.age ? Math.round(face.age) : 'N/A';
        const gender = face.gender || 'N/A';
        const genderConf = face.genderScore ? (face.genderScore * 100).toFixed(0) + '%' : '';
        const race = face.race || 'N/A';
        const raceConf = face.raceScore ? (face.raceScore * 100).toFixed(0) + '%' : '';
        const emotion = face.emotion || 'N/A';
        const emotionConf = face.emotionScore ? (face.emotionScore * 100).toFixed(0) + '%' : '';

        allData.push({ section: `SUBJECT ${idx + 1} — DEMOGRAPHICS` });
        allData.push({ key: 'AGE (estimated)', value: `${age} years` });
        allData.push({ key: 'GENDER', value: `${gender} ${genderConf}` });
        allData.push({ key: 'RACE', value: `${race} ${raceConf}` });
        allData.push({ key: 'EMOTION', value: `${emotion} ${emotionConf}` });
        allData.push({ key: 'FACE CONFIDENCE', value: `${conf}%` });
        allData.push({ key: 'BBOX', value: `${Math.round(w)}x${Math.round(h)}px at (${Math.round(x)},${Math.round(y)})` });
        allData.push({ spacer: true });
      });

      if (!result.face || result.face.length === 0) {
        allData.push({ section: 'DEMOGRAPHICS' });
        allData.push({ key: 'STATUS', value: 'No faces detected' });
        allData.push({ spacer: true });
      }
    }

    // Pose
    if (activeLayers.pose && result.body) {
      result.body.forEach((body, idx) => {
        if (!body.keypoints) return;

        ctx.fillStyle = '#ff4a4a';
        ctx.strokeStyle = '#ff4a4a';
        ctx.lineWidth = 2;

        for (const kp of body.keypoints) {
          if (kp.score > 0.3) {
            ctx.beginPath();
            ctx.arc(kp.position[0], kp.position[1], 4, 0, Math.PI * 2);
            ctx.fill();
          }
        }

        // Skeleton connections
        const connections = [
          [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
          [5, 11], [6, 12], [11, 12],
          [11, 13], [13, 15], [12, 14], [14, 16],
        ];
        for (const [a, b] of connections) {
          const kpA = body.keypoints[a];
          const kpB = body.keypoints[b];
          if (kpA && kpB && kpA.score > 0.3 && kpB.score > 0.3) {
            ctx.beginPath();
            ctx.moveTo(kpA.position[0], kpA.position[1]);
            ctx.lineTo(kpB.position[0], kpB.position[1]);
            ctx.stroke();
          }
        }

        const visible = body.keypoints.filter(k => k.score > 0.3);
        allData.push({ section: `BODY ${idx + 1} — POSE` });
        allData.push({ key: 'KEYPOINTS', value: `${visible.length} / ${body.keypoints.length}` });
        allData.push({ spacer: true });
      });

      if (!result.body || result.body.length === 0) {
        allData.push({ section: 'POSE TRACKING' });
        allData.push({ key: 'STATUS', value: 'No body detected' });
        allData.push({ spacer: true });
      }
    }
  }

  // ── COCO-SSD: object detection ──
  async function runObjectDetection(source, ctx, allData) {
    if (!models.cocoSsd) return;

    let predictions;
    try {
      predictions = await models.cocoSsd.detect(source);
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
    predictions.forEach(pred => {
      const [x, y, w, h] = pred.bbox;
      const label = pred.class;
      const conf = (pred.score * 100).toFixed(0);

      ctx.strokeStyle = '#ffd94a';
      ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      const textW = ctx.measureText(`${label} ${conf}%`).width + 10;
      ctx.fillRect(x, y - 16, textW, 16);
      ctx.fillStyle = '#ffd94a';
      ctx.font = '11px monospace';
      ctx.fillText(`${label} ${conf}%`, x + 4, y - 4);

      allData.push({ key: label.toUpperCase(), value: `${conf}%` });
    });
    allData.push({ spacer: true });
  }

  // ── NSFWJS: content classification ──
  async function runNSFW(source, allData) {
    if (!models.nsfw) return;

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
      predictions = await models.nsfw.classify(inputEl);
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
    if ((layer === 'demographics' || layer === 'pose') && !models.human) {
      try {
        models.human = new Human(humanConfig);
        await models.human.load();
        console.log('Human library loaded');
      } catch (err) {
        console.error('Human library failed:', err);
        models.human = null;
      }
    }
    if (layer === 'objects' && !models.cocoSsd) {
      try {
        models.cocoSsd = await cocoSsd.load();
        console.log('COCO-SSD loaded');
      } catch (err) {
        console.error('COCO-SSD failed:', err);
      }
    }
    if (layer === 'nsfw' && !models.nsfw) {
      try {
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
