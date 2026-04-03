/**
 * step6-clone.js — Face similarity comparison using face-api.js
 * Computes 128D face descriptors and measures euclidean distance
 */

(function () {
  const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.15/model';

  const fileInputA = document.getElementById('file-input-6a');
  const fileInputB = document.getElementById('file-input-6b');
  const uploadAreaA = document.getElementById('upload-area-6a');
  const uploadAreaB = document.getElementById('upload-area-6b');
  const canvasA = document.getElementById('canvas-clone-a');
  const canvasB = document.getElementById('canvas-clone-b');
  const compareBtn = document.getElementById('clone-compare-btn');
  const results = document.getElementById('step6-results');
  const verdictDiv = document.getElementById('clone-verdict');
  const similarityFill = document.getElementById('clone-similarity-fill');
  const similarityValue = document.getElementById('clone-similarity-value');
  const distanceFill = document.getElementById('clone-distance-fill');
  const distanceValue = document.getElementById('clone-distance-value');

  const loadingBar = document.getElementById('clone-loading-bar');
  const loadingFill = document.getElementById('clone-loading-fill');
  const loadingPercent = document.getElementById('clone-loading-percent');
  const loadingSteps = document.getElementById('clone-loading-steps');

  let imgA = null;
  let imgB = null;
  let modelsLoaded = false;

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

  function updateCompareButton() {
    if (imgA && imgB) {
      compareBtn.disabled = false;
      compareBtn.style.opacity = '1';
    } else {
      compareBtn.disabled = true;
      compareBtn.style.opacity = '0.4';
    }
  }

  // Fixed display size so both canvases always match
  const DISPLAY_W = 380;
  const DISPLAY_H = 380;

  function drawImageWithBox(canvas, img, detection) {
    canvas.width = DISPLAY_W;
    canvas.height = DISPLAY_H;
    const ctx = canvas.getContext('2d');

    // Fill background
    ctx.fillStyle = '#141414';
    ctx.fillRect(0, 0, DISPLAY_W, DISPLAY_H);

    // Fit image within the square, centered
    const imgAspect = img.width / img.height;
    let drawW, drawH, offsetX, offsetY;
    if (imgAspect > 1) {
      drawW = DISPLAY_W;
      drawH = DISPLAY_W / imgAspect;
      offsetX = 0;
      offsetY = (DISPLAY_H - drawH) / 2;
    } else {
      drawH = DISPLAY_H;
      drawW = DISPLAY_H * imgAspect;
      offsetX = (DISPLAY_W - drawW) / 2;
      offsetY = 0;
    }

    ctx.drawImage(img, offsetX, offsetY, drawW, drawH);

    // Scale factor from original image coords to canvas coords
    const scaleX = drawW / img.width;
    const scaleY = drawH / img.height;

    if (detection) {
      // Draw bounding box
      const box = detection.detection.box;
      ctx.strokeStyle = '#4a9eff';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        box.x * scaleX + offsetX, box.y * scaleY + offsetY,
        box.width * scaleX, box.height * scaleY
      );
      ctx.fillStyle = 'rgba(74, 158, 255, 0.08)';
      ctx.fillRect(
        box.x * scaleX + offsetX, box.y * scaleY + offsetY,
        box.width * scaleX, box.height * scaleY
      );

      // Draw 68 face landmarks
      if (detection.landmarks) {
        const points = detection.landmarks.positions;
        // Draw landmark points
        ctx.fillStyle = '#00ff66';
        for (const pt of points) {
          ctx.beginPath();
          ctx.arc(pt.x * scaleX + offsetX, pt.y * scaleY + offsetY, 2, 0, Math.PI * 2);
          ctx.fill();
        }
        // Connect landmark groups with lines
        const groups = [
          [0, 16],   // jawline
          [17, 21],  // left eyebrow
          [22, 26],  // right eyebrow
          [27, 30],  // nose bridge
          [31, 35],  // nose bottom
          [36, 41],  // left eye (closed loop)
          [42, 47],  // right eye (closed loop)
          [48, 59],  // outer lip (closed loop)
          [60, 67],  // inner lip (closed loop)
        ];
        ctx.strokeStyle = 'rgba(0, 255, 102, 0.4)';
        ctx.lineWidth = 1;
        for (const [start, end] of groups) {
          ctx.beginPath();
          for (let i = start; i <= end; i++) {
            const x = points[i].x * scaleX + offsetX;
            const y = points[i].y * scaleY + offsetY;
            if (i === start) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          // Close loops for eyes and lips
          if (start === 36 || start === 42 || start === 48 || start === 60) {
            const x = points[start].x * scaleX + offsetX;
            const y = points[start].y * scaleY + offsetY;
            ctx.lineTo(x, y);
          }
          ctx.stroke();
        }
      }
    }
  }

  async function loadRecognitionModels() {
    if (modelsLoaded) return;

    showLoading('Loading SSD MobileNet...', 10);
    await faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL);

    showLoading('Loading face landmark model...', 40);
    await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);

    showLoading('Loading face recognition model (~6MB)...', 60);
    await faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL);

    showLoading('All models loaded', 100);
    modelsLoaded = true;
    hideLoading();
  }

  async function detectFace(img) {
    // Create a temp canvas at original size for detection
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = img.width;
    tempCanvas.height = img.height;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(img, 0, 0);

    const detection = await faceapi
      .detectSingleFace(tempCanvas)
      .withFaceLandmarks()
      .withFaceDescriptor();

    return detection;
  }

  function drawDescriptor(canvasId, descriptor) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const w = canvas.clientWidth;
    const h = 40;
    canvas.width = w;
    canvas.height = h;

    ctx.clearRect(0, 0, w, h);
    const barW = w / 128;

    for (let i = 0; i < 128; i++) {
      const val = descriptor[i];
      // Map roughly from [-0.5, 0.5] to color
      const normalized = Math.max(0, Math.min(1, (val + 0.4) / 0.8));
      const r = Math.round(normalized * 255);
      const b = Math.round((1 - normalized) * 255);
      const g = Math.round(Math.min(normalized, 1 - normalized) * 2 * 180);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(i * barW, 0, barW + 0.5, h);
    }
  }

  // Handle uploads
  fileInputA.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    uploadAreaA.classList.add('has-file');
    imgA = await App.loadImage(file);

    await loadRecognitionModels();
    const detection = await detectFace(imgA);
    drawImageWithBox(canvasA, imgA, detection);

    if (!detection) {
      canvasA.getContext('2d').fillStyle = 'rgba(255,74,74,0.8)';
      canvasA.getContext('2d').font = '14px monospace';
      canvasA.getContext('2d').fillText('No face detected', 10, 24);
    }

    updateCompareButton();
    results.classList.add('hidden');
  });

  fileInputB.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    uploadAreaB.classList.add('has-file');
    imgB = await App.loadImage(file);

    await loadRecognitionModels();
    const detection = await detectFace(imgB);
    drawImageWithBox(canvasB, imgB, detection);

    if (!detection) {
      canvasB.getContext('2d').fillStyle = 'rgba(255,74,74,0.8)';
      canvasB.getContext('2d').font = '14px monospace';
      canvasB.getContext('2d').fillText('No face detected', 10, 24);
    }

    updateCompareButton();
    results.classList.add('hidden');
  });

  // Compare button
  compareBtn.addEventListener('click', async () => {
    if (!imgA || !imgB) return;

    compareBtn.disabled = true;
    compareBtn.textContent = 'Analyzing...';

    await loadRecognitionModels();

    const [detA, detB] = await Promise.all([
      detectFace(imgA),
      detectFace(imgB)
    ]);

    // Redraw with boxes
    drawImageWithBox(canvasA, imgA, detA);
    drawImageWithBox(canvasB, imgB, detB);

    if (!detA || !detB) {
      verdictDiv.innerHTML = `
        <h3 style="color:var(--red)">Face Not Detected</h3>
        <p style="color:var(--text-dim)">${!detA && !detB ? 'No face found in either photo.' : !detA ? 'No face found in Photo A.' : 'No face found in Photo B.'} Try a clearer photo with a visible face.</p>
      `;
      verdictDiv.style.borderLeftColor = 'var(--red)';
      results.classList.remove('hidden');
      compareBtn.disabled = false;
      compareBtn.textContent = 'Compare Faces';
      return;
    }

    // Compute euclidean distance
    const distance = faceapi.euclideanDistance(detA.descriptor, detB.descriptor);
    const similarity = Math.max(0, Math.min(100, 100 * (1 - distance / 1.2)));

    // Determine verdict
    let verdictText, verdictColor, verdictLabel;
    if (distance < 0.4) {
      verdictText = 'The machine thinks these are the <strong>same person</strong>.';
      verdictColor = 'var(--green)';
      verdictLabel = 'MATCH';
    } else if (distance < 0.6) {
      verdictText = 'The machine is <strong>uncertain</strong> — these faces are similar but not a clear match.';
      verdictColor = 'var(--yellow)';
      verdictLabel = 'UNCERTAIN';
    } else {
      verdictText = 'The machine thinks these are <strong>different people</strong>.';
      verdictColor = 'var(--red)';
      verdictLabel = 'DIFFERENT';
    }

    verdictDiv.style.borderLeftColor = verdictColor;
    verdictDiv.innerHTML = `
      <h3 style="color:${verdictColor}; font-size:1.1rem; margin-bottom:8px">${verdictLabel}</h3>
      <p style="color:var(--text-dim)">${verdictText}</p>
      <p style="color:var(--text-dim); font-size:0.8rem; margin-top:8px">Euclidean distance: <strong style="color:var(--text)">${distance.toFixed(4)}</strong> — Threshold: &lt;0.4 match, 0.4–0.6 uncertain, &gt;0.6 different</p>
    `;

    // Similarity bar
    similarityFill.style.width = similarity.toFixed(1) + '%';
    similarityFill.style.background = verdictColor;
    similarityValue.textContent = similarity.toFixed(1) + '%';

    // Distance bar (normalize to 0-1.2 range for display)
    const distPct = Math.min(100, (distance / 1.2) * 100);
    distanceFill.style.width = distPct.toFixed(1) + '%';
    distanceFill.style.background = verdictColor;
    distanceValue.textContent = distance.toFixed(4);

    // Draw descriptor visualizations
    drawDescriptor('canvas-descriptor-a', detA.descriptor);
    drawDescriptor('canvas-descriptor-b', detB.descriptor);

    results.classList.remove('hidden');
    compareBtn.disabled = false;
    compareBtn.textContent = 'Compare Faces';
  });
})();
