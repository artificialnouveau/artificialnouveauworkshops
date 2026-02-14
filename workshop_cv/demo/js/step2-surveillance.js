/**
 * step2-surveillance.js — Face detection with surveillance-style aesthetic
 */

(function () {
  const fileInput = document.getElementById('file-input-2');
  const results = document.getElementById('step2-results');
  const uploadArea = document.getElementById('upload-area-2');

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    uploadArea.classList.add('has-file');
    const img = await App.loadImage(file);
    results.classList.remove('hidden');

    // Update timestamp
    const now = new Date();
    document.getElementById('timestamp').textContent =
      now.toISOString().replace('T', ' ').slice(0, 19);

    await drawSurveillance(img);
  });

  async function drawSurveillance(img) {
    const canvas = document.getElementById('canvas-surveillance');
    const ctx = App.drawToCanvas(canvas, img, 600);
    const scaleX = canvas.width / img.width;
    const scaleY = canvas.height / img.height;

    const dataDiv = document.getElementById('surveillance-data');
    const statsDiv = document.getElementById('face-stats');

    // Detect faces with BlazeFace
    if (!App.models.blazeface) {
      dataDiv.innerHTML = '<div class="data-row">ERR: FACE DETECTION MODEL UNAVAILABLE</div>';
      return;
    }

    const predictions = await App.models.blazeface.estimateFaces(canvas, false);

    if (predictions.length === 0) {
      dataDiv.innerHTML = '<div class="data-row">NO FACES DETECTED</div>';
      statsDiv.innerHTML = 'SUBJECTS: 0';
      return;
    }

    // Draw bounding boxes and landmarks — surveillance style
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#00ff66';
    ctx.fillStyle = '#00ff66';
    ctx.font = '12px monospace';

    let dataHTML = '';

    predictions.forEach((face, idx) => {
      const x1 = face.topLeft[0];
      const y1 = face.topLeft[1];
      const x2 = face.bottomRight[0];
      const y2 = face.bottomRight[1];
      const w = x2 - x1;
      const h = y2 - y1;

      // Bounding box
      ctx.strokeRect(x1, y1, w, h);

      // Corner brackets (surveillance aesthetic)
      const bracketLen = Math.min(w, h) * 0.15;
      drawBrackets(ctx, x1, y1, w, h, bracketLen);

      // Label
      const confidence = (face.probability[0] * 100).toFixed(1);
      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(x1, y1 - 18, 140, 18);
      ctx.fillStyle = '#00ff66';
      ctx.fillText(`SUBJ-${String(idx + 1).padStart(2, '0')}  ${confidence}%`, x1 + 4, y1 - 5);

      // Landmarks (eyes, nose, mouth, ears)
      if (face.landmarks) {
        face.landmarks.forEach(point => {
          ctx.beginPath();
          ctx.arc(point[0], point[1], 3, 0, Math.PI * 2);
          ctx.fillStyle = '#ff4a4a';
          ctx.fill();
        });
      }

      // Data extraction readout
      const faceWidth = Math.round(w / scaleX);
      const faceHeight = Math.round(h / scaleY);
      const centerX = Math.round((x1 + w / 2) / scaleX);
      const centerY = Math.round((y1 + h / 2) / scaleY);

      dataHTML += `
        <div class="data-row">SUBJECT ${idx + 1} ━━━━━━━━━━━━━━━━</div>
        <div class="data-row">CONFIDENCE: ${confidence}%</div>
        <div class="data-row">BBOX: ${faceWidth}x${faceHeight}px</div>
        <div class="data-row">POSITION: (${centerX}, ${centerY})</div>
        <div class="data-row">LANDMARKS: ${face.landmarks ? face.landmarks.length : 0} points extracted</div>
        <div class="data-row">FACE RATIO: ${(faceWidth / faceHeight).toFixed(2)}</div>
        <div class="data-row">&nbsp;</div>
      `;
    });

    dataDiv.innerHTML = dataHTML;

    // Scan line effect
    drawScanLines(ctx, canvas.width, canvas.height);

    statsDiv.innerHTML = `
      SUBJECTS DETECTED: ${predictions.length}<br>
      FRAME RESOLUTION: ${canvas.width}x${canvas.height}<br>
      PROCESSING: CLIENT-SIDE (no data transmitted)<br>
      <br>
      In a real surveillance system, this data would be stored,<br>
      cross-referenced, and used without your knowledge or consent.
    `;
  }

  function drawBrackets(ctx, x, y, w, h, len) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#00ff66';

    // Top-left
    ctx.beginPath();
    ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y);
    ctx.stroke();
    // Top-right
    ctx.beginPath();
    ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
    ctx.stroke();
    // Bottom-left
    ctx.beginPath();
    ctx.moveTo(x, y + h - len); ctx.lineTo(x, y + h); ctx.lineTo(x + len, y + h);
    ctx.stroke();
    // Bottom-right
    ctx.beginPath();
    ctx.moveTo(x + w - len, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - len);
    ctx.stroke();
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
