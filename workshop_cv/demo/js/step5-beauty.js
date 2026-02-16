/**
 * step5-beauty.js — Face mesh landmarks + beauty filter deconstruction
 */

(function () {
  const fileInput = document.getElementById('file-input-5');
  const results = document.getElementById('step5-results');
  const uploadArea = document.getElementById('upload-area-5');

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    uploadArea.classList.add('has-file');
    const img = await App.loadImage(file);
    results.classList.remove('hidden');

    await processBeautyFilter(img);
  });

  async function processBeautyFilter(img) {
    const landmarkCanvas = document.getElementById('canvas-landmarks');
    const filteredCanvas = document.getElementById('canvas-filtered');
    const detailsDiv = document.getElementById('filter-details');

    const maxW = 360;
    const scale = Math.min(1, maxW / img.width);
    const w = Math.floor(img.width * scale);
    const h = Math.floor(img.height * scale);

    // Draw original with landmarks
    landmarkCanvas.width = w;
    landmarkCanvas.height = h;
    const lCtx = landmarkCanvas.getContext('2d');
    lCtx.drawImage(img, 0, 0, w, h);

    // Draw filtered version
    filteredCanvas.width = w;
    filteredCanvas.height = h;
    const fCtx = filteredCanvas.getContext('2d');
    fCtx.drawImage(img, 0, 0, w, h);

    if (!App.models.facemesh) {
      detailsDiv.innerHTML = 'Face mesh model unavailable. Showing basic filter only.';
      applyBasicFilter(fCtx, w, h);
      return;
    }

    // Detect face landmarks (new API: estimateFaces takes the element directly)
    let faces;
    try {
      faces = await App.models.facemesh.estimateFaces(landmarkCanvas);
    } catch (err) {
      console.error('Face mesh error:', err);
      detailsDiv.innerHTML = 'Face mesh detection failed. Showing basic filter.';
      applyBasicFilter(fCtx, w, h);
      return;
    }

    if (!faces || faces.length === 0) {
      detailsDiv.innerHTML = 'No face detected. Try a clearer selfie with good lighting.';
      applyBasicFilter(fCtx, w, h);
      return;
    }

    // New API returns keypoints as [{x, y, z, name?}, ...] — convert to [x,y,z] arrays
    const rawKeypoints = faces[0].keypoints || faces[0].scaledMesh || faces[0].mesh;
    const keypoints = rawKeypoints.map(kp =>
      Array.isArray(kp) ? kp : [kp.x, kp.y, kp.z || 0]
    );

    // Draw landmarks on original
    drawLandmarks(lCtx, keypoints, w, h);

    // Apply beauty filter transformations
    const changes = applyBeautyFilter(fCtx, keypoints, w, h, img);

    // Show what was changed
    detailsDiv.innerHTML = changes.map(c =>
      `<div class="change-item">
        <span class="change-label">${c.label}</span>
        <span class="change-value">${c.value}</span>
      </div>`
    ).join('');

    // Run symmetry & proportionality analysis
    const symmetryScore = analyzeSymmetry(keypoints);
    const proportionScore = analyzeProportions(keypoints);

    // Compute and display composite attractiveness score
    computeAttractivenessScore(keypoints, symmetryScore, proportionScore);
  }

  function drawLandmarks(ctx, keypoints, w, h, color = 'rgba(74, 158, 255, 0.5)') {
    ctx.fillStyle = color;
    for (const point of keypoints) {
      const x = point[0];
      const y = point[1];
      if (x >= 0 && x <= w && y >= 0 && y <= h) {
        ctx.beginPath();
        ctx.arc(x, y, 1, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Draw key contours: jaw, lips, eyes, nose
    const contours = {
      jaw: [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109],
      leftEye: [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
      rightEye: [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
      lips: [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
      nose: [168, 6, 197, 195, 5, 4, 1, 19, 94, 2],
    };

    ctx.lineWidth = 0.8;
    for (const [name, indices] of Object.entries(contours)) {
      ctx.strokeStyle = name === 'lips' ? 'rgba(255, 74, 74, 0.4)' :
                        name.includes('Eye') ? 'rgba(74, 255, 158, 0.4)' :
                        'rgba(74, 158, 255, 0.3)';
      ctx.beginPath();
      for (let i = 0; i < indices.length; i++) {
        const pt = keypoints[indices[i]];
        if (!pt) continue;
        if (i === 0) ctx.moveTo(pt[0], pt[1]);
        else ctx.lineTo(pt[0], pt[1]);
      }
      ctx.closePath();
      ctx.stroke();
    }
  }

  function applyBeautyFilter(ctx, keypoints, w, h, img) {
    const changes = [];

    // 1. Wrinkle detection — highlight detected wrinkles/creases on the filtered canvas
    const faceRegion = getFaceBounds(keypoints);
    if (faceRegion) {
      const { x, y, fw, fh } = faceRegion;

      // Create a blurred reference to compare against
      const faceData = ctx.getImageData(x, y, fw, fh);
      const blurRef = ctx.getImageData(x, y, fw, fh);
      boxBlur(blurRef, Math.max(2, Math.floor(fw / 50)));

      // Build a wrinkle mask
      const wrinkleMask = new Uint8Array(fw * fh);
      let wrinkleCount = 0;
      for (let py = 0; py < fh; py++) {
        for (let px = 0; px < fw; px++) {
          const i = (py * fw + px) * 4;
          const luma = faceData.data[i] * 0.299 + faceData.data[i + 1] * 0.587 + faceData.data[i + 2] * 0.114;
          const lumaBlur = blurRef.data[i] * 0.299 + blurRef.data[i + 1] * 0.587 + blurRef.data[i + 2] * 0.114;
          const highPass = luma - lumaBlur;

          if (highPass < -6) {
            wrinkleMask[py * fw + px] = 1;
            wrinkleCount++;
          }
        }
      }

      // Draw wrinkle pixels as colored overlay lines on the canvas
      ctx.save();
      ctx.strokeStyle = 'rgba(255, 74, 74, 0.7)';
      ctx.lineWidth = 1.5;

      // Trace connected wrinkle pixels as short line segments
      for (let py = 0; py < fh; py++) {
        for (let px = 0; px < fw; px++) {
          if (!wrinkleMask[py * fw + px]) continue;

          // Check right and bottom neighbours to draw connecting lines
          const absX = x + px;
          const absY = y + py;

          if (px + 1 < fw && wrinkleMask[py * fw + px + 1]) {
            ctx.beginPath();
            ctx.moveTo(absX, absY);
            ctx.lineTo(absX + 1, absY);
            ctx.stroke();
          }
          if (py + 1 < fh && wrinkleMask[(py + 1) * fw + px]) {
            ctx.beginPath();
            ctx.moveTo(absX, absY);
            ctx.lineTo(absX, absY + 1);
            ctx.stroke();
          }
          // Diagonal
          if (px + 1 < fw && py + 1 < fh && wrinkleMask[(py + 1) * fw + px + 1]) {
            ctx.beginPath();
            ctx.moveTo(absX, absY);
            ctx.lineTo(absX + 1, absY + 1);
            ctx.stroke();
          }
        }
      }
      ctx.restore();

      const wrinklePct = (wrinkleCount / (fw * fh) * 100).toFixed(1);
      changes.push({ label: 'Wrinkles detected', value: `${wrinklePct}% of face area` });
    }

    // 2. Eye enlargement (draw eyes slightly larger)
    const leftEyeCenter = getCenter(keypoints, [33, 133]);
    const rightEyeCenter = getCenter(keypoints, [362, 263]);
    if (leftEyeCenter && rightEyeCenter) {
      const eyeScale = 1.12;
      enlargeRegion(ctx, leftEyeCenter, 25, eyeScale);
      enlargeRegion(ctx, rightEyeCenter, 25, eyeScale);
      changes.push({ label: 'Eye enlargement', value: `${((eyeScale - 1) * 100).toFixed(0)}% larger` });
    }

    // 3. Symmetry nudge — measure asymmetry
    const noseTip = keypoints[1];
    const leftCheek = keypoints[234];
    const rightCheek = keypoints[454];
    if (noseTip && leftCheek && rightCheek) {
      const leftDist = Math.abs(noseTip[0] - leftCheek[0]);
      const rightDist = Math.abs(noseTip[0] - rightCheek[0]);
      const asymmetry = Math.abs(leftDist - rightDist).toFixed(1);
      changes.push({ label: 'Facial symmetry', value: `${asymmetry}px asymmetry detected` });
    }

    return changes;
  }

  function applyBasicFilter(ctx, w, h) {
    // Fallback without face mesh: no-op, image stays as-is
  }

  function getFaceBounds(keypoints) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const pt of keypoints) {
      if (pt[0] < minX) minX = pt[0];
      if (pt[1] < minY) minY = pt[1];
      if (pt[0] > maxX) maxX = pt[0];
      if (pt[1] > maxY) maxY = pt[1];
    }
    const x = Math.max(0, Math.floor(minX));
    const y = Math.max(0, Math.floor(minY));
    const fw = Math.floor(maxX - minX);
    const fh = Math.floor(maxY - minY);
    if (fw <= 0 || fh <= 0) return null;
    return { x, y, fw, fh };
  }

  function getCenter(keypoints, indices) {
    let sx = 0, sy = 0, count = 0;
    for (const i of indices) {
      if (keypoints[i]) {
        sx += keypoints[i][0];
        sy += keypoints[i][1];
        count++;
      }
    }
    if (count === 0) return null;
    return [sx / count, sy / count];
  }

  function enlargeRegion(ctx, center, radius, scale) {
    const [cx, cy] = center;
    const sx = Math.max(0, Math.floor(cx - radius));
    const sy = Math.max(0, Math.floor(cy - radius));
    const size = radius * 2;

    try {
      const region = ctx.getImageData(sx, sy, size, size);
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = size;
      tempCanvas.height = size;
      const tCtx = tempCanvas.getContext('2d');
      tCtx.putImageData(region, 0, 0);

      const newSize = size * scale;
      const offset = (newSize - size) / 2;
      ctx.drawImage(tempCanvas, sx - offset, sy - offset, newSize, newSize);
    } catch (e) {
      // Silently fail if region is out of bounds
    }
  }

  function boxBlur(imageData, radius) {
    const { data, width, height } = imageData;
    const copy = new Uint8ClampedArray(data);
    const size = radius * 2 + 1;

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let r = 0, g = 0, b = 0, count = 0;
        for (let dy = -radius; dy <= radius; dy++) {
          for (let dx = -radius; dx <= radius; dx++) {
            const nx = x + dx;
            const ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const i = (ny * width + nx) * 4;
              r += copy[i];
              g += copy[i + 1];
              b += copy[i + 2];
              count++;
            }
          }
        }
        const i = (y * width + x) * 4;
        data[i] = r / count;
        data[i + 1] = g / count;
        data[i + 2] = b / count;
      }
    }
  }

  function clamp(v) {
    return Math.max(0, Math.min(255, Math.round(v)));
  }

  function dist(a, b) {
    return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
  }

  // ── Symmetry Analysis ──
  // Compares left/right landmark pairs relative to the nose midline
  function analyzeSymmetry(kp) {
    const div = document.getElementById('symmetry-analysis');
    if (!div) return 50;

    // Landmark pairs: [leftIndex, rightIndex, label]
    const pairs = [
      [33, 263, 'Eye (outer corner)'],
      [133, 362, 'Eye (inner corner)'],
      [70, 300, 'Eyebrow (inner)'],
      [105, 334, 'Eyebrow (peak)'],
      [234, 454, 'Cheekbone'],
      [58, 288, 'Jaw (mid)'],
      [132, 361, 'Jaw (lower)'],
      [61, 291, 'Mouth corner'],
    ];

    // Nose midline reference point
    const noseBridge = kp[6];   // bridge of nose
    const noseTip = kp[1];     // tip of nose
    if (!noseBridge || !noseTip) {
      div.innerHTML = 'Could not determine facial midline.';
      return 50;
    }
    const midX = (noseBridge[0] + noseTip[0]) / 2;

    let totalDeviation = 0;
    let count = 0;
    const rows = [];

    for (const [li, ri, label] of pairs) {
      const left = kp[li];
      const right = kp[ri];
      if (!left || !right) continue;

      const leftDist = Math.abs(left[0] - midX);
      const rightDist = Math.abs(right[0] - midX);
      const diff = Math.abs(leftDist - rightDist);
      const avgDist = (leftDist + rightDist) / 2;
      const pctDiff = avgDist > 0 ? (diff / avgDist * 100) : 0;

      totalDeviation += pctDiff;
      count++;

      let cls = 'good';
      if (pctDiff > 15) cls = 'off';
      else if (pctDiff > 7) cls = 'mid';

      rows.push(`<div class="analysis-row">
        <span class="label">${label}</span>
        <span class="value ${cls}">${pctDiff.toFixed(1)}% offset</span>
      </div>`);
    }

    const avgDeviation = count > 0 ? totalDeviation / count : 0;
    const symmetryScore = Math.max(0, 100 - avgDeviation * 3);
    let scoreColor = 'var(--green)';
    if (symmetryScore < 60) scoreColor = 'var(--red)';
    else if (symmetryScore < 80) scoreColor = 'var(--yellow)';

    div.innerHTML = `
      <div class="analysis-title">Bilateral Symmetry</div>
      ${rows.join('')}
      <div class="score-bar">
        <div class="score-label">
          <span>Symmetry Score</span>
          <span style="color:${scoreColor}">${symmetryScore.toFixed(0)}%</span>
        </div>
        <div class="score-track">
          <div class="score-fill" style="width:${symmetryScore}%; background:${scoreColor}"></div>
        </div>
        <div class="score-note">100% = perfectly mirrored. No real human face is perfectly symmetric.</div>
      </div>
    `;

    return symmetryScore;
  }

  // ── Proportionality Analysis ──
  // Checks ratios against classical facial proportion ideals
  function analyzeProportions(kp) {
    const div = document.getElementById('proportion-analysis');
    if (!div) return 50;

    // Key landmarks
    const forehead = kp[10];   // top of forehead
    const browMid = kp[9];    // between brows
    const noseBase = kp[2];   // base of nose
    const chin = kp[152];     // bottom of chin
    const leftEyeOuter = kp[33];
    const rightEyeOuter = kp[263];
    const leftEyeInner = kp[133];
    const rightEyeInner = kp[362];
    const noseWidth_L = kp[48];  // left nostril
    const noseWidth_R = kp[278]; // right nostril
    const mouthLeft = kp[61];
    const mouthRight = kp[291];
    const upperLip = kp[0];
    const lowerLip = kp[17];

    if (!forehead || !chin || !browMid || !noseBase) {
      div.innerHTML = 'Could not measure facial proportions.';
      return 50;
    }

    const rows = [];
    let totalDev = 0;
    let ratioCount = 0;

    function addRatio(label, actual, ideal, unit) {
      const pctOff = Math.abs((actual - ideal) / ideal * 100);
      totalDev += pctOff;
      ratioCount++;
      let cls = 'good';
      if (pctOff > 20) cls = 'off';
      else if (pctOff > 10) cls = 'mid';
      rows.push(`<div class="analysis-row">
        <span class="label">${label}</span>
        <span class="value ${cls}">${actual.toFixed(2)} ${unit} (ideal: ${ideal.toFixed(2)})</span>
      </div>`);
    }

    // 1. Rule of thirds: forehead-to-brow, brow-to-nose-base, nose-base-to-chin should be equal
    const thirdTop = dist(forehead, browMid);
    const thirdMid = dist(browMid, noseBase);
    const thirdBot = dist(noseBase, chin);
    const avgThird = (thirdTop + thirdMid + thirdBot) / 3;

    if (avgThird > 0) {
      addRatio('Upper third (forehead)', thirdTop / avgThird, 1.0, 'ratio');
      addRatio('Middle third (nose)', thirdMid / avgThird, 1.0, 'ratio');
      addRatio('Lower third (chin)', thirdBot / avgThird, 1.0, 'ratio');
    }

    // 2. Eye spacing: distance between inner eyes should ~ equal eye width
    if (leftEyeOuter && leftEyeInner && rightEyeOuter && rightEyeInner) {
      const leftEyeWidth = dist(leftEyeOuter, leftEyeInner);
      const rightEyeWidth = dist(rightEyeInner, rightEyeOuter);
      const eyeGap = dist(leftEyeInner, rightEyeInner);
      const avgEyeWidth = (leftEyeWidth + rightEyeWidth) / 2;
      if (avgEyeWidth > 0) {
        addRatio('Eye spacing / eye width', eyeGap / avgEyeWidth, 1.0, 'ratio');
      }
    }

    // 3. Nose width should ~ equal eye gap (inter-canthal distance)
    if (noseWidth_L && noseWidth_R && leftEyeInner && rightEyeInner) {
      const noseW = dist(noseWidth_L, noseWidth_R);
      const eyeGap = dist(leftEyeInner, rightEyeInner);
      if (eyeGap > 0) {
        addRatio('Nose width / eye gap', noseW / eyeGap, 1.0, 'ratio');
      }
    }

    // 4. Mouth width should ~ 1.5x nose width
    if (mouthLeft && mouthRight && noseWidth_L && noseWidth_R) {
      const mouthW = dist(mouthLeft, mouthRight);
      const noseW = dist(noseWidth_L, noseWidth_R);
      if (noseW > 0) {
        addRatio('Mouth width / nose width', mouthW / noseW, 1.5, 'ratio');
      }
    }

    // 5. Face width-to-height ratio (ideal ~1.6 golden ratio)
    const faceWidth = leftEyeOuter && rightEyeOuter ? dist(leftEyeOuter, rightEyeOuter) : 0;
    const faceHeight = dist(forehead, chin);
    if (faceWidth > 0 && faceHeight > 0) {
      addRatio('Face height / width', faceHeight / faceWidth, 1.6, 'ratio');
    }

    const avgDev = ratioCount > 0 ? totalDev / ratioCount : 0;
    const proportionScore = Math.max(0, 100 - avgDev * 2.5);
    let scoreColor = 'var(--green)';
    if (proportionScore < 60) scoreColor = 'var(--red)';
    else if (proportionScore < 80) scoreColor = 'var(--yellow)';

    div.innerHTML = `
      <div class="analysis-title">Facial Proportions</div>
      ${rows.join('')}
      <div class="score-bar">
        <div class="score-label">
          <span>Proportion Score</span>
          <span style="color:${scoreColor}">${proportionScore.toFixed(0)}%</span>
        </div>
        <div class="score-track">
          <div class="score-fill" style="width:${proportionScore}%; background:${scoreColor}"></div>
        </div>
        <div class="score-note">Based on classical "ideal" ratios (rule of thirds, golden ratio). These are cultural constructs, not objective truths.</div>
      </div>
    `;

    return proportionScore;
  }

  // ── Attractiveness / Beauty Score ──
  // Combines symmetry, proportion, and golden ratio checks into a composite score
  function computeAttractivenessScore(kp, symmetryScore, proportionScore) {
    const div = document.getElementById('attractiveness-score');
    if (!div) return;

    // Default scores if upstream analysis failed
    const symScore = (typeof symmetryScore === 'number') ? symmetryScore : 50;
    const propScore = (typeof proportionScore === 'number') ? proportionScore : 50;

    // Golden ratio checks (phi = 1.618)
    const PHI = 1.618;
    let goldenTotal = 0;
    let goldenCount = 0;

    // 1. Face length / face width ~ phi
    const forehead = kp[10];
    const chin = kp[152];
    const leftCheek = kp[234];
    const rightCheek = kp[454];
    if (forehead && chin && leftCheek && rightCheek) {
      const faceLength = dist(forehead, chin);
      const faceWidth = dist(leftCheek, rightCheek);
      if (faceWidth > 0) {
        const ratio = faceLength / faceWidth;
        const deviation = Math.abs(ratio - PHI) / PHI;
        goldenTotal += Math.max(0, 100 - deviation * 200);
        goldenCount++;
      }
    }

    // 2. Mouth width / nose width ~ phi
    const mouthLeft = kp[61];
    const mouthRight = kp[291];
    const noseLeft = kp[48];
    const noseRight = kp[278];
    if (mouthLeft && mouthRight && noseLeft && noseRight) {
      const mouthW = dist(mouthLeft, mouthRight);
      const noseW = dist(noseLeft, noseRight);
      if (noseW > 0) {
        const ratio = mouthW / noseW;
        const deviation = Math.abs(ratio - PHI) / PHI;
        goldenTotal += Math.max(0, 100 - deviation * 200);
        goldenCount++;
      }
    }

    // 3. Eye width / inter-pupil distance
    const leftEyeOuter = kp[33];
    const leftEyeInner = kp[133];
    const rightEyeOuter = kp[263];
    const rightEyeInner = kp[362];
    if (leftEyeOuter && leftEyeInner && rightEyeOuter && rightEyeInner) {
      const avgEyeWidth = (dist(leftEyeOuter, leftEyeInner) + dist(rightEyeInner, rightEyeOuter)) / 2;
      const pupilDist = dist(getCenter(kp, [33, 133]), getCenter(kp, [362, 263]));
      if (pupilDist > 0) {
        const ratio = avgEyeWidth / pupilDist;
        // Ideal eye width / inter-pupil ~ 0.618 (1/phi)
        const ideal = 1 / PHI;
        const deviation = Math.abs(ratio - ideal) / ideal;
        goldenTotal += Math.max(0, 100 - deviation * 200);
        goldenCount++;
      }
    }

    const goldenScore = goldenCount > 0 ? goldenTotal / goldenCount : 50;

    // Weighted composite: symmetry 35%, proportion 35%, golden ratio 30%
    const composite = symScore * 0.35 + propScore * 0.35 + goldenScore * 0.30;

    let scoreColor = 'var(--green)';
    if (composite < 50) scoreColor = 'var(--red)';
    else if (composite < 70) scoreColor = 'var(--yellow)';

    let goldenColor = 'var(--green)';
    if (goldenScore < 50) goldenColor = 'var(--red)';
    else if (goldenScore < 70) goldenColor = 'var(--yellow)';

    let symColor = 'var(--green)';
    if (symScore < 60) symColor = 'var(--red)';
    else if (symScore < 80) symColor = 'var(--yellow)';

    let propColor = 'var(--green)';
    if (propScore < 60) propColor = 'var(--red)';
    else if (propScore < 80) propColor = 'var(--yellow)';

    div.innerHTML = `
      <div class="analysis-title" style="color:var(--red)">Composite "Beauty" Score</div>

      <div class="score-bar" style="margin-bottom:12px">
        <div class="score-label">
          <span>Overall Attractiveness</span>
          <span style="color:${scoreColor};font-size:1.4em">${composite.toFixed(0)}%</span>
        </div>
        <div class="score-track">
          <div class="score-fill" style="width:${composite}%; background:${scoreColor}"></div>
        </div>
      </div>

      <div class="analysis-row">
        <span class="label">Symmetry (35%)</span>
        <span class="value" style="color:${symColor}">${symScore.toFixed(0)}%</span>
      </div>
      <div class="analysis-row">
        <span class="label">Proportions (35%)</span>
        <span class="value" style="color:${propColor}">${propScore.toFixed(0)}%</span>
      </div>
      <div class="analysis-row">
        <span class="label">Golden Ratio (30%)</span>
        <span class="value" style="color:${goldenColor}">${goldenScore.toFixed(0)}%</span>
      </div>

      <div class="score-note" style="margin-top:12px; color:var(--red); border-top:1px solid rgba(255,74,74,0.3); padding-top:8px">
        This score is pseudoscience. "Attractiveness" cannot be reduced to geometry. These ratios reflect Eurocentric beauty standards from the Renaissance — they are cultural constructs, not universal truths. Algorithms like this are used in dating apps, hiring tools, and social media ranking systems to score real people.
      </div>
    `;
  }
})();
