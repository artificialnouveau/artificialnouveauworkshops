/**
 * step-dam.js — Dam Square Group Detection (fully client-side)
 *
 * Uses COCO-SSD (already loaded by the page) for person detection,
 * DBSCAN in plain JS for clustering, and canvas for rendering
 * group overlays + per-person glitch effects.
 *
 * Video source: user's webcam.
 */

(function () {
  var video = document.getElementById('dam-video');
  var canvas = document.getElementById('dam-canvas');
  var ctx = canvas.getContext('2d');
  var startBtn = document.getElementById('dam-start-btn');
  var stopBtn = document.getElementById('dam-stop-btn');
  var peopleEl = document.getElementById('dam-people');
  var groupsEl = document.getElementById('dam-groups');
  var loneEl = document.getElementById('dam-lone');
  var legendEl = document.getElementById('dam-legend');
  var modeGroupsBtn = document.getElementById('dam-mode-groups');
  var modeGlitchBtn = document.getElementById('dam-mode-glitch');
  var distSlider = document.getElementById('dam-dist');
  var distVal = document.getElementById('dam-dist-val');
  var minSlider = document.getElementById('dam-min');
  var minVal = document.getElementById('dam-min-val');
  var confSlider = document.getElementById('dam-conf');
  var confVal = document.getElementById('dam-conf-val');
  var glitchSlider = document.getElementById('dam-glitch');
  var glitchVal = document.getElementById('dam-glitch-val');
  var decaySlider = document.getElementById('dam-decay');
  var decayVal = document.getElementById('dam-decay-val');
  var clearBtn = document.getElementById('dam-clear');
  var groupControls = document.getElementById('dam-group-controls');
  var glitchControls = document.getElementById('dam-glitch-controls');

  var model = null;
  var stream = null;
  var running = false;
  var mode = 'groups';
  var heatmap = null;

  var GROUP_COLORS = [
    '#ff0000', '#ffc800', '#0064ff', '#64ff00', '#c800ff',
    '#ffff00', '#ff3264', '#00ffff', '#32ff32', '#ff00c8',
  ];
  var NEON_COLORS = [
    [255, 50, 50], [50, 255, 50], [50, 50, 255],
    [255, 255, 0], [255, 0, 255], [0, 255, 255],
  ];

  // ── DBSCAN ──────────────────────────────────────────────────────────

  function dbscan(points, eps, minPts) {
    var n = points.length;
    var labels = new Array(n).fill(-1); // -1 = noise
    var clusterId = 0;

    function dist(a, b) {
      var dx = a[0] - b[0], dy = a[1] - b[1];
      return Math.sqrt(dx * dx + dy * dy);
    }

    function regionQuery(idx) {
      var neighbors = [];
      for (var i = 0; i < n; i++) {
        if (dist(points[idx], points[i]) <= eps) neighbors.push(i);
      }
      return neighbors;
    }

    for (var i = 0; i < n; i++) {
      if (labels[i] !== -1) continue;
      var neighbors = regionQuery(i);
      if (neighbors.length < minPts) continue; // noise
      labels[i] = clusterId;
      var seed = neighbors.slice();
      var j = 0;
      while (j < seed.length) {
        var q = seed[j];
        if (labels[q] === -1) labels[q] = clusterId; // was noise, claim it
        if (labels[q] !== -1 && labels[q] !== clusterId) { j++; continue; } // already in another cluster — skip expansion but keep assignment above
        labels[q] = clusterId;
        var qNeighbors = regionQuery(q);
        if (qNeighbors.length >= minPts) {
          for (var k = 0; k < qNeighbors.length; k++) {
            if (seed.indexOf(qNeighbors[k]) === -1) seed.push(qNeighbors[k]);
          }
        }
        j++;
      }
      clusterId++;
    }
    return labels;
  }

  // ── Heatmap ─────────────────────────────────────────────────────────

  function stampHeatmap(w, h, centers) {
    if (!heatmap || heatmap.width !== w || heatmap.height !== h) {
      var c = document.createElement('canvas');
      c.width = w;
      c.height = h;
      heatmap = c;
      var hctx = heatmap.getContext('2d');
      hctx.fillStyle = '#000';
      hctx.fillRect(0, 0, w, h);
    }
    var hctx = heatmap.getContext('2d');

    // Decay
    var decay = parseFloat(decaySlider.value) / 1000;
    var imgData = hctx.getImageData(0, 0, w, h);
    var d = imgData.data;
    for (var i = 0; i < d.length; i += 4) {
      d[i] = Math.floor(d[i] * decay);
      d[i + 1] = Math.floor(d[i + 1] * decay);
      d[i + 2] = Math.floor(d[i + 2] * decay);
    }
    hctx.putImageData(imgData, 0, 0);

    // Stamp blobs
    for (var i = 0; i < centers.length; i++) {
      var cx = centers[i][0], cy = centers[i][1];
      var grad = hctx.createRadialGradient(cx, cy, 0, cx, cy, 25);
      grad.addColorStop(0, 'rgba(255, 100, 0, 0.8)');
      grad.addColorStop(0.5, 'rgba(255, 0, 100, 0.3)');
      grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      hctx.fillStyle = grad;
      hctx.fillRect(cx - 25, cy - 25, 50, 50);
    }
  }

  // ── Glitch effects ──────────────────────────────────────────────────

  function drawGlitchPerson(ctx, box, intensity, colorIdx) {
    var x = box[0], y = box[1], w = box[2] - box[0], h = box[3] - box[1];
    var color = NEON_COLORS[colorIdx % NEON_COLORS.length];
    var pad = Math.floor(5 + intensity * 15);

    // Expanded region
    var gx = Math.max(0, x - pad);
    var gy = Math.max(0, y - pad);
    var gw = Math.min(canvas.width - gx, w + pad * 2);
    var gh = Math.min(canvas.height - gy, h + pad * 2);
    if (gw < 4 || gh < 4) return;

    // Ghost echoes (offset chromatic copies)
    var numEchoes = Math.floor(1 + intensity * 2);
    ctx.globalAlpha = 0.15 + intensity * 0.1;
    for (var e = 0; e < numEchoes; e++) {
      var dx = Math.floor((Math.random() - 0.5) * (10 + intensity * 25));
      var dy = Math.floor((Math.random() - 0.5) * (5 + intensity * 12));
      try {
        var imgData = ctx.getImageData(gx, gy, gw, gh);
        // Tint to single channel
        var d = imgData.data;
        var ch = e % 3;
        for (var i = 0; i < d.length; i += 4) {
          if (ch !== 0) d[i] = 0;
          if (ch !== 1) d[i + 1] = 0;
          if (ch !== 2) d[i + 2] = 0;
        }
        ctx.putImageData(imgData, gx + dx, gy + dy);
      } catch (err) { /* ignore out-of-bounds */ }
    }
    ctx.globalAlpha = 1.0;

    // RGB split on the person region
    try {
      var shift = Math.floor(2 + intensity * 8);
      var region = ctx.getImageData(gx, gy, gw, gh);
      var rd = region.data;
      var copy = new Uint8ClampedArray(rd);
      for (var row = 0; row < gh; row++) {
        for (var col = 0; col < gw; col++) {
          var idx = (row * gw + col) * 4;
          // Shift red channel right
          var srcR = col - shift;
          if (srcR >= 0) rd[idx] = copy[(row * gw + srcR) * 4];
          // Shift blue channel left
          var srcB = col + shift;
          if (srcB < gw) rd[idx + 2] = copy[(row * gw + srcB) * 4 + 2];
        }
      }

      // Row displacement
      var numSlices = Math.floor(2 + intensity * 5);
      for (var s = 0; s < numSlices; s++) {
        var sy = Math.floor(Math.random() * gh);
        var sh = Math.floor(1 + Math.random() * (2 + intensity * 4));
        var sShift = Math.floor((Math.random() - 0.5) * (8 + intensity * 30));
        for (var row = sy; row < Math.min(sy + sh, gh); row++) {
          for (var col = 0; col < gw; col++) {
            var src = col - sShift;
            if (src >= 0 && src < gw) {
              var dIdx = (row * gw + col) * 4;
              var sIdx = (row * gw + src) * 4;
              rd[dIdx] = copy[sIdx];
              rd[dIdx + 1] = copy[sIdx + 1];
              rd[dIdx + 2] = copy[sIdx + 2];
            }
          }
        }
      }

      // Scanlines
      var spacing = Math.max(2, Math.floor(4 - intensity * 2));
      for (var row = 0; row < gh; row += spacing) {
        for (var col = 0; col < gw; col++) {
          var idx = (row * gw + col) * 4;
          rd[idx] = Math.floor(rd[idx] * 0.5);
          rd[idx + 1] = Math.floor(rd[idx + 1] * 0.5);
          rd[idx + 2] = Math.floor(rd[idx + 2] * 0.5);
        }
      }

      // Color tint
      var tintAlpha = 0.1 + intensity * 0.12;
      for (var i = 0; i < rd.length; i += 4) {
        rd[i] = Math.floor(rd[i] * (1 - tintAlpha) + color[0] * tintAlpha);
        rd[i + 1] = Math.floor(rd[i + 1] * (1 - tintAlpha) + color[1] * tintAlpha);
        rd[i + 2] = Math.floor(rd[i + 2] * (1 - tintAlpha) + color[2] * tintAlpha);
      }

      ctx.putImageData(region, gx, gy);
    } catch (err) { /* ignore */ }

    // Double-stroke bounding box
    var offset = Math.floor(2 + intensity * 4);
    ctx.strokeStyle = 'rgb(' + color[0] + ',' + color[1] + ',' + color[2] + ')';
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, w, h);
    ctx.strokeStyle = 'rgb(' + color[0] + ',0,' + color[2] + ')';
    ctx.strokeRect(x + offset, y - offset, w, h);
  }

  // ── Drawing ─────────────────────────────────────────────────────────

  function drawGroups(boxes, labels) {
    var groupPoints = {};
    for (var i = 0; i < boxes.length; i++) {
      var b = boxes[i], l = labels[i];
      var cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2;
      var color = l === -1 ? '#888' : GROUP_COLORS[l % GROUP_COLORS.length];

      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);

      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.font = '12px system-ui';
      ctx.fillStyle = color;
      ctx.fillText(l === -1 ? 'individual' : 'Group ' + (l + 1), b[0], b[1] - 6);

      if (l !== -1) {
        if (!groupPoints[l]) groupPoints[l] = [];
        groupPoints[l].push([cx, cy]);
      }
    }

    // Draw connecting lines for groups
    var keys = Object.keys(groupPoints);
    for (var k = 0; k < keys.length; k++) {
      var label = keys[k];
      var pts = groupPoints[label];
      var color = GROUP_COLORS[parseInt(label) % GROUP_COLORS.length];
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      for (var i = 0; i < pts.length; i++) {
        for (var j = i + 1; j < pts.length; j++) {
          ctx.beginPath();
          ctx.moveTo(pts[i][0], pts[i][1]);
          ctx.lineTo(pts[j][0], pts[j][1]);
          ctx.stroke();
        }
      }

      // Convex hull fill (simplified: just connect all points)
      if (pts.length >= 3) {
        ctx.fillStyle = color.replace(')', ',0.1)').replace('rgb', 'rgba');
        ctx.beginPath();
        ctx.moveTo(pts[0][0], pts[0][1]);
        for (var i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i][0], pts[i][1]);
        }
        ctx.closePath();
        ctx.fill();
      }

      // Group size label
      var avgX = 0, avgY = 0;
      for (var i = 0; i < pts.length; i++) { avgX += pts[i][0]; avgY += pts[i][1]; }
      avgX /= pts.length; avgY /= pts.length;
      ctx.font = 'bold 14px system-ui';
      ctx.fillStyle = color;
      ctx.fillText(pts.length + ' people', avgX - 25, avgY - 15);
    }
  }

  function updateLegend(labels) {
    legendEl.innerHTML = '';
    var counts = {};
    for (var i = 0; i < labels.length; i++) {
      if (labels[i] !== -1) {
        counts[labels[i]] = (counts[labels[i]] || 0) + 1;
      }
    }
    var keys = Object.keys(counts).sort(function (a, b) { return parseInt(a) - parseInt(b); });
    for (var i = 0; i < keys.length; i++) {
      var l = keys[i];
      var div = document.createElement('div');
      div.style.cssText = 'display:flex;align-items:center;gap:8px;margin-bottom:6px;font-size:0.85rem;';
      var swatch = document.createElement('div');
      swatch.style.cssText = 'width:20px;height:10px;border-radius:3px;background:' +
        GROUP_COLORS[parseInt(l) % GROUP_COLORS.length];
      var span = document.createElement('span');
      span.textContent = 'Group ' + (parseInt(l) + 1) + ' (' + counts[l] + ' people)';
      div.appendChild(swatch);
      div.appendChild(span);
      legendEl.appendChild(div);
    }
  }

  // ── Main detection loop ─────────────────────────────────────────────

  async function loadModel() {
    if (model) return;
    startBtn.textContent = 'Loading model...';
    startBtn.disabled = true;
    model = await cocoSsd.load();
    startBtn.textContent = 'Start Webcam';
    startBtn.disabled = false;
  }

  async function detect() {
    if (!running) return;

    var predictions = await model.detect(video);

    // Filter to person class only
    var conf = parseFloat(confSlider.value) / 100;
    var persons = predictions.filter(function (p) {
      return p.class === 'person' && p.score >= conf;
    });

    var boxes = persons.map(function (p) {
      return [
        Math.round(p.bbox[0]),
        Math.round(p.bbox[1]),
        Math.round(p.bbox[0] + p.bbox[2]),
        Math.round(p.bbox[1] + p.bbox[3]),
      ];
    });

    var centers = boxes.map(function (b) {
      return [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
    });

    var eps = parseFloat(distSlider.value);
    var minPts = parseInt(minSlider.value);
    var labels = centers.length >= minPts ? dbscan(centers, eps, minPts) : centers.map(function () { return -1; });

    // Draw frame
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    // Always update heatmap
    stampHeatmap(canvas.width, canvas.height, centers);

    if (mode === 'heatmap') {
      // Draw heatmap underneath
      ctx.globalAlpha = 0.5;
      ctx.drawImage(heatmap, 0, 0);
      ctx.globalAlpha = 1.0;

      // Per-person glitch
      var intensity = parseFloat(glitchSlider.value) / 100;
      for (var i = 0; i < boxes.length; i++) {
        drawGlitchPerson(ctx, boxes[i], intensity, i);
      }

      // Glitchy HUD
      ctx.font = 'bold 16px monospace';
      ctx.fillStyle = '#00ffff';
      ctx.fillText('TRACES: ' + boxes.length + '  //  GLITCH', 12, 28);
      ctx.fillStyle = '#ff00c8';
      ctx.fillText('TRACES: ' + boxes.length + '  //  GLITCH', 10, 26);
    } else {
      drawGroups(boxes, labels);
      ctx.font = 'bold 16px system-ui';
      ctx.fillStyle = '#fff';
      ctx.fillText('People: ' + boxes.length + '  Groups: ' + Object.keys(labels.reduce(function (a, l) { if (l !== -1) a[l] = 1; return a; }, {})).length, 10, 26);
    }

    // Update stats
    var groupCounts = {};
    var lone = 0;
    for (var i = 0; i < labels.length; i++) {
      if (labels[i] === -1) lone++;
      else groupCounts[labels[i]] = (groupCounts[labels[i]] || 0) + 1;
    }
    peopleEl.textContent = boxes.length;
    groupsEl.textContent = Object.keys(groupCounts).length;
    loneEl.textContent = lone;
    updateLegend(labels);

    requestAnimationFrame(detect);
  }

  async function startWebcam() {
    await loadModel();
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } },
      });
      video.srcObject = stream;
      await video.play();
      running = true;
      startBtn.style.display = 'none';
      stopBtn.style.display = 'inline-block';
      canvas.style.display = 'block';
      detect();
    } catch (err) {
      alert('Could not access webcam: ' + err.message);
    }
  }

  function stopWebcam() {
    running = false;
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    video.srcObject = null;
    startBtn.style.display = 'inline-block';
    stopBtn.style.display = 'none';
  }

  // ── Event listeners ─────────────────────────────────────────────────

  startBtn.addEventListener('click', startWebcam);
  stopBtn.addEventListener('click', stopWebcam);

  modeGroupsBtn.addEventListener('click', function () {
    mode = 'groups';
    modeGroupsBtn.classList.add('active-mode');
    modeGlitchBtn.classList.remove('active-mode');
    groupControls.style.display = 'block';
    glitchControls.style.display = 'none';
  });

  modeGlitchBtn.addEventListener('click', function () {
    mode = 'heatmap';
    modeGlitchBtn.classList.add('active-mode');
    modeGroupsBtn.classList.remove('active-mode');
    groupControls.style.display = 'none';
    glitchControls.style.display = 'block';
  });

  distSlider.addEventListener('input', function () { distVal.textContent = this.value + 'px'; });
  minSlider.addEventListener('input', function () { minVal.textContent = this.value; });
  confSlider.addEventListener('input', function () { confVal.textContent = (this.value / 100).toFixed(2); });
  glitchSlider.addEventListener('input', function () { glitchVal.textContent = (this.value / 100).toFixed(2); });
  decaySlider.addEventListener('input', function () { decayVal.textContent = (this.value / 1000).toFixed(3); });
  clearBtn.addEventListener('click', function () { heatmap = null; });

})();
