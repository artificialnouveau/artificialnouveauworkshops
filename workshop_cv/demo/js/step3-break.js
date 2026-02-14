/**
 * step3-break.js — Single upload → classify → challenge → compare
 */

(function () {
  const fileInput = document.getElementById('file-input-3');
  const fileInputB = document.getElementById('file-input-3b');
  const fileInputRetry = document.getElementById('file-input-3-retry');
  const uploadArea = document.getElementById('upload-area-3');
  const uploadAreaB = document.getElementById('upload-area-3b');
  const phase1 = document.getElementById('step3-phase1');
  const phase2 = document.getElementById('step3-phase2');

  let imgA = null;
  let predsA = null;

  // ── First upload: classify and show results ──
  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    uploadArea.classList.add('has-file');
    phase1.classList.remove('hidden');
    phase2.classList.add('hidden');

    imgA = await App.loadImage(file);
    predsA = await classifyAndDraw(imgA, 'canvas-break-a', 'predictions-a', 'meter-a', 'meter-value-a');
  });

  // ── Challenge upload: classify trick photo, show comparison ──
  fileInputB.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    uploadAreaB.classList.add('has-file');

    const imgB = await App.loadImage(file);

    // Re-draw original in comparison view
    classifyAndDraw(imgA, 'canvas-break-a2', 'predictions-a2', 'meter-a2', 'meter-value-a2');

    // Classify trick photo
    const predsB = await classifyAndDraw(imgB, 'canvas-break-b', 'predictions-b', 'meter-b', 'meter-value-b');

    phase2.classList.remove('hidden');
    showVerdict(predsA, predsB);
  });

  // ── Retry: start over with new photo ──
  fileInputRetry.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Reset state
    uploadArea.classList.add('has-file');
    uploadAreaB.classList.remove('has-file');
    phase2.classList.add('hidden');

    imgA = await App.loadImage(file);
    predsA = await classifyAndDraw(imgA, 'canvas-break-a', 'predictions-a', 'meter-a', 'meter-value-a');
  });

  async function classifyAndDraw(img, canvasId, predsId, meterId, meterValId) {
    const canvas = document.getElementById(canvasId);
    App.drawToCanvas(canvas, img, 340);

    if (!App.models.mobilenet) {
      document.getElementById(predsId).innerHTML = '<p style="color:var(--red)">Model unavailable</p>';
      return null;
    }

    const predictions = await App.models.mobilenet.classify(img, 5);
    App.renderPredictions(document.getElementById(predsId), predictions);

    // Confidence meter
    const topConf = predictions[0].probability * 100;
    const meterFill = document.getElementById(meterId);
    const meterVal = document.getElementById(meterValId);
    meterFill.style.width = topConf + '%';
    meterFill.className = 'meter-fill ' + (topConf > 70 ? 'high' : topConf > 40 ? 'medium' : 'low');
    meterVal.textContent = topConf.toFixed(1) + '%';

    return predictions;
  }

  function showVerdict(predsA, predsB) {
    const verdict = document.getElementById('break-verdict');
    if (!predsA || !predsB) {
      verdict.innerHTML = '<p>Could not compare — model unavailable.</p>';
      return;
    }

    const confA = predsA[0].probability;
    const confB = predsB[0].probability;
    const sameLabel = predsA[0].className === predsB[0].className;
    const confDrop = ((confA - confB) / confA * 100).toFixed(0);

    if (!sameLabel) {
      verdict.innerHTML = `<p>You changed the machine's mind entirely.<br>
        It saw "<strong>${predsA[0].className.split(',')[0]}</strong>" before, now it sees "<strong>${predsB[0].className.split(',')[0]}</strong>".<br>
        <span style="color:var(--accent)">That's how fragile "machine understanding" is.</span></p>`;
      verdict.style.borderLeftColor = 'var(--green, #00ff66)';
    } else if (confB < confA * 0.7) {
      verdict.innerHTML = `<p>Same label, but confidence dropped by ${Math.abs(confDrop)}%.<br>
        The model is less sure about what it sees.<br>
        <span style="color:var(--accent)">A small change in framing destabilizes the whole system.</span></p>`;
      verdict.style.borderLeftColor = 'var(--yellow, #ffd94a)';
    } else {
      verdict.innerHTML = `<p>The model held steady this time — same label, similar confidence.<br>
        <span style="color:var(--accent)">Try again: cover part of the subject, change the angle dramatically, or add something unexpected to the frame.</span></p>`;
      verdict.style.borderLeftColor = 'var(--accent)';
    }
  }
})();
