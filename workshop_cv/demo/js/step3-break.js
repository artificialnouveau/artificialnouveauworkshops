/**
 * step3-break.js — Side-by-side "stump the model" comparison
 */

(function () {
  const fileInputA = document.getElementById('file-input-3a');
  const fileInputB = document.getElementById('file-input-3b');
  const results = document.getElementById('step3-results');
  const uploadAreaA = document.getElementById('upload-area-3a');
  const uploadAreaB = document.getElementById('upload-area-3b');

  let imgA = null;
  let imgB = null;
  let predsA = null;
  let predsB = null;

  fileInputA.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    uploadAreaA.classList.add('has-file');
    imgA = await App.loadImage(file);
    predsA = await classifyAndDraw(imgA, 'canvas-break-a', 'predictions-a', 'meter-a', 'meter-value-a');
    checkBothReady();
  });

  fileInputB.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    uploadAreaB.classList.add('has-file');
    imgB = await App.loadImage(file);
    predsB = await classifyAndDraw(imgB, 'canvas-break-b', 'predictions-b', 'meter-b', 'meter-value-b');
    checkBothReady();
  });

  async function classifyAndDraw(img, canvasId, predsId, meterId, meterValId) {
    const canvas = document.getElementById(canvasId);
    App.drawToCanvas(canvas, img, 340);
    results.classList.remove('hidden');

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

  function checkBothReady() {
    if (!predsA || !predsB) return;

    const verdict = document.getElementById('break-verdict');
    const confA = predsA[0].probability;
    const confB = predsB[0].probability;
    const sameLabel = predsA[0].className === predsB[0].className;
    const confDrop = ((confA - confB) / confA * 100).toFixed(0);

    if (!sameLabel) {
      verdict.innerHTML = `<p>You changed the machine's mind entirely.<br>
        It saw "<strong>${predsA[0].className.split(',')[0]}</strong>" before, now it sees "<strong>${predsB[0].className.split(',')[0]}</strong>".<br>
        <span style="color:var(--accent)">That's how fragile "machine understanding" is.</span></p>`;
    } else if (confB < confA * 0.7) {
      verdict.innerHTML = `<p>Same label, but confidence dropped by ${Math.abs(confDrop)}%.<br>
        The model is less sure about what it sees.<br>
        <span style="color:var(--accent)">A small change in framing destabilizes the whole system.</span></p>`;
    } else {
      verdict.innerHTML = `<p>The model held steady this time — same label, similar confidence.<br>
        <span style="color:var(--accent)">Try again: cover part of the subject, change the angle dramatically, or add something unexpected to the frame.</span></p>`;
    }
  }
})();
