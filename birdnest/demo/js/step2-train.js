/**
 * step2-train.js — Transfer learning using @tensorflow-models/mobilenet embeddings
 */

(function () {
  let mobilenetModel = null;  // The @tensorflow-models/mobilenet instance
  let trainedModel = null;
  let trainingLogs = { loss: [], val_loss: [], acc: [], val_acc: [] };

  // Config
  let epochs = 20;
  let batchSize = 16;
  let learningRate = 0.001;

  const IMAGE_SIZE = 224;

  /* ---- Controls ---- */
  const epochsSlider = document.getElementById('epochs-slider');
  const epochsValue = document.getElementById('epochs-value');
  const btnTrain = document.getElementById('btn-train');
  const btnSave = document.getElementById('btn-save-model');
  const btnLoad = document.getElementById('btn-load-model');
  const modelFileInput = document.getElementById('model-file-input');

  epochsSlider.addEventListener('input', () => {
    epochs = parseInt(epochsSlider.value);
    epochsValue.textContent = epochs;
  });

  // Batch size toggle
  document.querySelectorAll('[data-batch]').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('[data-batch]').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      batchSize = parseInt(chip.dataset.batch);
    });
  });

  // Learning rate toggle
  document.querySelectorAll('[data-lr]').forEach(chip => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('[data-lr]').forEach(c => c.classList.remove('active'));
      chip.classList.add('active');
      learningRate = parseFloat(chip.dataset.lr);
    });
  });

  /* ---- Load MobileNet via official package ---- */
  async function loadMobileNet() {
    const dot = document.getElementById('mobilenet-dot');
    const detail = document.getElementById('mobilenet-detail');
    detail.textContent = 'loading (~7 MB)...';

    try {
      // Use the @tensorflow-models/mobilenet package (loaded via CDN)
      mobilenetModel = await mobilenet.load({ version: 2, alpha: 1.0 });

      dot.className = 'model-load-dot ready';
      detail.textContent = 'ready';
      btnTrain.disabled = false;
    } catch (err) {
      dot.className = 'model-load-dot error';
      detail.textContent = 'failed — ' + err.message;
      console.error('MobileNet load error:', err);
    }
  }

  /* ---- Extract embeddings from an image element ---- */
  function getEmbedding(imgElement) {
    // infer(img, embedding=true) returns a 1D feature vector (1280-dim for MobileNet v2)
    return mobilenetModel.infer(imgElement, true);
  }

  /* ---- Build Dataset Tensors ---- */
  async function buildDataset(statusFn) {
    const birdnestImages = await App.getImagesByCategory('birdnest');
    const notBirdnestImages = await App.getImagesByCategory('not_birdnest');

    if (birdnestImages.length < 2 || notBirdnestImages.length < 2) {
      alert('Need at least 2 images per category to train.');
      return null;
    }

    const allImages = [
      ...birdnestImages.map(r => ({ blob: r.blob, label: 1 })),
      ...notBirdnestImages.map(r => ({ blob: r.blob, label: 0 }))
    ];

    // Shuffle
    for (let i = allImages.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [allImages[i], allImages[j]] = [allImages[j], allImages[i]];
    }

    const features = [];
    const labels = [];

    for (let i = 0; i < allImages.length; i++) {
      const item = allImages[i];
      statusFn(`Extracting features: ${i + 1}/${allImages.length}`);
      const img = await App.loadImage(item.blob);
      const embedding = getEmbedding(img);
      features.push(embedding);
      labels.push(item.label);
      // Small delay to keep UI responsive
      if (i % 5 === 0) await tf.nextFrame();
    }

    const xs = tf.concat(features);
    features.forEach(f => f.dispose());
    const ys = tf.tensor1d(labels, 'int32');

    return { xs, ys, total: allImages.length };
  }

  /* ---- Build Classification Head ---- */
  function buildHead(inputShape) {
    const model = tf.sequential();
    // Input is a flat 1280-dim embedding vector from MobileNet
    model.add(tf.layers.dense({ inputShape, units: 128, activation: 'relu' }));
    model.add(tf.layers.dropout({ rate: 0.3 }));
    model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: 'binaryCrossentropy',
      metrics: ['accuracy']
    });

    return model;
  }

  /* ---- Train ---- */
  btnTrain.addEventListener('click', async () => {
    btnTrain.disabled = true;
    trainingLogs = { loss: [], val_loss: [], acc: [], val_acc: [] };

    const progressBar = document.getElementById('training-progress');
    const fill = document.getElementById('train-progress-fill');
    const pct = document.getElementById('train-progress-pct');
    const steps = document.getElementById('train-progress-steps');
    const chartContainer = document.getElementById('training-chart-container');
    const confusionContainer = document.getElementById('confusion-matrix');
    const modelActions = document.getElementById('model-actions');

    progressBar.classList.add('visible');
    chartContainer.classList.remove('hidden');
    confusionContainer.classList.add('hidden');
    modelActions.classList.add('hidden');

    const dataset = await buildDataset((msg) => {
      steps.textContent = msg;
    });
    if (!dataset) {
      progressBar.classList.remove('visible');
      btnTrain.disabled = false;
      return;
    }

    const { xs, ys, total } = dataset;

    // Split train/validation (80/20)
    const splitIdx = Math.max(1, Math.floor(total * 0.8));
    const xTrain = xs.slice(0, splitIdx);
    const yTrain = ys.slice(0, splitIdx).toFloat();
    const xVal = xs.slice(splitIdx);
    const yVal = ys.slice(splitIdx).toFloat();

    // Build head — input shape is [1280] for mobilenet v2
    const featureShape = xs.shape.slice(1);
    const head = buildHead(featureShape);

    steps.textContent = 'Training...';

    await head.fit(xTrain, yTrain, {
      epochs,
      batchSize,
      validationData: [xVal, yVal],
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const p = ((epoch + 1) / epochs) * 100;
          fill.style.width = p + '%';
          pct.textContent = Math.round(p) + '%';
          steps.textContent = `Epoch ${epoch + 1} / ${epochs}`;

          trainingLogs.loss.push(logs.loss);
          trainingLogs.val_loss.push(logs.val_loss);
          trainingLogs.acc.push(logs.acc);
          trainingLogs.val_acc.push(logs.val_acc);

          drawChart('chart-loss', trainingLogs.loss, trainingLogs.val_loss, 'Loss');
          drawChart('chart-accuracy', trainingLogs.acc, trainingLogs.val_acc, 'Accuracy');
        }
      }
    });

    trainedModel = head;

    // Confusion matrix on validation set
    const valPreds = head.predict(xVal);
    const predLabels = (await valPreds.data()).map(v => v > 0.5 ? 1 : 0);
    const trueLabels = await yVal.data();
    showConfusionMatrix(Array.from(trueLabels), predLabels);

    // Cleanup tensors
    xs.dispose();
    ys.dispose();
    xTrain.dispose();
    yTrain.dispose();
    xVal.dispose();
    yVal.dispose();
    valPreds.dispose();

    // Cache model in IndexedDB
    await cacheModel(head);

    confusionContainer.classList.remove('hidden');
    modelActions.classList.remove('hidden');
    btnTrain.disabled = false;

    // Update Step 3 status
    const classifyDot = document.getElementById('classify-model-dot');
    const classifyDetail = document.getElementById('classify-model-detail');
    if (classifyDot) {
      classifyDot.className = 'model-load-dot ready';
      classifyDetail.textContent = 'ready';
    }
  });

  /* ---- Chart Drawing ---- */
  function drawChart(canvasId, trainData, valData, label) {
    const canvas = document.getElementById(canvasId);
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const pad = { top: 10, right: 10, bottom: 25, left: 40 };

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#0a0a0a';
    ctx.fillRect(0, 0, w, h);

    if (trainData.length === 0) return;

    const all = [...trainData, ...valData];
    const maxVal = Math.max(...all) * 1.1 || 1;
    const minVal = Math.min(0, Math.min(...all));
    const range = maxVal - minVal || 1;

    const plotW = w - pad.left - pad.right;
    const plotH = h - pad.top - pad.bottom;

    // Grid lines
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (plotH / 4) * i;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(w - pad.right, y);
      ctx.stroke();

      const val = maxVal - (range / 4) * i;
      ctx.fillStyle = '#555';
      ctx.font = '9px monospace';
      ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(2), pad.left - 4, y + 3);
    }

    // Epoch labels
    ctx.fillStyle = '#555';
    ctx.textAlign = 'center';
    ctx.font = '9px monospace';
    const step = Math.max(1, Math.floor(trainData.length / 5));
    for (let i = 0; i < trainData.length; i += step) {
      const x = pad.left + (i / (trainData.length - 1 || 1)) * plotW;
      ctx.fillText(i + 1, x, h - 5);
    }

    // Draw lines
    function drawLine(data, color) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();
      data.forEach((val, i) => {
        const x = pad.left + (i / (data.length - 1 || 1)) * plotW;
        const y = pad.top + ((maxVal - val) / range) * plotH;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    drawLine(trainData, '#4a9eff');
    drawLine(valData, '#ffd94a');

    // Legend
    ctx.font = '9px monospace';
    ctx.fillStyle = '#4a9eff';
    ctx.textAlign = 'left';
    ctx.fillText('train', pad.left + 4, pad.top + 12);
    ctx.fillStyle = '#ffd94a';
    ctx.fillText('val', pad.left + 44, pad.top + 12);
  }

  /* ---- Confusion Matrix ---- */
  function showConfusionMatrix(trueLabels, predLabels) {
    let tp = 0, fp = 0, fn = 0, tn = 0;
    for (let i = 0; i < trueLabels.length; i++) {
      if (trueLabels[i] === 1 && predLabels[i] === 1) tp++;
      else if (trueLabels[i] === 0 && predLabels[i] === 1) fp++;
      else if (trueLabels[i] === 1 && predLabels[i] === 0) fn++;
      else tn++;
    }

    const grid = document.getElementById('matrix-grid');
    grid.innerHTML = `
      <div class="matrix-header"></div>
      <div class="matrix-header">Pred: Nest</div>
      <div class="matrix-header">Pred: Not</div>
      <div class="matrix-label">Actual: Nest</div>
      <div class="matrix-cell correct">${tp}</div>
      <div class="matrix-cell incorrect">${fn}</div>
      <div class="matrix-label">Actual: Not</div>
      <div class="matrix-cell incorrect">${fp}</div>
      <div class="matrix-cell correct">${tn}</div>
    `;
  }

  /* ---- Save / Load Model ---- */
  async function cacheModel(model) {
    await model.save('indexeddb://birdnest-classifier');
  }

  btnSave.addEventListener('click', async () => {
    if (!trainedModel) return;
    await trainedModel.save('downloads://birdnest-model');
  });

  btnLoad.addEventListener('click', () => modelFileInput.click());

  modelFileInput.addEventListener('change', async (e) => {
    const files = Array.from(e.target.files);
    if (files.length === 0) return;

    const jsonFile = files.find(f => f.name.endsWith('.json'));
    if (!jsonFile) {
      alert('Please select the model .json file');
      return;
    }

    try {
      trainedModel = await tf.loadLayersModel(tf.io.browserFiles(files));
      await cacheModel(trainedModel);

      const classifyDot = document.getElementById('classify-model-dot');
      const classifyDetail = document.getElementById('classify-model-detail');
      classifyDot.className = 'model-load-dot ready';
      classifyDetail.textContent = 'ready (loaded)';

      document.getElementById('model-actions').classList.remove('hidden');
    } catch (err) {
      console.error('Model load error:', err);
      alert('Failed to load model. Make sure you select both .json and .bin files.');
    }
  });

  /* ---- Try loading cached model on init ---- */
  async function tryLoadCachedModel() {
    try {
      trainedModel = await tf.loadLayersModel('indexeddb://birdnest-classifier');
      const classifyDot = document.getElementById('classify-model-dot');
      const classifyDetail = document.getElementById('classify-model-detail');
      classifyDot.className = 'model-load-dot ready';
      classifyDetail.textContent = 'ready (cached)';
      document.getElementById('model-actions').classList.remove('hidden');
    } catch {
      // No cached model
    }
  }

  // Expose for Step 3
  window.BirdnestTrain = {
    getMobileNet: () => mobilenetModel,
    getModel: () => trainedModel,
    getEmbedding,
    IMAGE_SIZE
  };

  // Init
  const origInit = App.init.bind(App);
  App.init = async function () {
    await origInit();
    loadMobileNet();
    tryLoadCachedModel();
  };

  // Allow multi-file select for model loading
  modelFileInput.setAttribute('multiple', '');
})();
