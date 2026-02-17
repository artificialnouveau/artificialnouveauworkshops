/**
 * step3-classify.js — Run classifier on new images
 */

(function () {
  const fileInput = document.getElementById('classify-file-input');
  const uploadArea = document.getElementById('upload-area-classify');
  const resultsGrid = document.getElementById('classify-results');

  uploadArea.addEventListener('click', () => fileInput.click());

  // Drag and drop
  uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
  });

  uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
  });

  uploadArea.addEventListener('drop', async (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files).filter(f => f.type.startsWith('image/'));
    await classifyFiles(files);
  });

  fileInput.addEventListener('change', async () => {
    const files = Array.from(fileInput.files);
    await classifyFiles(files);
    fileInput.value = '';
  });

  async function classifyFiles(files) {
    const mobilenet = BirdnestTrain.getMobileNet();
    const model = BirdnestTrain.getModel();

    if (!mobilenet || !model) {
      alert('No trained model available. Please train a model in Step 2 first.');
      return;
    }

    for (const file of files) {
      const blob = await App.fileToBlob(file);
      const img = await App.loadImage(blob);
      const result = await classifySingle(mobilenet, model, img);
      addResultCard(blob, result);
    }
  }

  async function classifySingle(mobilenet, model, imgElement) {
    const preprocessed = BirdnestTrain.preprocessImage(imgElement);
    const features = mobilenet.predict(preprocessed);
    const prediction = model.predict(features);
    const score = (await prediction.data())[0];

    preprocessed.dispose();
    features.dispose();
    prediction.dispose();

    return {
      label: score > 0.5 ? 'Bird Nest' : 'Not Bird Nest',
      isBirdnest: score > 0.5,
      confidence: score > 0.5 ? score : 1 - score,
      rawScore: score
    };
  }

  function addResultCard(blob, result) {
    const card = document.createElement('div');
    card.className = 'result-card';

    const imgEl = document.createElement('img');
    imgEl.src = URL.createObjectURL(blob);

    const info = document.createElement('div');
    info.className = 'result-info';

    const labelClass = result.isBirdnest ? 'birdnest' : 'not-birdnest';
    const barColor = result.isBirdnest ? 'green' : 'red';
    const pct = (result.confidence * 100).toFixed(1);
    const nestPct = (result.rawScore * 100).toFixed(1);
    const notNestPct = ((1 - result.rawScore) * 100).toFixed(1);

    info.innerHTML = `
      <div class="result-label ${labelClass}">${result.label}</div>
      <div class="prediction-bar">
        <span class="prediction-bar-label">Nest</span>
        <div class="prediction-track">
          <div class="prediction-fill green" style="width:${nestPct}%"></div>
        </div>
        <span class="prediction-value">${nestPct}%</span>
      </div>
      <div class="prediction-bar">
        <span class="prediction-bar-label">Not Nest</span>
        <div class="prediction-track">
          <div class="prediction-fill red" style="width:${notNestPct}%"></div>
        </div>
        <span class="prediction-value">${notNestPct}%</span>
      </div>
      <button class="btn-add-to-dataset" data-category="${result.isBirdnest ? 'not_birdnest' : 'birdnest'}">
        Add to ${result.isBirdnest ? 'Not Bird Nest' : 'Bird Nest'} dataset
      </button>
    `;

    card.appendChild(imgEl);
    card.appendChild(info);
    resultsGrid.prepend(card);

    // Add-to-dataset button
    const addBtn = info.querySelector('.btn-add-to-dataset');
    addBtn.addEventListener('click', async () => {
      const category = addBtn.dataset.category;
      await App.addImage(category, blob);
      addBtn.textContent = 'Added!';
      addBtn.style.color = 'var(--green)';
      addBtn.style.borderColor = 'var(--green)';
      addBtn.disabled = true;
    });
  }
})();
