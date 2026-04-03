/**
 * step2-machine-speaks.js — Image captioning + zero-shot classification via Transformers.js
 * Models: Xenova/vit-gpt2-image-captioning (captioning), Xenova/clip-vit-base-patch32 (zero-shot)
 * Must be loaded as <script type="module">
 */

import { pipeline } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

const fileInput = document.getElementById('file-input-2');
const uploadArea = document.getElementById('upload-area-2');
const results = document.getElementById('step2-results');
const canvas = document.getElementById('canvas-speaks');
const captionOutput = document.getElementById('caption-output');
const zeroshotLabels = document.getElementById('zeroshot-labels');
const zeroshotRunBtn = document.getElementById('zeroshot-run');
const zeroshotResults = document.getElementById('zeroshot-results');
const loadingBar = document.getElementById('speaks-loading-bar');
const loadingFill = document.getElementById('speaks-loading-fill');
const loadingPercent = document.getElementById('speaks-loading-percent');
const loadingSteps = document.getElementById('speaks-loading-steps');

let captioner = null;
let classifier = null;
let currentBlobUrl = null;

// ── Loading bar ──
function showLoading(stepText, percent) {
  loadingBar.classList.add('visible');
  loadingFill.style.width = percent + '%';
  loadingPercent.textContent = Math.round(percent) + '%';
  loadingSteps.textContent = stepText;
}

function hideLoading() {
  loadingFill.style.width = '100%';
  loadingPercent.textContent = '100%';
  loadingSteps.textContent = 'Models ready';
  setTimeout(() => { loadingBar.classList.remove('visible'); }, 600);
}

// ── Progress callback for model downloads ──
function makeProgressCallback(label) {
  return (progress) => {
    if (progress.status === 'progress' && progress.progress != null) {
      const file = progress.file ? progress.file.split('/').pop() : '';
      showLoading(`${label}: ${file}`, progress.progress);
    } else if (progress.status === 'done') {
      showLoading(`${label}: complete`, 100);
    } else if (progress.status === 'initiate') {
      showLoading(`${label}: downloading...`, 0);
    }
  };
}

// ── Lazy-load models ──
async function ensureCaptioner() {
  if (captioner) return captioner;
  showLoading('Loading captioning model...', 0);
  captioner = await pipeline('image-to-text', 'Xenova/vit-gpt2-image-captioning', {
    progress_callback: makeProgressCallback('Captioning model'),
  });
  return captioner;
}

async function ensureClassifier() {
  if (classifier) return classifier;
  showLoading('Loading classification model...', 0);
  classifier = await pipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32', {
    progress_callback: makeProgressCallback('CLIP model'),
  });
  return classifier;
}

// ── Typing animation ──
function typeText(element, text, speed = 30) {
  element.textContent = '';
  let i = 0;
  return new Promise((resolve) => {
    function tick() {
      if (i < text.length) {
        element.textContent += text[i];
        i++;
        setTimeout(tick, speed);
      } else {
        resolve();
      }
    }
    tick();
  });
}

// ── Image upload ──
fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  uploadArea.classList.add('has-file');
  results.classList.remove('hidden');
  captionOutput.textContent = 'Loading model...';
  zeroshotResults.innerHTML = '';

  // Draw to canvas
  const img = await App.loadImage(file);
  App.drawToCanvas(canvas, img, 500);

  // Create blob URL for Transformers.js
  if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
  currentBlobUrl = URL.createObjectURL(file);

  // Run captioning
  try {
    const cap = await ensureCaptioner();
    hideLoading();
    captionOutput.textContent = '';
    const result = await cap(currentBlobUrl);
    const caption = result[0].generated_text || result[0].text || JSON.stringify(result);
    await typeText(captionOutput, caption);
  } catch (err) {
    console.error('Captioning error:', err);
    captionOutput.textContent = 'Captioning failed: ' + err.message;
    hideLoading();
  }
});

// ── Zero-shot classification ──
zeroshotRunBtn.addEventListener('click', async () => {
  const raw = zeroshotLabels.value.trim();
  if (!raw) return;
  if (!currentBlobUrl) {
    zeroshotResults.innerHTML = '<p style="color:var(--red)">Upload an image first.</p>';
    return;
  }

  const labels = raw.split(',').map(s => s.trim()).filter(Boolean);
  if (labels.length < 2) {
    zeroshotResults.innerHTML = '<p style="color:var(--yellow)">Enter at least 2 labels separated by commas.</p>';
    return;
  }

  zeroshotResults.innerHTML = '<p style="color:var(--text-dim)">Classifying...</p>';

  try {
    const cls = await ensureClassifier();
    hideLoading();
    const result = await cls(currentBlobUrl, labels);

    // Render as bar chart
    zeroshotResults.innerHTML = result.map(item => {
      const pct = (item.score * 100).toFixed(1);
      return `
        <div class="prediction-bar">
          <span class="prediction-label">${item.label}</span>
          <div class="prediction-track">
            <div class="prediction-fill" style="width:${pct}%"></div>
          </div>
          <span class="prediction-value">${pct}%</span>
        </div>
      `;
    }).join('');
  } catch (err) {
    console.error('Zero-shot error:', err);
    zeroshotResults.innerHTML = `<p style="color:var(--red)">Classification failed: ${err.message}</p>`;
    hideLoading();
  }
});

// ── Preset chips ──
document.querySelectorAll('.preset-chip').forEach(chip => {
  chip.addEventListener('click', () => {
    zeroshotLabels.value = chip.dataset.labels;
  });
});
