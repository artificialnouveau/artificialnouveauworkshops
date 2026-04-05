/**
 * step2-machine-speaks.js — Caption comparison: 4 models on the same image
 * Models: ViT-GPT2 (captioning), COCO-SSD (object detection), Florence-2 (vision), Moondream 2 (VLM)
 * Plus: CLIP zero-shot classification
 * Must be loaded as <script type="module">
 */

import {
  pipeline,
  Florence2ForConditionalGeneration,
  AutoProcessor,
  AutoTokenizer,
  Moondream1ForConditionalGeneration,
  RawImage,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

const fileInput = document.getElementById('file-input-2');
const uploadArea = document.getElementById('upload-area-2');
const results = document.getElementById('step2-results');
const canvas = document.getElementById('canvas-speaks');
const zeroshotLabels = document.getElementById('zeroshot-labels');
const zeroshotRunBtn = document.getElementById('zeroshot-run');
const zeroshotResults = document.getElementById('zeroshot-results');
const loadingBar = document.getElementById('speaks-loading-bar');
const loadingFill = document.getElementById('speaks-loading-fill');
const loadingPercent = document.getElementById('speaks-loading-percent');
const loadingSteps = document.getElementById('speaks-loading-steps');

// Caption output elements
const captionVitgpt2 = document.getElementById('caption-vitgpt2');
const captionCocossd = document.getElementById('caption-cocossd');
const captionFlorence = document.getElementById('caption-florence');
const captionMoondream = document.getElementById('caption-moondream');

let captioner = null;
let classifier = null;
let florenceModel = null;
let florenceProcessor = null;
let moondreamModel = null;
let moondreamProcessor = null;
let moondreamTokenizer = null;
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

// ── Lazy-load models ──
async function ensureCaptioner() {
  if (captioner) return captioner;
  showLoading('Loading ViT-GPT2...', 0);
  captioner = await pipeline('image-to-text', 'Xenova/vit-gpt2-image-captioning', {
    progress_callback: makeProgressCallback('ViT-GPT2'),
  });
  return captioner;
}

async function ensureClassifier() {
  if (classifier) return classifier;
  showLoading('Loading CLIP...', 0);
  classifier = await pipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32', {
    progress_callback: makeProgressCallback('CLIP'),
  });
  return classifier;
}

async function ensureFlorence() {
  if (florenceModel) return { model: florenceModel, processor: florenceProcessor };
  showLoading('Loading Florence-2...', 0);
  florenceModel = await Florence2ForConditionalGeneration.from_pretrained(
    'onnx-community/Florence-2-base-ft', {
      dtype: 'fp32',
      progress_callback: makeProgressCallback('Florence-2'),
    }
  );
  florenceProcessor = await AutoProcessor.from_pretrained('onnx-community/Florence-2-base-ft');
  return { model: florenceModel, processor: florenceProcessor };
}

async function ensureMoondream() {
  if (moondreamModel) return { model: moondreamModel, processor: moondreamProcessor, tokenizer: moondreamTokenizer };
  showLoading('Loading Moondream 2...', 0);
  moondreamModel = await Moondream1ForConditionalGeneration.from_pretrained(
    'Xenova/moondream2', {
      dtype: {
        embed_tokens: 'fp16',
        vision_encoder: 'q8',
        decoder_model_merged: 'q4',
      },
      device: 'webgpu',
      progress_callback: makeProgressCallback('Moondream 2'),
    }
  );
  moondreamProcessor = await AutoProcessor.from_pretrained('Xenova/moondream2');
  moondreamTokenizer = await AutoTokenizer.from_pretrained('Xenova/moondream2');
  return { model: moondreamModel, processor: moondreamProcessor, tokenizer: moondreamTokenizer };
}

// ── Run individual models ──
async function runVitGpt2(blobUrl, element) {
  element.textContent = 'Loading model...';
  element.style.color = 'var(--dim)';
  try {
    const cap = await ensureCaptioner();
    element.textContent = '';
    element.style.color = 'var(--text)';
    const result = await cap(blobUrl);
    const caption = result[0].generated_text || result[0].text || JSON.stringify(result);
    await typeText(element, caption);
  } catch (err) {
    console.error('ViT-GPT2 error:', err);
    element.textContent = 'Failed: ' + err.message;
    element.style.color = 'var(--red)';
  }
}

async function runCocoSsd(imgElement, element) {
  element.textContent = 'Detecting objects...';
  element.style.color = 'var(--dim)';
  try {
    // COCO-SSD is loaded globally via TF.js script tag
    const model = await cocoSsd.load();
    const predictions = await model.detect(imgElement);
    if (predictions.length === 0) {
      element.textContent = 'No objects detected.';
      element.style.color = 'var(--dim)';
    } else {
      const items = predictions.map(p => `${p.class} (${(p.score * 100).toFixed(0)}%)`);
      element.style.color = 'var(--text)';
      await typeText(element, items.join(', '));
    }
  } catch (err) {
    console.error('COCO-SSD error:', err);
    element.textContent = 'Failed: ' + err.message;
    element.style.color = 'var(--red)';
  }
}

async function runFlorence(blobUrl, element) {
  element.textContent = 'Loading model...';
  element.style.color = 'var(--dim)';
  try {
    const { model, processor } = await ensureFlorence();
    element.textContent = 'Generating caption...';
    const image = await RawImage.fromURL(blobUrl);
    const prompt = '<MORE_DETAILED_CAPTION>';
    const inputs = await processor(image, prompt);
    const generatedIds = await model.generate({
      ...inputs,
      max_new_tokens: 100,
    });
    const generatedText = processor.batch_decode(generatedIds, { skip_special_tokens: false })[0];
    // Florence wraps output in task tokens — extract the caption
    const match = generatedText.match(/<MORE_DETAILED_CAPTION>(.*?)(<\/s>|$)/);
    const caption = match ? match[1].trim() : generatedText.replace(/<[^>]+>/g, '').trim();
    element.style.color = 'var(--text)';
    await typeText(element, caption);
  } catch (err) {
    console.error('Florence-2 error:', err);
    element.textContent = 'Failed: ' + err.message;
    element.style.color = 'var(--red)';
  }
}

async function runMoondream(blobUrl, element) {
  element.textContent = 'Loading model...';
  element.style.color = 'var(--dim)';
  try {
    const { model, processor, tokenizer } = await ensureMoondream();
    element.textContent = 'Generating caption...';
    const prompt = 'Describe this image.';
    const text = `<image>\n\nQuestion: ${prompt}\n\nAnswer:`;
    const textInputs = tokenizer(text);
    const image = await RawImage.fromURL(blobUrl);
    const visionInputs = await processor(image);
    const output = await model.generate({
      ...textInputs,
      ...visionInputs,
      do_sample: false,
      max_new_tokens: 100,
    });
    const decoded = tokenizer.batch_decode(output, { skip_special_tokens: true });
    // Extract answer after "Answer:" if present
    let caption = decoded[0] || '';
    const answerIdx = caption.lastIndexOf('Answer:');
    if (answerIdx !== -1) caption = caption.substring(answerIdx + 7).trim();
    element.style.color = 'var(--text)';
    await typeText(element, caption);
  } catch (err) {
    console.error('Moondream 2 error:', err);
    element.textContent = 'Failed: ' + err.message;
    element.style.color = 'var(--red)';
  }
}

// ── Image upload — run all models in parallel ──
fileInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  uploadArea.classList.add('has-file');
  results.classList.remove('hidden');
  zeroshotResults.innerHTML = '';

  // Reset all caption outputs
  [captionVitgpt2, captionCocossd, captionFlorence, captionMoondream].forEach(el => {
    el.textContent = 'Processing...';
    el.style.color = 'var(--dim)';
  });

  // Draw to canvas
  const img = await App.loadImage(file);
  App.drawToCanvas(canvas, img, 500);

  // Create blob URL for Transformers.js
  if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
  currentBlobUrl = URL.createObjectURL(file);

  hideLoading();

  // Run all 4 models in parallel
  runVitGpt2(currentBlobUrl, captionVitgpt2);
  runCocoSsd(img, captionCocossd);
  runFlorence(currentBlobUrl, captionFlorence);
  runMoondream(currentBlobUrl, captionMoondream);
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
