/**
 * step2-machine-speaks.js — Caption comparison: 5 models on the same image
 * Models: ViT-GPT2, COCO-SSD, Florence-2, Moondream 2, Gemma 4 E2B
 * Plus: CLIP zero-shot classification
 * Must be loaded as <script type="module">
 */

const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1';

// We import pipeline normally; Florence-2 and Moondream use dynamic imports
// because they need specific model classes not supported by pipeline()
const { pipeline, RawImage } = await import(TRANSFORMERS_URL);

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
const captionGemma4 = document.getElementById('caption-gemma4');

let captioner = null;
let classifier = null;
let florenceModel = null;
let florenceProcessor = null;
let gemma4Model = null;
let gemma4Processor = null;
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

// ── Per-card progress bar ──
function showCardProgress(progressEl, percent) {
  if (!progressEl) return;
  if (!progressEl.querySelector('.fill')) {
    progressEl.innerHTML = '<div class="fill"></div>';
  }
  progressEl.classList.add('active');
  progressEl.querySelector('.fill').style.width = percent + '%';
}

function hideCardProgress(progressEl) {
  if (!progressEl) return;
  progressEl.classList.remove('active');
}

function makeCardProgressCallback(label, progressEl, statusEl) {
  return (progress) => {
    if (progress.status === 'progress' && progress.progress != null) {
      showCardProgress(progressEl, progress.progress);
      const file = progress.file ? progress.file.split('/').pop() : '';
      setStatus(statusEl, `Downloading ${file}... ${Math.round(progress.progress)}%`, 'var(--dim)');
    } else if (progress.status === 'done') {
      showCardProgress(progressEl, 100);
    } else if (progress.status === 'initiate') {
      showCardProgress(progressEl, 0);
      setStatus(statusEl, `Downloading ${label}...`, 'var(--dim)');
    }
  };
}

// ── Typing animation ──
function typeText(element, text, speed = 25) {
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

// ── Status helpers ──
function setStatus(el, text, color) {
  el.textContent = text;
  el.style.color = color || 'var(--dim)';
}

// ── Run ViT-GPT2 ──
async function runVitGpt2(blobUrl, element) {
  setStatus(element, 'Loading ViT-GPT2...', 'var(--dim)');
  try {
    if (!captioner) {
      captioner = await pipeline('image-to-text', 'Xenova/vit-gpt2-image-captioning', {
        progress_callback: makeProgressCallback('ViT-GPT2'),
      });
    }
    setStatus(element, 'Generating caption...', 'var(--dim)');
    const result = await captioner(blobUrl);
    const caption = result[0].generated_text || result[0].text || JSON.stringify(result);
    element.style.color = 'var(--text)';
    await typeText(element, caption);
  } catch (err) {
    console.error('ViT-GPT2 error:', err);
    setStatus(element, 'Error: ' + err.message, 'var(--red)');
  }
}

// ── Run COCO-SSD ──
async function runCocoSsd(imgElement, element) {
  setStatus(element, 'Loading COCO-SSD...', 'var(--dim)');
  try {
    if (typeof window.cocoSsd === 'undefined') {
      setStatus(element, 'COCO-SSD not available (script not loaded)', 'var(--red)');
      return;
    }
    const model = await window.cocoSsd.load();
    setStatus(element, 'Detecting objects...', 'var(--dim)');
    const predictions = await model.detect(imgElement);
    if (predictions.length === 0) {
      setStatus(element, 'No objects detected.', 'var(--dim)');
    } else {
      const items = predictions.map(p => `${p.class} (${(p.score * 100).toFixed(0)}%)`);
      element.style.color = 'var(--text)';
      await typeText(element, items.join(', '));
    }
  } catch (err) {
    console.error('COCO-SSD error:', err);
    setStatus(element, 'Error: ' + err.message, 'var(--red)');
  }
}

// ── Run Florence-2 ──
async function runFlorence(blobUrl, element) {
  const progressEl = document.getElementById('progress-florence');
  setStatus(element, 'Loading Florence-2...', 'var(--dim)');
  try {
    if (!florenceModel) {
      const {
        Florence2ForConditionalGeneration,
        AutoProcessor,
      } = await import(TRANSFORMERS_URL);

      florenceModel = await Florence2ForConditionalGeneration.from_pretrained(
        'onnx-community/Florence-2-base-ft', {
          dtype: 'fp32',
          progress_callback: makeCardProgressCallback('Florence-2', progressEl, element),
        }
      );
      florenceProcessor = await AutoProcessor.from_pretrained(
        'onnx-community/Florence-2-base-ft'
      );
      hideCardProgress(progressEl);
    }
    setStatus(element, 'Generating caption...', 'var(--dim)');

    const image = await RawImage.fromURL(blobUrl);
    const prompt = '<MORE_DETAILED_CAPTION>';
    const inputs = await florenceProcessor(image, prompt);
    const generatedIds = await florenceModel.generate({
      ...inputs,
      max_new_tokens: 100,
    });
    const generatedText = florenceProcessor.batch_decode(generatedIds, { skip_special_tokens: false })[0];

    // Florence wraps output in task tokens — extract the caption
    const match = generatedText.match(/<MORE_DETAILED_CAPTION>(.*?)(<\/s>|$)/s);
    const caption = match ? match[1].trim() : generatedText.replace(/<[^>]+>/g, '').trim();

    element.style.color = 'var(--text)';
    await typeText(element, caption);
  } catch (err) {
    console.error('Florence-2 error:', err);
    setStatus(element, 'Error: ' + err.message, 'var(--red)');
  }
}

// ── Run Moondream 2 ──
async function runMoondream(blobUrl, element) {
  const progressEl = document.getElementById('progress-moondream');
  setStatus(element, 'Loading Moondream 2...', 'var(--dim)');
  try {
    if (!moondreamModel) {
      const {
        Moondream1ForConditionalGeneration,
        AutoProcessor,
        AutoTokenizer,
      } = await import(TRANSFORMERS_URL);

      moondreamTokenizer = await AutoTokenizer.from_pretrained('Xenova/moondream2');
      moondreamProcessor = await AutoProcessor.from_pretrained('Xenova/moondream2');
      moondreamModel = await Moondream1ForConditionalGeneration.from_pretrained(
        'Xenova/moondream2', {
          dtype: {
            embed_tokens: 'fp16',
            vision_encoder: 'q8',
            decoder_model_merged: 'q4',
          },
          device: 'webgpu',
          progress_callback: makeCardProgressCallback('Moondream 2', progressEl, element),
        }
      );
      hideCardProgress(progressEl);
    }
    setStatus(element, 'Generating caption...', 'var(--dim)');

    const prompt = 'Describe this image.';
    const text = `<image>\n\nQuestion: ${prompt}\n\nAnswer:`;
    const textInputs = moondreamTokenizer(text);

    // Resize image to 378x378 (Moondream's expected input size)
    const image = await RawImage.fromURL(blobUrl);
    const resized = await image.resize(378, 378);
    const visionInputs = await moondreamProcessor(resized);

    const output = await moondreamModel.generate({
      ...textInputs,
      ...visionInputs,
      do_sample: false,
      max_new_tokens: 100,
    });
    const decoded = moondreamTokenizer.batch_decode(output, { skip_special_tokens: true });

    // Extract answer after "Answer:" if present
    let caption = decoded[0] || '';
    const answerIdx = caption.lastIndexOf('Answer:');
    if (answerIdx !== -1) caption = caption.substring(answerIdx + 7).trim();

    element.style.color = 'var(--text)';
    await typeText(element, caption);
  } catch (err) {
    console.error('Moondream 2 error:', err);
    hideCardProgress(progressEl);
    if (err.message && err.message.includes('webgpu')) {
      setStatus(element, 'WebGPU not supported in this browser. Try Chrome.', 'var(--red)');
    } else {
      setStatus(element, 'Error: ' + err.message, 'var(--red)');
    }
  }
}

// ── Run Gemma 4 ──
async function runGemma4(blobUrl, element) {
  const progressEl = document.getElementById('progress-gemma4');
  setStatus(element, 'Loading Gemma 4 (large model, may take several minutes)...', 'var(--dim)');
  try {
    if (!gemma4Model) {
      const {
        Gemma4ForConditionalGeneration,
        AutoProcessor,
      } = await import(TRANSFORMERS_URL);

      gemma4Processor = await AutoProcessor.from_pretrained(
        'onnx-community/gemma-4-E2B-it-ONNX'
      );
      gemma4Model = await Gemma4ForConditionalGeneration.from_pretrained(
        'onnx-community/gemma-4-E2B-it-ONNX', {
          dtype: 'q4f16',
          device: 'webgpu',
          progress_callback: makeCardProgressCallback('Gemma 4', progressEl, element),
        }
      );
      hideCardProgress(progressEl);
    }
    setStatus(element, 'Generating caption...', 'var(--dim)');

    const image = await RawImage.fromURL(blobUrl);
    const messages = [
      {
        role: 'user',
        content: [
          { type: 'image' },
          { type: 'text', text: 'Describe this image in one or two sentences.' },
        ],
      },
    ];
    const prompt = gemma4Processor.apply_chat_template(messages, {
      enable_thinking: false,
      add_generation_prompt: true,
    });
    const inputs = await gemma4Processor(prompt, image, null, {
      add_special_tokens: false,
    });
    const output = await gemma4Model.generate({
      ...inputs,
      max_new_tokens: 100,
      do_sample: false,
    });
    // Decode only the new tokens (skip the prompt)
    const promptLength = inputs.input_ids.dims[1];
    const newTokens = output.slice(null, [promptLength, null]);
    const decoded = gemma4Processor.batch_decode(newTokens, { skip_special_tokens: true });
    const caption = (decoded[0] || '').trim();

    element.style.color = 'var(--text)';
    await typeText(element, caption);
  } catch (err) {
    console.error('Gemma 4 error:', err);
    hideCardProgress(progressEl);
    if (err.message && err.message.includes('webgpu')) {
      setStatus(element, 'WebGPU not supported in this browser. Try Chrome.', 'var(--red)');
    } else {
      setStatus(element, 'Error: ' + err.message, 'var(--red)');
    }
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
  setStatus(captionVitgpt2, 'Starting...', 'var(--dim)');
  setStatus(captionCocossd, 'Starting...', 'var(--dim)');
  setStatus(captionFlorence, 'Starting...', 'var(--dim)');
  setStatus(captionMoondream, 'Starting...', 'var(--dim)');
  setStatus(captionGemma4, 'Starting...', 'var(--dim)');

  // Draw to canvas
  const img = await App.loadImage(file);
  App.drawToCanvas(canvas, img, 500);

  // Create blob URL for Transformers.js
  if (currentBlobUrl) URL.revokeObjectURL(currentBlobUrl);
  currentBlobUrl = URL.createObjectURL(file);

  hideLoading();

  // Run all 5 models in parallel — each handles its own errors
  runVitGpt2(currentBlobUrl, captionVitgpt2);
  runCocoSsd(img, captionCocossd);
  runFlorence(currentBlobUrl, captionFlorence);
  runMoondream(currentBlobUrl, captionMoondream);
  runGemma4(currentBlobUrl, captionGemma4);
});

// ── Zero-shot classification (CLIP) ──
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

  zeroshotResults.innerHTML = '<p style="color:var(--dim)">Classifying...</p>';

  try {
    if (!classifier) {
      classifier = await pipeline('zero-shot-image-classification', 'Xenova/clip-vit-base-patch32', {
        progress_callback: makeProgressCallback('CLIP'),
      });
    }
    hideLoading();
    const result = await classifier(currentBlobUrl, labels);

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
