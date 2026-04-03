/**
 * Step 7 — The Commercial Eye: Google Cloud Vision API + Gemini
 *
 * Sends the image to a local proxy server which forwards it to Google Vision
 * and (on demand) to Gemini for multimodal narration.
 * Displays results side-by-side with local MobileNet predictions.
 */

(function () {
  const fileInput = document.getElementById('file-input-7');
  const uploadArea = document.getElementById('upload-area-7');
  const resultsDiv = document.getElementById('step7-results');
  const loadingDiv = document.getElementById('vision-loading');
  const errorDiv = document.getElementById('vision-error');
  const serverUrlInput = document.getElementById('vision-server-url');

  // Gemini elements
  const geminiPromptInput = document.getElementById('gemini-prompt-input');
  const geminiRunBtn = document.getElementById('gemini-run');
  const geminiPresets = document.querySelectorAll('.gemini-chip');
  const geminiLoading = document.getElementById('gemini-loading');
  const geminiResponse = document.getElementById('gemini-response');
  const geminiBudget = document.getElementById('gemini-budget');
  const geminiError = document.getElementById('gemini-error');

  // Keep a reference to the current file for Gemini re-queries
  let currentFile = null;

  function getServerUrl() {
    return (serverUrlInput.value || 'http://localhost:8000').replace(/\/+$/, '');
  }

  fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    currentFile = file;
    uploadArea.classList.add('has-file');
    resultsDiv.classList.add('hidden');
    loadingDiv.classList.remove('hidden');
    errorDiv.classList.add('hidden');

    // Reset Gemini state for new image
    geminiResponse.classList.add('hidden');
    geminiError.classList.add('hidden');
    geminiBudget.textContent = '';

    try {
      // Load image locally for display and MobileNet comparison
      const img = await App.loadImage(file);

      // Draw to canvas
      const canvas = document.getElementById('canvas-vision');
      App.drawToCanvas(canvas, img, 500);

      // Run local MobileNet in parallel with the API call
      const [visionResult, mobilenetResult] = await Promise.all([
        callVisionProxy(file),
        runLocalMobileNet(img),
      ]);

      loadingDiv.classList.add('hidden');
      resultsDiv.classList.remove('hidden');

      // Render all sections
      renderBudget(visionResult);
      renderLabelsComparison(visionResult.labels, mobilenetResult);
      renderObjects(canvas, img, visionResult.objects);
      renderFaces(visionResult.faces);
      renderSafeSearch(visionResult.safeSearch);
      renderColors(visionResult.dominantColors);
    } catch (err) {
      loadingDiv.classList.add('hidden');
      errorDiv.classList.remove('hidden');
      errorDiv.textContent = err.message || 'Failed to reach the Vision proxy server.';
    }
  });

  // ------------------------------------------------------------------
  // Gemini: preset chips fill the prompt input
  // ------------------------------------------------------------------
  geminiPresets.forEach(chip => {
    chip.addEventListener('click', () => {
      geminiPromptInput.value = chip.dataset.prompt;
    });
  });

  // ------------------------------------------------------------------
  // Gemini: run button
  // ------------------------------------------------------------------
  geminiRunBtn.addEventListener('click', () => runGemini());
  geminiPromptInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') runGemini();
  });

  async function runGemini() {
    if (!currentFile) return;
    const prompt = geminiPromptInput.value.trim();
    if (!prompt) {
      geminiPromptInput.value = 'Describe this image in detail.';
    }

    geminiLoading.classList.remove('hidden');
    geminiResponse.classList.add('hidden');
    geminiError.classList.add('hidden');

    try {
      const result = await callGeminiProxy(currentFile, geminiPromptInput.value.trim());
      geminiLoading.classList.add('hidden');
      geminiResponse.textContent = result.text;
      geminiResponse.classList.remove('hidden');
      geminiBudget.textContent = `Gemini requests: ${result.requestsUsed} / ${result.requestsUsed + result.requestsRemaining} used`;
    } catch (err) {
      geminiLoading.classList.add('hidden');
      geminiError.classList.remove('hidden');
      geminiError.textContent = err.message || 'Failed to reach Gemini proxy.';
    }
  }

  // ------------------------------------------------------------------
  // API call to proxy
  // ------------------------------------------------------------------
  async function callVisionProxy(file) {
    const formData = new FormData();
    formData.append('file', file);

    const resp = await fetch(`${getServerUrl()}/vision`, {
      method: 'POST',
      body: formData,
    });

    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `Server error (${resp.status})`);
    }

    return resp.json();
  }

  // ------------------------------------------------------------------
  // API call to Gemini proxy
  // ------------------------------------------------------------------
  async function callGeminiProxy(file, prompt) {
    const formData = new FormData();
    formData.append('file', file);

    const url = new URL(`${getServerUrl()}/gemini`);
    url.searchParams.set('prompt', prompt);

    const resp = await fetch(url.toString(), {
      method: 'POST',
      body: formData,
    });

    if (!resp.ok) {
      const body = await resp.json().catch(() => ({}));
      throw new Error(body.detail || `Gemini server error (${resp.status})`);
    }

    return resp.json();
  }

  // ------------------------------------------------------------------
  // Local MobileNet for comparison
  // ------------------------------------------------------------------
  async function runLocalMobileNet(img) {
    if (!App.models.mobilenet) return [];
    try {
      return await App.models.mobilenet.classify(img, 10);
    } catch {
      return [];
    }
  }

  // ------------------------------------------------------------------
  // Renderers
  // ------------------------------------------------------------------

  function renderBudget(data) {
    const el = document.getElementById('vision-budget');
    const used = data.requestsUsed || 0;
    const remaining = data.requestsRemaining || 0;
    const total = used + remaining;
    el.textContent = `API requests: ${used} / ${total} used`;
  }

  function renderLabelsComparison(googleLabels, mobilenetPreds) {
    // Google labels
    const googleContainer = document.getElementById('vision-labels');
    googleContainer.innerHTML = googleLabels.map(l => {
      const pct = (l.score * 100).toFixed(1);
      return `
        <div class="prediction-bar">
          <span class="prediction-label">${l.name}</span>
          <div class="prediction-track">
            <div class="prediction-fill" style="width:${pct}%; background:var(--yellow)"></div>
          </div>
          <span class="prediction-value">${pct}%</span>
        </div>
      `;
    }).join('');

    // MobileNet labels
    const mobileContainer = document.getElementById('vision-mobilenet-labels');
    if (!mobilenetPreds || mobilenetPreds.length === 0) {
      mobileContainer.innerHTML = '<p style="color:var(--text-dim); font-size:0.8rem;">MobileNet not loaded</p>';
      return;
    }
    mobileContainer.innerHTML = mobilenetPreds.map(p => {
      const pct = (p.probability * 100).toFixed(1);
      return `
        <div class="prediction-bar">
          <span class="prediction-label">${p.className.split(',')[0]}</span>
          <div class="prediction-track">
            <div class="prediction-fill" style="width:${pct}%"></div>
          </div>
          <span class="prediction-value">${pct}%</span>
        </div>
      `;
    }).join('');
  }

  function renderObjects(sourceCanvas, img, objects) {
    const canvas = document.getElementById('canvas-vision-objects');
    const ctx = App.drawToCanvas(canvas, img, 500);
    const scaleX = canvas.width / img.naturalWidth;
    const scaleY = canvas.height / img.naturalHeight;

    objects.forEach((obj, i) => {
      const verts = (obj.boundingPoly && obj.boundingPoly.normalizedVertices) || [];
      if (verts.length < 2) return;

      const x = verts[0].x * canvas.width;
      const y = verts[0].y * canvas.height;
      const w = (verts[2].x - verts[0].x) * canvas.width;
      const h = (verts[2].y - verts[0].y) * canvas.height;

      ctx.strokeStyle = '#ffd94a';
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);

      const label = `${obj.name} ${(obj.score * 100).toFixed(0)}%`;
      ctx.font = `bold ${Math.max(12, canvas.width * 0.025)}px monospace`;
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      ctx.fillRect(x, y - 18, textWidth + 8, 20);
      ctx.fillStyle = '#ffd94a';
      ctx.fillText(label, x + 4, y - 3);
    });
  }

  function renderFaces(faces) {
    const section = document.getElementById('vision-faces-section');
    const container = document.getElementById('vision-faces');

    if (!faces || faces.length === 0) {
      section.classList.add('hidden');
      return;
    }

    section.classList.remove('hidden');

    const likelihoodMap = {
      'VERY_UNLIKELY': 0,
      'UNLIKELY': 25,
      'POSSIBLE': 50,
      'LIKELY': 75,
      'VERY_LIKELY': 100,
    };

    container.innerHTML = faces.map((f, i) => {
      const emotions = [
        { name: 'Joy', value: f.joy },
        { name: 'Sorrow', value: f.sorrow },
        { name: 'Anger', value: f.anger },
        { name: 'Surprise', value: f.surprise },
      ];

      const bars = emotions.map(e => {
        const pct = likelihoodMap[e.value] ?? 0;
        const colorClass = pct >= 75 ? 'var(--green)' : pct >= 50 ? 'var(--yellow)' : 'var(--text-dim)';
        return `
          <div class="prediction-bar">
            <span class="prediction-label">${e.name}</span>
            <div class="prediction-track">
              <div class="prediction-fill" style="width:${pct}%; background:${colorClass}"></div>
            </div>
            <span class="prediction-value">${e.value.replace('_', ' ').toLowerCase()}</span>
          </div>
        `;
      }).join('');

      return `
        <div style="background:var(--bg-card); padding:16px; border-radius:var(--radius); margin-bottom:12px;">
          <div class="mono" style="font-size:0.8rem; color:var(--green); margin-bottom:8px;">
            Face ${i + 1} — Confidence: ${(f.confidence * 100).toFixed(0)}%
            ${f.headwear !== 'VERY_UNLIKELY' ? ' — Headwear: ' + f.headwear.replace('_', ' ').toLowerCase() : ''}
          </div>
          ${bars}
        </div>
      `;
    }).join('');
  }

  function renderSafeSearch(ss) {
    const container = document.getElementById('vision-safesearch');
    if (!ss || Object.keys(ss).length === 0) {
      container.innerHTML = '<p style="color:var(--text-dim); font-size:0.8rem;">No safe search data</p>';
      return;
    }

    const likelihoodMap = {
      'VERY_UNLIKELY': 0,
      'UNLIKELY': 25,
      'POSSIBLE': 50,
      'LIKELY': 75,
      'VERY_LIKELY': 100,
    };

    const categories = [
      { key: 'adult', label: 'Adult' },
      { key: 'spoof', label: 'Spoof / Meme' },
      { key: 'medical', label: 'Medical' },
      { key: 'violence', label: 'Violence' },
      { key: 'racy', label: 'Racy' },
    ];

    container.innerHTML = `<div style="background:var(--bg-card); padding:16px; border-radius:var(--radius); margin-bottom:24px;">` +
      categories.map(c => {
        const val = ss[c.key] || 'UNKNOWN';
        const pct = likelihoodMap[val] ?? 0;
        const color = pct >= 75 ? 'var(--red)' : pct >= 50 ? 'var(--yellow)' : 'var(--text-dim)';
        return `
          <div class="prediction-bar">
            <span class="prediction-label">${c.label}</span>
            <div class="prediction-track">
              <div class="prediction-fill" style="width:${pct}%; background:${color}"></div>
            </div>
            <span class="prediction-value">${val.replace(/_/g, ' ').toLowerCase()}</span>
          </div>
        `;
      }).join('') +
      `</div>`;
  }

  function renderColors(colors) {
    const container = document.getElementById('vision-colors');
    if (!colors || colors.length === 0) {
      container.innerHTML = '<p style="color:var(--text-dim); font-size:0.8rem;">No color data</p>';
      return;
    }

    container.innerHTML = colors.map(c => {
      const rgb = `rgb(${c.r}, ${c.g}, ${c.b})`;
      const pct = (c.pixelFraction * 100).toFixed(1);
      return `
        <div style="text-align:center;">
          <div style="
            width:60px; height:60px; border-radius:var(--radius);
            background:${rgb}; border:1px solid #333;
          "></div>
          <div class="mono" style="font-size:0.65rem; color:var(--text-dim); margin-top:4px;">
            ${c.r},${c.g},${c.b}<br>${pct}%
          </div>
        </div>
      `;
    }).join('');
  }
})();
