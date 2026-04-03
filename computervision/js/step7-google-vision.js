/**
 * Step 7 — The Commercial Eye: Google Cloud Vision API + Gemini
 *
 * Sends the image to a local proxy server which forwards it to Google Vision
 * and (on demand) to Gemini for multimodal narration.
 * Displays results side-by-side with local MobileNet predictions.
 */

(function () {
  // Default to Cloudflare Worker proxy; override via ?server=1 to show config field
  const DEFAULT_SERVER = 'https://artificialnouveauworkshops.artificialnouveau.workers.dev';

  const fileInput = document.getElementById('file-input-7');
  const uploadArea = document.getElementById('upload-area-7');
  const resultsDiv = document.getElementById('step7-results');
  const loadingDiv = document.getElementById('vision-loading');
  const errorDiv = document.getElementById('vision-error');
  const serverUrlInput = document.getElementById('vision-server-url');
  const serverConfig = document.getElementById('vision-server-config');

  // Show server config field only if ?server=1 is in the URL (for presenter)
  if (new URLSearchParams(window.location.search).get('server') === '1') {
    serverConfig.classList.remove('hidden');
  }

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
    const val = serverUrlInput.value.trim();
    // Use the input value only if the config is visible and has been changed
    return (val && val !== 'http://localhost:8000' ? val : DEFAULT_SERVER).replace(/\/+$/, '');
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

      // Call Google Vision API
      const visionResult = await callVisionProxy(file);

      loadingDiv.classList.add('hidden');
      resultsDiv.classList.remove('hidden');

      // Render all sections
      renderLabels(visionResult.labels);
      renderFaces(visionResult.faces);
      renderLandmarks(visionResult.landmarks);
      renderLogos(visionResult.logos);
      renderSafeSearch(visionResult.safeSearch);
      renderText(visionResult.fullText);
      renderColors(visionResult.dominantColors);
      renderWeb(visionResult);
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
      // Handle both raw Gemini response and pre-parsed proxy response
      const text = result.text || (result.candidates && result.candidates[0]?.content?.parts?.[0]?.text) || '(No response from Gemini)';
      geminiResponse.textContent = text;
      geminiResponse.classList.remove('hidden');
      if (result.requestsUsed != null) {
        geminiBudget.textContent = `Gemini requests: ${result.requestsUsed} / ${result.requestsUsed + result.requestsRemaining} used`;
      }
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
      throw new Error(body.detail || body.error?.message || `Server error (${resp.status})`);
    }

    const data = await resp.json();

    // Handle raw Google Vision response (Cloudflare Worker passes it through)
    const r = (data.responses && data.responses[0]) || data;

    // If already parsed by FastAPI proxy, return as-is
    if (data.labels) return data;

    // Parse raw Google response
    const labels = (r.labelAnnotations || []).map(l => ({
      name: l.description, score: l.score,
    }));

    const faces = (r.faceAnnotations || []).map(f => ({
      joy: f.joyLikelihood || 'UNKNOWN',
      sorrow: f.sorrowLikelihood || 'UNKNOWN',
      anger: f.angerLikelihood || 'UNKNOWN',
      surprise: f.surpriseLikelihood || 'UNKNOWN',
      headwear: f.headwearLikelihood || 'UNKNOWN',
      confidence: f.detectionConfidence || 0,
      boundingPoly: f.boundingPoly || {},
    }));

    const objects = (r.localizedObjectAnnotations || []).map(o => ({
      name: o.name, score: o.score,
      boundingPoly: o.boundingPoly || {},
    }));

    const safeSearch = r.safeSearchAnnotation || {};

    const colors = [];
    const props = r.imagePropertiesAnnotation || {};
    for (const c of (props.dominantColors?.colors || []).slice(0, 5)) {
      const rgb = c.color || {};
      colors.push({
        r: Math.round(rgb.red || 0), g: Math.round(rgb.green || 0), b: Math.round(rgb.blue || 0),
        score: c.score || 0, pixelFraction: c.pixelFraction || 0,
      });
    }

    // Parse text detection (OCR)
    const textAnnotations = r.textAnnotations || [];
    const fullText = textAnnotations.length > 0 ? textAnnotations[0].description : '';

    // Parse landmark detection
    const landmarks = (r.landmarkAnnotations || []).map(l => ({
      name: l.description,
      score: l.score || 0,
      locations: (l.locations || []).map(loc => loc.latLng || {}),
    }));

    // Parse logo detection
    const logos = (r.logoAnnotations || []).map(l => ({
      name: l.description,
      score: l.score || 0,
    }));

    // Parse web detection
    const webDetection = r.webDetection || {};
    const webEntities = (webDetection.webEntities || []).filter(e => e.description).slice(0, 8);
    const bestGuessLabels = (webDetection.bestGuessLabels || []).map(l => l.label);
    const pagesWithMatchingImages = (webDetection.pagesWithMatchingImages || []).slice(0, 5);
    const visuallySimilarImages = (webDetection.visuallySimilarImages || []).slice(0, 6);

    return {
      labels, faces, objects, safeSearch, dominantColors: colors,
      fullText, landmarks, logos,
      webEntities, bestGuessLabels, pagesWithMatchingImages, visuallySimilarImages,
    };
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
      throw new Error(body.detail || body.error?.message || `Gemini server error (${resp.status})`);
    }

    return resp.json();
  }

  // ------------------------------------------------------------------
  // Renderers
  // ------------------------------------------------------------------

  function renderLabels(googleLabels) {
    const container = document.getElementById('vision-labels');
    if (!googleLabels || googleLabels.length === 0) {
      container.innerHTML = '<p style="color:var(--text-dim); font-size:0.85rem;">No labels detected</p>';
      return;
    }
    container.innerHTML = `<div style="background:var(--bg-card); padding:12px 16px; border-radius:var(--radius);">` +
      googleLabels.map(l => {
        const pct = (l.score * 100).toFixed(1);
        return `
          <div style="display:flex; align-items:center; gap:8px; margin-bottom:4px; font-family:var(--mono); font-size:0.75rem;">
            <span style="width:120px; text-align:right; color:var(--text); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">${l.name}</span>
            <div style="flex:1; height:14px; background:#222; border-radius:3px; overflow:hidden;">
              <div style="height:100%; width:${pct}%; background:var(--yellow); border-radius:3px;"></div>
            </div>
            <span style="width:45px; color:var(--text-dim); white-space:nowrap;">${pct}%</span>
          </div>
        `;
      }).join('') +
      `</div>`;
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

      const rows = emotions.map(e => {
        const label = e.value.replace(/_/g, ' ').toLowerCase();
        const color = e.value === 'VERY_LIKELY' ? 'var(--green)' : e.value === 'LIKELY' ? 'var(--green)' : e.value === 'POSSIBLE' ? 'var(--yellow)' : 'var(--text-dim)';
        return `<span style="color:var(--text);">${e.name}:</span> <span style="color:${color};">${label}</span>`;
      }).join(' &nbsp;&middot;&nbsp; ');

      return `
        <div style="background:var(--bg-card); padding:10px 16px; border-radius:var(--radius); margin-bottom:6px; font-family:var(--mono); font-size:0.75rem; line-height:1.8;">
          <span style="color:var(--green);">Face ${i + 1}</span> <span style="color:var(--text-dim);">(${(f.confidence * 100).toFixed(0)}%)</span><br>
          ${rows}
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

    const rows = categories.map(c => {
      const val = ss[c.key] || 'UNKNOWN';
      const label = val.replace(/_/g, ' ').toLowerCase();
      const color = val === 'VERY_LIKELY' || val === 'LIKELY' ? 'var(--red)' : val === 'POSSIBLE' ? 'var(--yellow)' : 'var(--text-dim)';
      return `<span style="color:var(--text);">${c.label}:</span> <span style="color:${color};">${label}</span>`;
    }).join(' &nbsp;&middot;&nbsp; ');

    container.innerHTML = `
      <div style="background:var(--bg-card); padding:10px 16px; border-radius:var(--radius); margin-bottom:24px; font-family:var(--mono); font-size:0.75rem; line-height:1.8;">
        ${rows}
      </div>
    `;
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

  function renderLandmarks(landmarks) {
    const section = document.getElementById('vision-landmarks-section');
    const container = document.getElementById('vision-landmarks');
    section.classList.remove('hidden');
    if (!landmarks || landmarks.length === 0) {
      container.innerHTML = `<div style="background:var(--bg-card); padding:10px 16px; border-radius:var(--radius); margin-bottom:20px; font-family:var(--mono); font-size:0.75rem; color:var(--text-dim);">No landmarks detected</div>`;
      return;
    }
    container.innerHTML = `
      <div style="background:var(--bg-card); padding:10px 16px; border-radius:var(--radius); margin-bottom:20px; font-family:var(--mono); font-size:0.75rem; line-height:1.8;">
        ${landmarks.map(l => {
          const score = (l.score * 100).toFixed(0);
          const coords = l.locations && l.locations[0] && l.locations[0].latitude
            ? ` <span style="color:var(--text-dim);">(${l.locations[0].latitude.toFixed(4)}, ${l.locations[0].longitude.toFixed(4)})</span>`
            : '';
          return `<span style="color:var(--text);">${l.name}</span> <span style="color:var(--yellow);">${score}%</span>${coords}`;
        }).join('<br>')}
      </div>
    `;
  }

  function renderLogos(logos) {
    const section = document.getElementById('vision-logos-section');
    const container = document.getElementById('vision-logos');
    section.classList.remove('hidden');
    if (!logos || logos.length === 0) {
      container.innerHTML = `<div style="background:var(--bg-card); padding:10px 16px; border-radius:var(--radius); margin-bottom:20px; font-family:var(--mono); font-size:0.75rem; color:var(--text-dim);">No logos detected</div>`;
      return;
    }
    container.innerHTML = `
      <div style="background:var(--bg-card); padding:10px 16px; border-radius:var(--radius); margin-bottom:20px; font-family:var(--mono); font-size:0.75rem; line-height:1.8;">
        ${logos.map(l => {
          const score = (l.score * 100).toFixed(0);
          return `<span style="color:var(--text);">${l.name}</span> <span style="color:var(--yellow);">${score}%</span>`;
        }).join(' &middot; ')}
      </div>
    `;
  }

  function renderText(text) {
    const section = document.getElementById('vision-text-section');
    const container = document.getElementById('vision-text');
    if (!text) {
      section.classList.add('hidden');
      return;
    }
    section.classList.remove('hidden');
    container.innerHTML = `
      <div style="background:var(--bg-card); padding:10px 16px; border-radius:var(--radius); margin-bottom:24px; font-family:var(--mono); font-size:0.75rem; color:var(--text); white-space:pre-wrap; line-height:1.6;">${text}</div>
    `;
  }

  function renderWeb(data) {
    const section = document.getElementById('vision-web-section');
    const container = document.getElementById('vision-web');
    const hasContent = (data.webEntities && data.webEntities.length > 0) ||
                       (data.bestGuessLabels && data.bestGuessLabels.length > 0) ||
                       (data.visuallySimilarImages && data.visuallySimilarImages.length > 0);
    if (!hasContent) {
      section.classList.add('hidden');
      return;
    }
    section.classList.remove('hidden');

    let html = '<div style="background:var(--bg-card); padding:12px 16px; border-radius:var(--radius); margin-bottom:24px; font-family:var(--mono); font-size:0.75rem;">';

    // Best guess
    if (data.bestGuessLabels && data.bestGuessLabels.length > 0) {
      html += `<div style="margin-bottom:10px;"><span style="color:var(--text-dim);">Best guess:</span> <span style="color:var(--accent);">${data.bestGuessLabels.join(', ')}</span></div>`;
    }

    // Web entities
    if (data.webEntities && data.webEntities.length > 0) {
      html += `<div style="margin-bottom:10px;"><span style="color:var(--text-dim);">Related concepts:</span> `;
      html += data.webEntities.map(e => {
        const score = e.score ? ` (${(e.score * 100).toFixed(0)}%)` : '';
        return `<span style="color:var(--text);">${e.description}${score}</span>`;
      }).join(' &middot; ');
      html += `</div>`;
    }

    // Visually similar images
    if (data.visuallySimilarImages && data.visuallySimilarImages.length > 0) {
      html += `<div><span style="color:var(--text-dim);">Visually similar images found online:</span>
        <div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:8px;">`;
      html += data.visuallySimilarImages.map(img => {
        const imgUrl = img.url || '';
        return `<a href="${imgUrl}" target="_blank" rel="noopener" style="display:block; width:80px; height:80px; border-radius:4px; overflow:hidden; border:1px solid #333;">
          <img src="${imgUrl}" style="width:100%; height:100%; object-fit:cover;" onerror="this.parentElement.style.display='none'">
        </a>`;
      }).join('');
      html += `</div></div>`;
    }

    html += '</div>';
    container.innerHTML = html;
  }
})();
