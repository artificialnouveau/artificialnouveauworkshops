/**
 * step1-dataset.js — Dataset creation: Wikimedia presets, upload, URL fetch, IndexedDB persistence
 */

(function () {
  const CATEGORIES = ['birdnest', 'not_birdnest'];
  const MIN_IMAGE_WIDTH = 80;
  const MIN_IMAGE_HEIGHT = 80;

  function setupPanel(category) {
    const panel = document.getElementById(`panel-${category}`);
    const uploadArea = panel.querySelector('.upload-area');
    const fileInput = panel.querySelector('.file-input');
    const urlTextarea = panel.querySelector('.url-textarea');
    const fetchBtn = panel.querySelector('.btn-fetch-urls');
    const clearBtn = panel.querySelector('.btn-clear');
    const grid = document.getElementById(`grid-${category}`);
    const presetStatus = panel.querySelector('.preset-status');

    // ---- Wikimedia preset buttons ----
    panel.querySelectorAll('.btn-preset').forEach(btn => {
      btn.addEventListener('click', async () => {
        const wikiCategory = btn.dataset.wikiCategory;
        const count = parseInt(btn.dataset.count) || 50;
        btn.disabled = true;
        const origText = btn.textContent;
        btn.textContent = 'Loading...';
        showStatus(presetStatus, `Fetching from Wikimedia Commons: Category:${wikiCategory}...`, '');

        try {
          const urls = await fetchWikimediaCategory(wikiCategory, count);
          if (urls.length === 0) {
            showStatus(presetStatus, 'No images found in this category', 'error');
            btn.textContent = origText;
            btn.disabled = false;
            return;
          }

          showStatus(presetStatus, `Found ${urls.length} images. Downloading...`, '');
          let loaded = 0;
          let failed = 0;
          for (let i = 0; i < urls.length; i++) {
            const ok = await addImageFromUrl(category, urls[i]);
            if (ok) loaded++;
            else failed++;
            btn.textContent = `${loaded}/${urls.length}`;
            showStatus(presetStatus, `Downloaded ${loaded}/${urls.length}${failed ? ` (${failed} skipped)` : ''}`, '');
          }
          showStatus(presetStatus, `Done! Added ${loaded} images`, 'done');
        } catch (err) {
          console.error('Wikimedia fetch error:', err);
          showStatus(presetStatus, `Failed: ${err.message}`, 'error');
        }

        btn.textContent = origText;
        btn.disabled = false;
      });
    });

    // ---- Click to upload ----
    uploadArea.addEventListener('click', () => fileInput.click());

    // ---- File input change ----
    fileInput.addEventListener('change', async () => {
      const files = Array.from(fileInput.files);
      for (const file of files) {
        await addImageFile(category, file);
      }
      fileInput.value = '';
    });

    // ---- Drag and drop ----
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
      for (const file of files) {
        await addImageFile(category, file);
      }
    });

    // ---- Fetch direct image URLs ----
    fetchBtn.addEventListener('click', async () => {
      const scrapeStatus = panel.querySelector('.url-paste-section .scrape-status');
      const urls = urlTextarea.value.split('\n').map(u => u.trim()).filter(u => u);
      if (urls.length === 0) return;
      fetchBtn.disabled = true;
      fetchBtn.textContent = `Loading 0/${urls.length}...`;
      let loaded = 0;
      for (const url of urls) {
        const ok = await addImageFromUrl(category, url);
        if (ok) loaded++;
        fetchBtn.textContent = `${loaded}/${urls.length}`;
      }
      fetchBtn.textContent = 'Fetch URLs';
      fetchBtn.disabled = false;
      urlTextarea.value = '';
      showStatus(scrapeStatus, `Loaded ${loaded} of ${urls.length} images`, 'done');
    });

    // ---- Folder upload ----
    const folderBtn = panel.querySelector('.btn-folder-upload');
    const folderInput = panel.querySelector('.folder-input');

    folderBtn.addEventListener('click', () => folderInput.click());

    folderInput.addEventListener('change', async () => {
      const files = Array.from(folderInput.files).filter(f => f.type.startsWith('image/'));
      if (files.length === 0) {
        alert('No image files found in the selected folder.');
        return;
      }
      folderBtn.textContent = `Loading 0/${files.length}...`;
      folderBtn.disabled = true;
      for (let i = 0; i < files.length; i++) {
        await addImageFile(category, files[i]);
        folderBtn.textContent = `${i + 1}/${files.length}`;
      }
      folderBtn.textContent = 'Upload Folder';
      folderBtn.disabled = false;
      folderInput.value = '';
    });

    // ---- Clear ----
    clearBtn.addEventListener('click', async () => {
      await App.clearCategory(category);
      grid.innerHTML = '';
      updateCounts();
    });
  }

  /* ---- Status helper ---- */
  function showStatus(el, msg, type) {
    el.classList.remove('hidden', 'done', 'error');
    if (type) el.classList.add(type);
    el.textContent = msg;
  }

  /* ---- Wikimedia Commons API (CORS-enabled natively) ---- */
  async function fetchWikimediaCategory(categoryName, maxImages) {
    const imageUrls = [];
    let gcmcontinue = '';

    while (imageUrls.length < maxImages) {
      const params = new URLSearchParams({
        action: 'query',
        generator: 'categorymembers',
        gcmtitle: `Category:${categoryName}`,
        gcmtype: 'file',
        gcmlimit: Math.min(50, maxImages - imageUrls.length).toString(),
        prop: 'imageinfo',
        iiprop: 'url|mime',
        iiurlwidth: '640',
        format: 'json',
        origin: '*',
      });
      if (gcmcontinue) params.set('gcmcontinue', gcmcontinue);

      const resp = await fetch(`https://commons.wikimedia.org/w/api.php?${params}`);
      if (!resp.ok) throw new Error(`Wikimedia API returned ${resp.status}`);
      const data = await resp.json();

      const pages = data.query?.pages;
      if (!pages) break;

      for (const page of Object.values(pages)) {
        const info = page.imageinfo?.[0];
        if (!info) continue;
        const mime = info.mime || '';
        if (!mime.startsWith('image/') || mime === 'image/svg+xml') continue;
        // Prefer the 640px thumbnail, fall back to full URL
        const url = info.thumburl || info.url;
        if (url) imageUrls.push(url);
      }

      // Pagination
      gcmcontinue = data.continue?.gcmcontinue;
      if (!gcmcontinue) break;
    }

    return imageUrls.slice(0, maxImages);
  }

  /* ---- Add image from file ---- */
  async function addImageFile(category, file) {
    const blob = await App.fileToBlob(file);
    const id = await App.addImage(category, blob);
    addThumbnail(category, id, blob);
    updateCounts();
  }

  /* ---- Add image from URL (using <img> tag) ---- */
  async function addImageFromUrl(category, url) {
    try {
      const img = await loadImageWithTimeout(url, 15000);
      if (img.naturalWidth < MIN_IMAGE_WIDTH || img.naturalHeight < MIN_IMAGE_HEIGHT) return false;

      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext('2d').drawImage(img, 0, 0);

      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.92));
      if (!blob) return false;

      const id = await App.addImage(category, blob);
      addThumbnail(category, id, blob);
      updateCounts();
      return true;
    } catch {
      console.warn('Failed to load image:', url);
      return false;
    }
  }

  function loadImageWithTimeout(src, ms) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      const timer = setTimeout(() => {
        img.src = '';
        reject(new Error('timeout'));
      }, ms);
      img.onload = () => { clearTimeout(timer); resolve(img); };
      img.onerror = () => { clearTimeout(timer); reject(new Error('load failed')); };
      img.src = src;
    });
  }

  /* ---- Thumbnail grid ---- */
  function addThumbnail(category, id, blob) {
    const grid = document.getElementById(`grid-${category}`);
    const item = document.createElement('div');
    item.className = 'thumbnail-item';
    item.dataset.id = id;

    const img = document.createElement('img');
    img.src = URL.createObjectURL(blob);

    const del = document.createElement('button');
    del.className = 'thumbnail-delete';
    del.textContent = '\u00d7';
    del.addEventListener('click', async (e) => {
      e.stopPropagation();
      await App.deleteImage(id);
      item.remove();
      updateCounts();
    });

    item.appendChild(img);
    item.appendChild(del);
    grid.appendChild(item);
  }

  async function updateCounts() {
    for (const cat of CATEGORIES) {
      const images = await App.getImagesByCategory(cat);
      const count = images.length;
      document.getElementById(`count-${cat}`).textContent = count;
      const label = cat === 'birdnest' ? 'Bird Nest' : 'Not Bird Nest';
      document.getElementById(`status-${cat}`).textContent = `${label}: ${count} images`;
    }
  }

  async function loadExisting() {
    for (const cat of CATEGORIES) {
      const images = await App.getImagesByCategory(cat);
      for (const record of images) {
        addThumbnail(cat, record.id, record.blob);
      }
    }
    updateCounts();
  }

  // Init
  const origInit = App.init.bind(App);
  App.init = async function () {
    await origInit();
    CATEGORIES.forEach(setupPanel);
    await loadExisting();
  };
})();
