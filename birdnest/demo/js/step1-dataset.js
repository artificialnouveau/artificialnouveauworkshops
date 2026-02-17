/**
 * step1-dataset.js — Dataset creation: upload, URL fetch, page scraping, thumbnail grid, IndexedDB persistence
 */

(function () {
  const CATEGORIES = ['birdnest', 'not_birdnest'];
  const CORS_PROXIES = [
    'https://api.allorigins.win/raw?url=',
    'https://corsproxy.io/?',
  ];
  const IMAGE_EXTENSIONS = /\.(jpe?g|png|gif|webp|bmp|svg|tiff?)(\?.*)?$/i;
  const MIN_IMAGE_SIZE = 5000; // Skip tiny images (icons, spacers) — 5KB

  function setupPanel(category) {
    const panel = document.getElementById(`panel-${category}`);
    const uploadArea = panel.querySelector('.upload-area');
    const fileInput = panel.querySelector('.file-input');
    const urlTextarea = panel.querySelector('.url-textarea');
    const fetchBtn = panel.querySelector('.btn-fetch-urls');
    const scrapeBtn = panel.querySelector('.btn-scrape-page');
    const scrapeStatus = panel.querySelector('.scrape-status');
    const clearBtn = panel.querySelector('.btn-clear');
    const grid = document.getElementById(`grid-${category}`);

    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', async () => {
      const files = Array.from(fileInput.files);
      for (const file of files) {
        await addImageFile(category, file);
      }
      fileInput.value = '';
    });

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
      for (const file of files) {
        await addImageFile(category, file);
      }
    });

    // Fetch direct image URLs
    fetchBtn.addEventListener('click', async () => {
      const urls = urlTextarea.value.split('\n').map(u => u.trim()).filter(u => u);
      if (urls.length === 0) return;
      fetchBtn.disabled = true;
      fetchBtn.textContent = `Loading 0/${urls.length}...`;
      let loaded = 0;
      for (const url of urls) {
        const ok = await addImageURL(category, url);
        if (ok) loaded++;
        fetchBtn.textContent = `Loading ${loaded}/${urls.length}...`;
      }
      fetchBtn.textContent = 'Fetch Direct URLs';
      fetchBtn.disabled = false;
      urlTextarea.value = '';
      showStatus(scrapeStatus, `Loaded ${loaded} of ${urls.length} images`, 'done');
    });

    // Scrape page for all images
    scrapeBtn.addEventListener('click', async () => {
      const input = urlTextarea.value.trim();
      if (!input) return;
      // Take the first line as the page URL
      const pageUrl = input.split('\n')[0].trim();
      if (!pageUrl.startsWith('http')) {
        showStatus(scrapeStatus, 'Enter a valid URL starting with http:// or https://', 'error');
        return;
      }

      scrapeBtn.disabled = true;
      scrapeBtn.textContent = 'Scraping...';
      showStatus(scrapeStatus, 'Fetching page...', '');

      try {
        const imageUrls = await scrapeImagesFromPage(pageUrl);
        if (imageUrls.length === 0) {
          showStatus(scrapeStatus, 'No images found on that page', 'error');
          scrapeBtn.textContent = 'Scrape Page for Images';
          scrapeBtn.disabled = false;
          return;
        }

        showStatus(scrapeStatus, `Found ${imageUrls.length} images. Downloading...`, '');
        let loaded = 0;
        let failed = 0;
        for (let i = 0; i < imageUrls.length; i++) {
          const ok = await addImageURL(category, imageUrls[i]);
          if (ok) loaded++;
          else failed++;
          showStatus(scrapeStatus, `Downloaded ${loaded}/${imageUrls.length} images${failed ? ` (${failed} failed)` : ''}...`, '');
        }
        showStatus(scrapeStatus, `Done! Added ${loaded} images${failed ? `, ${failed} failed` : ''}`, 'done');
        urlTextarea.value = '';
      } catch (err) {
        console.error('Scrape error:', err);
        showStatus(scrapeStatus, `Scrape failed: ${err.message}`, 'error');
      }

      scrapeBtn.textContent = 'Scrape Page for Images';
      scrapeBtn.disabled = false;
    });

    // Folder upload
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
        folderBtn.textContent = `Loading ${i + 1}/${files.length}...`;
      }
      folderBtn.textContent = 'Upload Folder';
      folderBtn.disabled = false;
      folderInput.value = '';
    });

    // Clear
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

  /* ---- Scrape images from a webpage ---- */
  async function scrapeImagesFromPage(pageUrl) {
    const html = await fetchPageHTML(pageUrl);
    if (!html) throw new Error('Could not fetch page (CORS blocked)');

    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');
    const baseUrl = new URL(pageUrl);
    const found = new Set();

    // Collect from <img> tags — src and data-src (lazy loading)
    doc.querySelectorAll('img').forEach(img => {
      const src = img.getAttribute('src') || img.getAttribute('data-src') ||
                  img.getAttribute('data-lazy-src') || img.getAttribute('data-original');
      if (src) addResolvedUrl(src, baseUrl, found);
    });

    // Collect from <a> tags linking to images
    doc.querySelectorAll('a[href]').forEach(a => {
      const href = a.getAttribute('href');
      if (href && IMAGE_EXTENSIONS.test(href)) {
        addResolvedUrl(href, baseUrl, found);
      }
    });

    // Collect from <source> tags (picture elements)
    doc.querySelectorAll('source[srcset]').forEach(source => {
      const srcset = source.getAttribute('srcset');
      if (srcset) {
        srcset.split(',').forEach(entry => {
          const url = entry.trim().split(/\s+/)[0];
          if (url) addResolvedUrl(url, baseUrl, found);
        });
      }
    });

    // Collect from CSS background-image in inline styles
    doc.querySelectorAll('[style]').forEach(el => {
      const style = el.getAttribute('style');
      const matches = style.match(/url\(['"]?([^'")\s]+)['"]?\)/g);
      if (matches) {
        matches.forEach(m => {
          const url = m.replace(/url\(['"]?/, '').replace(/['"]?\)/, '');
          if (IMAGE_EXTENSIONS.test(url)) addResolvedUrl(url, baseUrl, found);
        });
      }
    });

    // Collect from og:image and meta tags
    doc.querySelectorAll('meta[property="og:image"], meta[name="twitter:image"]').forEach(meta => {
      const content = meta.getAttribute('content');
      if (content) addResolvedUrl(content, baseUrl, found);
    });

    return Array.from(found);
  }

  function addResolvedUrl(src, baseUrl, set) {
    try {
      // Skip data URIs and tiny inline images
      if (src.startsWith('data:')) return;
      // Resolve relative URLs
      const resolved = new URL(src, baseUrl).href;
      set.add(resolved);
    } catch {
      // Invalid URL, skip
    }
  }

  async function fetchPageHTML(url) {
    // Try direct fetch first
    try {
      const resp = await fetch(url, { mode: 'cors' });
      if (resp.ok) return await resp.text();
    } catch { /* CORS blocked, try proxies */ }

    // Try CORS proxies
    for (const proxy of CORS_PROXIES) {
      try {
        const resp = await fetch(proxy + encodeURIComponent(url));
        if (resp.ok) return await resp.text();
      } catch { /* try next */ }
    }

    return null;
  }

  /* ---- Add image from file ---- */
  async function addImageFile(category, file) {
    const blob = await App.fileToBlob(file);
    const id = await App.addImage(category, blob);
    addThumbnail(category, id, blob);
    updateCounts();
  }

  /* ---- Add image from URL ---- */
  async function addImageURL(category, url) {
    try {
      // Try fetching as blob directly (handles more cases than <img> tag)
      let blob = await fetchImageAsBlob(url);
      if (blob && blob.size >= MIN_IMAGE_SIZE) {
        const id = await App.addImage(category, blob);
        addThumbnail(category, id, blob);
        updateCounts();
        return true;
      }

      // Fallback: load via <img> tag and draw to canvas
      const img = await App.loadImage(url);
      if (img.naturalWidth < 32 || img.naturalHeight < 32) return false; // too small
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext('2d').drawImage(img, 0, 0);
      blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
      if (blob && blob.size >= MIN_IMAGE_SIZE) {
        const id = await App.addImage(category, blob);
        addThumbnail(category, id, blob);
        updateCounts();
        return true;
      }
      return false;
    } catch (err) {
      console.warn('Failed to load image URL:', url, err);
      return false;
    }
  }

  async function fetchImageAsBlob(url) {
    // Direct fetch
    try {
      const resp = await fetch(url, { mode: 'cors' });
      if (resp.ok) {
        const ct = resp.headers.get('content-type') || '';
        if (ct.startsWith('image/')) return await resp.blob();
      }
    } catch { /* CORS blocked */ }

    // Try via CORS proxy
    for (const proxy of CORS_PROXIES) {
      try {
        const resp = await fetch(proxy + encodeURIComponent(url));
        if (resp.ok) {
          const ct = resp.headers.get('content-type') || '';
          if (ct.startsWith('image/')) return await resp.blob();
          // Some proxies don't pass content-type, check if it's valid image data
          const blob = await resp.blob();
          if (blob.type.startsWith('image/') || blob.size > MIN_IMAGE_SIZE) return blob;
        }
      } catch { /* try next */ }
    }

    return null;
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

  // Init when DOM and App are ready
  const origInit = App.init.bind(App);
  App.init = async function () {
    await origInit();
    CATEGORIES.forEach(setupPanel);
    await loadExisting();
  };
})();
