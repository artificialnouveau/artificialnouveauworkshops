/**
 * step1-dataset.js — Dataset creation: upload, URL fetch, page scraping, thumbnail grid, IndexedDB persistence
 */

(function () {
  const CATEGORIES = ['birdnest', 'not_birdnest'];

  // CORS proxies for fetching page HTML (tried in order)
  const CORS_PROXIES = [
    (url) => `https://api.allorigins.win/raw?url=${encodeURIComponent(url)}`,
    (url) => `https://corsproxy.io/?url=${encodeURIComponent(url)}`,
    (url) => `https://api.codetabs.com/v1/proxy?quest=${encodeURIComponent(url)}`,
  ];

  const IMAGE_EXTENSIONS = /\.(jpe?g|png|gif|webp|bmp|tiff?)(\?.*)?$/i;
  const MIN_IMAGE_WIDTH = 80;  // Skip tiny icons
  const MIN_IMAGE_HEIGHT = 80;

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
        const ok = await addImageFromUrl(category, url);
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
      const pageUrl = input.split('\n')[0].trim();
      if (!pageUrl.startsWith('http')) {
        showStatus(scrapeStatus, 'Enter a valid URL starting with http:// or https://', 'error');
        return;
      }

      scrapeBtn.disabled = true;
      scrapeBtn.textContent = 'Scraping...';
      showStatus(scrapeStatus, 'Fetching page via proxy...', '');

      try {
        const html = await fetchPageHTML(pageUrl);
        if (!html) {
          showStatus(scrapeStatus, 'Could not fetch page — all CORS proxies failed. Try pasting direct image URLs instead.', 'error');
          scrapeBtn.textContent = 'Scrape Page for Images';
          scrapeBtn.disabled = false;
          return;
        }

        showStatus(scrapeStatus, 'Parsing page for images...', '');
        const imageUrls = extractImageUrls(html, pageUrl);

        if (imageUrls.length === 0) {
          showStatus(scrapeStatus, 'No images found on that page (images may be loaded by JavaScript)', 'error');
          scrapeBtn.textContent = 'Scrape Page for Images';
          scrapeBtn.disabled = false;
          return;
        }

        showStatus(scrapeStatus, `Found ${imageUrls.length} image candidates. Downloading (skipping tiny ones)...`, '');
        let loaded = 0;
        let skipped = 0;
        for (let i = 0; i < imageUrls.length; i++) {
          const ok = await addImageFromUrl(category, imageUrls[i]);
          if (ok) loaded++;
          else skipped++;
          showStatus(scrapeStatus, `Progress: ${i + 1}/${imageUrls.length} — added ${loaded}, skipped ${skipped}`, '');
        }
        showStatus(scrapeStatus, `Done! Added ${loaded} images (${skipped} skipped as too small or failed)`, 'done');
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

  /* ---- Fetch page HTML through CORS proxies ---- */
  async function fetchPageHTML(url) {
    // Try direct fetch first (same-origin or CORS-enabled sites)
    try {
      const resp = await fetch(url);
      if (resp.ok) {
        const text = await resp.text();
        if (text.length > 100) return text; // sanity check
      }
    } catch { /* expected — CORS blocked */ }

    // Try each proxy
    for (const proxyFn of CORS_PROXIES) {
      try {
        const proxyUrl = proxyFn(url);
        console.log('Trying proxy:', proxyUrl);
        const resp = await fetch(proxyUrl);
        if (resp.ok) {
          const text = await resp.text();
          if (text.length > 100) return text;
        }
      } catch (err) {
        console.warn('Proxy failed:', err.message);
      }
    }

    return null;
  }

  /* ---- Extract image URLs from HTML ---- */
  function extractImageUrls(html, pageUrl) {
    const baseUrl = new URL(pageUrl);
    const found = new Set();

    // Parse as DOM
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, 'text/html');

    // 1) <img> tags — check src, data-src, data-lazy-src, data-original, data-file-width
    doc.querySelectorAll('img').forEach(img => {
      // All possible src attributes
      ['src', 'data-src', 'data-lazy-src', 'data-original', 'data-file-url'].forEach(attr => {
        const val = img.getAttribute(attr);
        if (val) addUrl(val, baseUrl, found);
      });
      // srcset on img tags (Wikipedia uses this heavily)
      const srcset = img.getAttribute('srcset');
      if (srcset) parseSrcset(srcset, baseUrl, found);
    });

    // 2) <source> srcset (inside <picture>)
    doc.querySelectorAll('source[srcset]').forEach(source => {
      parseSrcset(source.getAttribute('srcset'), baseUrl, found);
    });

    // 3) <a> tags linking directly to image files
    doc.querySelectorAll('a[href]').forEach(a => {
      const href = a.getAttribute('href');
      if (href && IMAGE_EXTENSIONS.test(href)) {
        addUrl(href, baseUrl, found);
      }
    });

    // 4) CSS background-image in inline styles
    doc.querySelectorAll('[style]').forEach(el => {
      const style = el.getAttribute('style');
      const matches = style.match(/url\(\s*['"]?([^'")\s]+)['"]?\s*\)/g);
      if (matches) {
        matches.forEach(m => {
          const url = m.replace(/url\(\s*['"]?/, '').replace(/['"]?\s*\)/, '');
          addUrl(url, baseUrl, found);
        });
      }
    });

    // 5) og:image and twitter:image meta
    doc.querySelectorAll('meta[property="og:image"], meta[name="twitter:image"], meta[name="twitter:image:src"]').forEach(meta => {
      const content = meta.getAttribute('content');
      if (content) addUrl(content, baseUrl, found);
    });

    // 6) Google Images: extract URLs from script tags and data attributes
    //    Google embeds real image URLs in encoded JSON inside script tags
    doc.querySelectorAll('script').forEach(script => {
      const text = script.textContent || '';
      // Match URLs that look like images in script content
      const urlPattern = /https?:\/\/[^\s"'\\]+\.(?:jpe?g|png|gif|webp)(?:[^\s"'\\]*)/gi;
      let match;
      while ((match = urlPattern.exec(text)) !== null) {
        let url = match[0];
        // Unescape common JS escapes
        url = url.replace(/\\u003d/gi, '=')
                 .replace(/\\u0026/gi, '&')
                 .replace(/\\x3d/gi, '=')
                 .replace(/\\x26/gi, '&')
                 .replace(/\\\//g, '/')
                 .replace(/\\"/g, '');
        // Skip Google's own UI images
        if (url.includes('gstatic.com/images') || url.includes('google.com/images')) continue;
        if (url.includes('googleusercontent') || url.includes('ggpht')) continue;
        addUrl(url, baseUrl, found);
      }
    });

    // 7) data-tbnid, data-ou, data-iurl — Google Images specific attributes
    doc.querySelectorAll('[data-ou]').forEach(el => {
      addUrl(el.getAttribute('data-ou'), baseUrl, found);
    });
    doc.querySelectorAll('[data-iurl]').forEach(el => {
      addUrl(el.getAttribute('data-iurl'), baseUrl, found);
    });

    // 8) Look for image URLs in all data- attributes as a catch-all
    doc.querySelectorAll('[data-src], [data-original], [data-lazy], [data-url]').forEach(el => {
      ['data-src', 'data-original', 'data-lazy', 'data-url'].forEach(attr => {
        const val = el.getAttribute(attr);
        if (val && isLikelyImageUrl(val)) addUrl(val, baseUrl, found);
      });
    });

    // Filter: prefer larger image versions (remove thumbnail variants if full version exists)
    return Array.from(found).filter(url => {
      // Skip SVGs (usually icons/logos)
      if (url.endsWith('.svg')) return false;
      // Skip data URIs
      if (url.startsWith('data:')) return false;
      return true;
    });
  }

  function parseSrcset(srcset, baseUrl, found) {
    if (!srcset) return;
    srcset.split(',').forEach(entry => {
      const parts = entry.trim().split(/\s+/);
      if (parts[0]) addUrl(parts[0], baseUrl, found);
    });
  }

  function isLikelyImageUrl(str) {
    if (!str) return false;
    if (str.startsWith('data:image/')) return false; // skip inline
    return IMAGE_EXTENSIONS.test(str) || str.includes('/image') || str.includes('photo');
  }

  function addUrl(src, baseUrl, set) {
    if (!src) return;
    try {
      if (src.startsWith('data:')) return;
      if (src.startsWith('//')) src = 'https:' + src;
      const resolved = new URL(src, baseUrl).href;
      set.add(resolved);
    } catch {
      // Invalid URL
    }
  }

  /* ---- Add image from file ---- */
  async function addImageFile(category, file) {
    const blob = await App.fileToBlob(file);
    const id = await App.addImage(category, blob);
    addThumbnail(category, id, blob);
    updateCounts();
  }

  /* ---- Add image from URL (using <img> tag — most CORS-friendly) ---- */
  async function addImageFromUrl(category, url) {
    try {
      const img = await loadImageWithTimeout(url, 10000);
      // Skip tiny images (icons, spacers, 1px trackers)
      if (img.naturalWidth < MIN_IMAGE_WIDTH || img.naturalHeight < MIN_IMAGE_HEIGHT) return false;

      // Draw to canvas and export as blob
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
      // Image failed to load — try via proxy as last resort
      try {
        const blob = await fetchImageViaProxy(url);
        if (!blob) return false;

        // Validate it's actually an image by loading it
        const testImg = await loadImageWithTimeout(URL.createObjectURL(blob), 5000);
        if (testImg.naturalWidth < MIN_IMAGE_WIDTH || testImg.naturalHeight < MIN_IMAGE_HEIGHT) return false;

        const id = await App.addImage(category, blob);
        addThumbnail(category, id, blob);
        updateCounts();
        return true;
      } catch {
        return false;
      }
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

  async function fetchImageViaProxy(url) {
    for (const proxyFn of CORS_PROXIES) {
      try {
        const resp = await fetch(proxyFn(url));
        if (resp.ok) {
          const blob = await resp.blob();
          if (blob.type.startsWith('image/') && blob.size > 1000) return blob;
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
