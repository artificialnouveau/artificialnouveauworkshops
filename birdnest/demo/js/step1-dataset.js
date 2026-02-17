/**
 * step1-dataset.js — Dataset creation: upload, URL fetch, thumbnail grid, IndexedDB persistence
 */

(function () {
  const CATEGORIES = ['birdnest', 'not_birdnest'];

  function setupPanel(category) {
    const panel = document.getElementById(`panel-${category}`);
    const uploadArea = panel.querySelector('.upload-area');
    const fileInput = panel.querySelector('.file-input');
    const urlTextarea = panel.querySelector('.url-textarea');
    const fetchBtn = panel.querySelector('.btn-fetch-urls');
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

    // URL fetch
    fetchBtn.addEventListener('click', async () => {
      const urls = urlTextarea.value.split('\n').map(u => u.trim()).filter(u => u);
      for (const url of urls) {
        await addImageURL(category, url);
      }
      urlTextarea.value = '';
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

  async function addImageFile(category, file) {
    const blob = await App.fileToBlob(file);
    const id = await App.addImage(category, blob);
    addThumbnail(category, id, blob);
    updateCounts();
  }

  async function addImageURL(category, url) {
    try {
      // Load image via img tag to handle CORS gracefully
      const img = await App.loadImage(url);
      const canvas = document.createElement('canvas');
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext('2d').drawImage(img, 0, 0);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.9));
      if (blob) {
        const id = await App.addImage(category, blob);
        addThumbnail(category, id, blob);
        updateCounts();
      }
    } catch (err) {
      console.warn('Failed to load image URL:', url, err);
    }
  }

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
