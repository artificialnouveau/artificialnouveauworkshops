/**
 * app.js — Shared utilities: step navigation, image loading, IndexedDB
 */

const App = {
  currentStep: 1,
  DB_NAME: 'birdnest-classifier',
  DB_VERSION: 1,
  STORE_IMAGES: 'images',
  STORE_MODEL: 'model',
  db: null,

  async init() {
    await this.openDB();
    this.setupNav();
    this.hideLoader();
  },

  /* ---- IndexedDB ---- */
  openDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.DB_NAME, this.DB_VERSION);
      request.onupgradeneeded = (e) => {
        const db = e.target.result;
        if (!db.objectStoreNames.contains(this.STORE_IMAGES)) {
          const store = db.createObjectStore(this.STORE_IMAGES, { keyPath: 'id', autoIncrement: true });
          store.createIndex('category', 'category', { unique: false });
        }
        if (!db.objectStoreNames.contains(this.STORE_MODEL)) {
          db.createObjectStore(this.STORE_MODEL, { keyPath: 'key' });
        }
      };
      request.onsuccess = (e) => {
        this.db = e.target.result;
        resolve();
      };
      request.onerror = (e) => reject(e.target.error);
    });
  },

  addImage(category, blob) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.STORE_IMAGES, 'readwrite');
      const store = tx.objectStore(this.STORE_IMAGES);
      const record = { category, blob, timestamp: Date.now() };
      const req = store.add(record);
      req.onsuccess = () => resolve(req.result);
      req.onerror = (e) => reject(e.target.error);
    });
  },

  getImagesByCategory(category) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.STORE_IMAGES, 'readonly');
      const store = tx.objectStore(this.STORE_IMAGES);
      const index = store.index('category');
      const req = index.getAll(category);
      req.onsuccess = () => resolve(req.result);
      req.onerror = (e) => reject(e.target.error);
    });
  },

  deleteImage(id) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.STORE_IMAGES, 'readwrite');
      const store = tx.objectStore(this.STORE_IMAGES);
      const req = store.delete(id);
      req.onsuccess = () => resolve();
      req.onerror = (e) => reject(e.target.error);
    });
  },

  clearCategory(category) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.STORE_IMAGES, 'readwrite');
      const store = tx.objectStore(this.STORE_IMAGES);
      const index = store.index('category');
      const req = index.openCursor(category);
      req.onsuccess = (e) => {
        const cursor = e.target.result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        } else {
          resolve();
        }
      };
      req.onerror = (e) => reject(e.target.error);
    });
  },

  saveModel(key, data) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.STORE_MODEL, 'readwrite');
      const store = tx.objectStore(this.STORE_MODEL);
      const req = store.put({ key, ...data });
      req.onsuccess = () => resolve();
      req.onerror = (e) => reject(e.target.error);
    });
  },

  getModel(key) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.STORE_MODEL, 'readonly');
      const store = tx.objectStore(this.STORE_MODEL);
      const req = store.get(key);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = (e) => reject(e.target.error);
    });
  },

  /* ---- Image Helpers ---- */
  loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = reject;
      if (src instanceof Blob) {
        img.src = URL.createObjectURL(src);
      } else {
        img.src = src;
      }
    });
  },

  fileToBlob(file) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => {
        const blob = new Blob([reader.result], { type: file.type });
        resolve(blob);
      };
      reader.readAsArrayBuffer(file);
    });
  },

  blobToDataURL(blob) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.readAsDataURL(blob);
    });
  },

  /* ---- Step Navigation ---- */
  setupNav() {
    document.querySelectorAll('.step-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const step = parseInt(tab.dataset.step);
        this.goToStep(step);
      });
    });
  },

  goToStep(n) {
    this.currentStep = n;
    document.querySelectorAll('.step').forEach(s => s.classList.remove('active'));
    document.querySelectorAll('.step-tab').forEach(t => t.classList.remove('active'));
    document.getElementById(`step-${n}`).classList.add('active');
    document.querySelector(`.step-tab[data-step="${n}"]`).classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
  },

  /* ---- Loader ---- */
  hideLoader() {
    const overlay = document.getElementById('loading-overlay');
    overlay.classList.add('hidden');
    setTimeout(() => overlay.remove(), 500);
  }
};

// Boot
document.addEventListener('DOMContentLoaded', () => App.init());
