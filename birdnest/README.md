# Birdnest Classifier Workshop

A browser-based binary image classifier that distinguishes **bird nests** from **non-bird-nest** objects. Built with TensorFlow.js and transfer learning from MobileNet v2.

## Quick Start

Open `demo/index.html` in your browser. No server required — everything runs client-side.

### Workflow

1. **Create Dataset** — Upload bird nest and non-bird-nest images via drag-and-drop, file picker, or URL paste. Images persist in IndexedDB.
2. **Train Model** — Transfer learning from MobileNet v2. Adjust epochs, batch size, and learning rate. Watch training progress with live loss/accuracy charts.
3. **Classify** — Upload new images to test your trained model. Misclassified images can be added back to the dataset.

## Python Scraping Scripts

Optional scripts to bulk-download training images.

### Setup

```bash
cd scripts
pip install -r requirements.txt
```

### scrape_google.py

Downloads images via Bing image search (more reliable than Google direct).

```bash
python scrape_google.py --query "bird nest" --count 50 --output-dir ../dataset/birdnest
python scrape_google.py --query "empty tree branch" --count 50 --output-dir ../dataset/not_birdnest
python scrape_google.py --query "basket weaving" --count 30 --output-dir ../dataset/not_birdnest
```

### scrape_websites.py

Downloads from nature photography sites: Wikimedia Commons, Flickr, iNaturalist.

```bash
# Wikimedia Commons (no API key needed)
python scrape_websites.py --source wikimedia --count 30 --output-dir ../dataset/birdnest

# Flickr (requires FLICKR_API_KEY env var)
export FLICKR_API_KEY=your_key_here
python scrape_websites.py --source flickr --count 30 --output-dir ../dataset/birdnest

# iNaturalist (no API key needed)
python scrape_websites.py --source inaturalist --count 30 --output-dir ../dataset/birdnest
```

### scrape_instagram.py

Downloads from Instagram hashtags using instaloader.

```bash
python scrape_instagram.py --hashtag birdnest --count 30 --output-dir ../dataset/birdnest
python scrape_instagram.py --hashtag birdsnest --count 20 --output-dir ../dataset/birdnest
```

For authenticated access (larger downloads):

```bash
export INSTAGRAM_USER=your_username
export INSTAGRAM_PASS=your_password
python scrape_instagram.py --hashtag birdnest --count 100 --output-dir ../dataset/birdnest
```

## Libraries

- [TensorFlow.js](https://www.tensorflow.org/js) v4.22.0 (CDN)
- MobileNet v2 (loaded from TensorFlow.js model hosting)

## Notes

- All browser processing happens on-device. No data is sent to any server.
- The dataset and trained model are cached in IndexedDB and persist across page reloads.
- Recommended: at least 20 images per category for reasonable results.
