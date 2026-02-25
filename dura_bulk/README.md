# Dura Bulk Detector

Scrape Instagram images by hashtag or profile, detect boats with YOLOv8, OCR for "Dura Bulk" text, and sort results.

## Setup

```bash
pip3 install instaloader
```

### Create an Instagram session (one-time)

```bash
python3 -m instaloader --login YOUR_USERNAME
```

> **Note:** Do NOT use `instaloader --login ...` directly — on many systems (especially macOS), `pip3` installs the CLI script to a directory that is not on your `PATH`, so the terminal will say "command not found". Using `python3 -m instaloader` always works.

### Save your username (so you don't need `--login` every time)

**Option 1 — Set it for this terminal session:**

```bash
export INSTA_USERNAME=your_username
```

**Option 2 — Set it permanently (add to your shell profile):**

```bash
echo 'export INSTA_USERNAME=your_username' >> ~/.zshrc
source ~/.zshrc
```

## Usage

```bash
# Download by hashtag
python3 download_images.py "#durabulk" --max 50

# Download by profile
python3 download_images.py "@durabulk" --max 50

# Override the saved username
python3 download_images.py "#durabulk" --login OTHER_USERNAME --max 50

# Specify date range
python3 download_images.py "#durabulk" --start 2025-01-01 --end 2025-06-30
```

The `--login` flag still works if you ever need to override the `INSTA_USERNAME` environment variable.
