#!/usr/bin/env python3
"""
scrape_websites.py — Download bird nest images from nature photography sites.

Supported sources:
    - wikimedia: Wikimedia Commons (search API)
    - flickr:    Flickr public photos (requires FLICKR_API_KEY env var)
    - inaturalist: iNaturalist observations API

Usage:
    python scrape_websites.py --source wikimedia --count 30 --output-dir dataset/birdnest
    python scrape_websites.py --source flickr --count 30 --output-dir dataset/birdnest
    python scrape_websites.py --source inaturalist --count 30 --output-dir dataset/birdnest
"""

import argparse
import os
import time
from pathlib import Path

import requests

HEADERS = {"User-Agent": "BirdnestClassifier/1.0 (educational workshop)"}


def download_image(url, output_dir, index):
    """Download a single image and save it."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, stream=True)
        resp.raise_for_status()
        content_type = resp.headers.get("Content-Type", "")
        ext = ".jpg"
        if "png" in content_type:
            ext = ".png"
        elif "gif" in content_type:
            ext = ".gif"

        filepath = output_dir / f"{index:04d}{ext}"
        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Failed to download {url}: {e}")
        return False


def scrape_wikimedia(count, output_dir):
    """Search Wikimedia Commons for bird nest images."""
    print("Scraping Wikimedia Commons...")
    url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": "bird nest",
        "gsrnamespace": 6,
        "gsrlimit": min(count, 50),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": 640,
        "format": "json",
    }

    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    pages = data.get("query", {}).get("pages", {})
    downloaded = 0
    for page in pages.values():
        if downloaded >= count:
            break
        info = page.get("imageinfo", [{}])[0]
        img_url = info.get("thumburl") or info.get("url")
        if img_url:
            if download_image(img_url, output_dir, downloaded):
                downloaded += 1
            time.sleep(0.5)

    print(f"  Downloaded {downloaded} images from Wikimedia Commons")


def scrape_flickr(count, output_dir):
    """Search Flickr for bird nest photos (requires FLICKR_API_KEY)."""
    api_key = os.environ.get("FLICKR_API_KEY")
    if not api_key:
        print("Error: Set FLICKR_API_KEY environment variable.")
        print("  Get a free key at https://www.flickr.com/services/api/misc.api_keys.html")
        return

    print("Scraping Flickr...")
    url = "https://www.flickr.com/services/rest/"
    params = {
        "method": "flickr.photos.search",
        "api_key": api_key,
        "text": "bird nest",
        "sort": "relevance",
        "per_page": min(count, 100),
        "format": "json",
        "nojsoncallback": 1,
        "license": "1,2,3,4,5,6",  # Creative Commons licenses
    }

    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    photos = data.get("photos", {}).get("photo", [])
    downloaded = 0
    for photo in photos:
        if downloaded >= count:
            break
        img_url = f"https://live.staticflickr.com/{photo['server']}/{photo['id']}_{photo['secret']}_z.jpg"
        if download_image(img_url, output_dir, downloaded):
            downloaded += 1
        time.sleep(0.3)

    print(f"  Downloaded {downloaded} images from Flickr")


def scrape_inaturalist(count, output_dir):
    """Download bird nest observations from iNaturalist."""
    print("Scraping iNaturalist...")
    url = "https://api.inaturalist.org/v1/observations"
    params = {
        "q": "bird nest",
        "photos": "true",
        "per_page": min(count, 50),
        "order": "desc",
        "order_by": "votes",
    }

    resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    results = data.get("results", [])
    downloaded = 0
    for obs in results:
        if downloaded >= count:
            break
        photos = obs.get("photos", [])
        if photos:
            img_url = photos[0].get("url", "").replace("square", "medium")
            if img_url and download_image(img_url, output_dir, downloaded):
                downloaded += 1
            time.sleep(0.3)

    print(f"  Downloaded {downloaded} images from iNaturalist")


SOURCES = {
    "wikimedia": scrape_wikimedia,
    "flickr": scrape_flickr,
    "inaturalist": scrape_inaturalist,
}


def main():
    parser = argparse.ArgumentParser(description="Download bird nest images from nature sites")
    parser.add_argument(
        "--source",
        required=True,
        choices=list(SOURCES.keys()),
        help="Image source to scrape",
    )
    parser.add_argument("--count", type=int, default=30, help="Number of images to download")
    parser.add_argument("--output-dir", required=True, help="Output directory for images")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    SOURCES[args.source](args.count, output)


if __name__ == "__main__":
    main()
