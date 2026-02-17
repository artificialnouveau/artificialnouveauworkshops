#!/usr/bin/env python3
"""
Download images from Martin Parr's Instagram (@martinparrstudio).
Run this once locally before deploying the static site.

Usage:
    pip install instaloader
    python download_images.py
"""

import instaloader
import json
import os
import shutil
import sys

PROFILE = "martinparrstudio"
OUTPUT_DIR = "images"
MAX_POSTS = 100  # adjust as needed


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    L = instaloader.Instaloader(
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        post_metadata_txt_pattern="",
    )

    # Optional: login for private profiles or higher rate limits
    # L.login("your_username", "your_password")

    print(f"Fetching posts from @{PROFILE}...")
    profile = instaloader.Profile.from_username(L.context, PROFILE)

    image_files = []
    count = 0

    for post in profile.get_posts():
        if count >= MAX_POSTS:
            break

        if not post.is_video:
            filename = f"{post.date_utc.strftime('%Y%m%d_%H%M%S')}_{post.shortcode}.jpg"
            filepath = os.path.join(OUTPUT_DIR, filename)

            if not os.path.exists(filepath):
                L.download_pic(filepath, post.url, post.date_utc)
                # instaloader appends extension, rename if needed
                downloaded = filepath + ".jpg" if not filepath.endswith(".jpg") else filepath
                if os.path.exists(downloaded) and downloaded != filepath:
                    shutil.move(downloaded, filepath)

            image_files.append(filename)
            count += 1
            print(f"  [{count}/{MAX_POSTS}] {filename}")

    # Also handle sidecar (carousel) posts - download first image
    print(f"\nDownloaded {len(image_files)} images.")

    # Generate image list JSON for the static site
    # Re-scan the directory to catch any existing images too
    all_images = sorted(
        f for f in os.listdir(OUTPUT_DIR)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))
    )

    with open("image-list.json", "w") as f:
        json.dump(all_images, f, indent=2)

    print(f"Wrote image-list.json with {len(all_images)} entries.")


if __name__ == "__main__":
    main()
