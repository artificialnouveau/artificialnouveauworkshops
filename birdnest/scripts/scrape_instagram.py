#!/usr/bin/env python3
"""
scrape_instagram.py — Download bird nest images from Instagram hashtags using instaloader.

Usage:
    python scrape_instagram.py --hashtag birdnest --count 30 --output-dir dataset/birdnest
    python scrape_instagram.py --hashtag birdsnest --count 20 --output-dir dataset/birdnest

Suggested hashtags:
    birdnest, birdsnest, nestbuilding, birdnesting, nesthunting

Note: Instagram may require login for larger downloads. Set INSTAGRAM_USER and
INSTAGRAM_PASS environment variables for authenticated access.
"""

import argparse
import os
import shutil
from pathlib import Path

import instaloader


def main():
    parser = argparse.ArgumentParser(description="Download images from Instagram hashtags")
    parser.add_argument("--hashtag", required=True, help="Instagram hashtag (without #)")
    parser.add_argument("--count", type=int, default=30, help="Max images to download")
    parser.add_argument("--output-dir", required=True, help="Output directory for images")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    L = instaloader.Instaloader(
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        post_metadata_txt_pattern="",
    )

    # Optional login
    username = os.environ.get("INSTAGRAM_USER")
    password = os.environ.get("INSTAGRAM_PASS")
    if username and password:
        try:
            L.login(username, password)
            print(f"Logged in as {username}")
        except Exception as e:
            print(f"Login failed: {e}. Continuing without authentication.")

    hashtag = instaloader.Hashtag.from_name(L.context, args.hashtag)
    print(f"Downloading from #{args.hashtag}...")

    downloaded = 0
    for post in hashtag.get_posts():
        if downloaded >= args.count:
            break

        try:
            L.download_post(post, target=str(output / "_temp"))

            # Move image files to output directory
            temp_dir = output / "_temp"
            if temp_dir.exists():
                for img_file in temp_dir.glob("*.jpg"):
                    dest = output / f"{downloaded:04d}.jpg"
                    shutil.move(str(img_file), str(dest))
                    downloaded += 1
                    break  # One image per post

                # Clean up temp
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            print(f"  Skipping post: {e}")

    print(f"Downloaded {downloaded} images to {output}")


if __name__ == "__main__":
    main()
