#!/usr/bin/env python3
"""
Download images from Instagram by hashtag or profile.

Usage:
    pip3 install instaloader

    # Create a session first (one-time):
    python3 -m instaloader --login YOUR_USERNAME

    # Set your username so you don't have to pass --login every time:
    export INSTA_USERNAME=your_username  # add to ~/.zshrc to persist

    # Download by hashtag:
    python3 download_images.py "#durabulk" --max 50

    # Download by profile:
    python3 download_images.py "@durabulk" --max 50

    # Or pass --login explicitly to override:
    python3 download_images.py "#durabulk" --login OTHER_USERNAME --max 50
"""

import argparse
import instaloader
import json
import os
import shutil
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(
        description="Download Instagram images by hashtag or profile."
    )
    parser.add_argument(
        "target",
        help='Hashtag (e.g. "#durabulk") or profile (e.g. "@durabulk")',
    )
    parser.add_argument(
        "--login",
        default=os.environ.get("INSTA_USERNAME"),
        help="Instagram username (default: $INSTA_USERNAME env var)",
    )
    parser.add_argument("--max", type=int, default=100, help="Max posts to download (default: 100)")
    parser.add_argument("--start", default="2025-01-01", help="Start date YYYY-MM-DD (default: 2025-01-01)")
    parser.add_argument("--end", default="2025-12-31", help="End date YYYY-MM-DD (default: 2025-12-31)")
    parser.add_argument("--output", default="images", help="Output directory (default: images)")
    args = parser.parse_args()

    if not args.login:
        print(
            "ERROR: No Instagram username provided.\n"
            "Either pass --login USERNAME or set the INSTA_USERNAME env var:\n"
            "  export INSTA_USERNAME=your_username  # add to ~/.zshrc to persist"
        )
        return

    os.makedirs(args.output, exist_ok=True)

    L = instaloader.Instaloader(
        download_videos=False,
        download_video_thumbnails=False,
        download_geotags=False,
        download_comments=False,
        save_metadata=False,
        compress_json=False,
        post_metadata_txt_pattern="",
    )

    # Load session file (created with: python3 -m instaloader --login USERNAME)
    print(f"Loading session for @{args.login}...")
    try:
        L.load_session_from_file(args.login)
        print("Session loaded successfully.")
    except FileNotFoundError:
        print(
            f"ERROR: Session file not found for '{args.login}'.\n"
            f"Create one first by running:\n"
            f"  python3 -m instaloader --login {args.login}"
        )
        return

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    target = args.target.strip()

    # Determine if target is a hashtag or profile
    if target.startswith("#"):
        hashtag = target.lstrip("#")
        print(f"Fetching posts from #{hashtag}...")
        posts = instaloader.Hashtag.from_name(L.context, hashtag).get_posts()
    elif target.startswith("@"):
        profile_name = target.lstrip("@")
        print(f"Fetching posts from @{profile_name}...")
        profile = instaloader.Profile.from_username(L.context, profile_name)
        posts = profile.get_posts()
    else:
        # Default to hashtag if no prefix
        print(f"Fetching posts from #{target}...")
        posts = instaloader.Hashtag.from_name(L.context, target).get_posts()

    image_files = []
    count = 0

    for post in posts:
        if count >= args.max:
            break
        post_date = post.date_utc
        if post_date.date() > end_dt.date():
            continue
        if post_date.date() < start_dt.date():
            break
        if post.is_video:
            continue

        filename = f"{post.date_utc.strftime('%Y%m%d_%H%M%S')}_{post.shortcode}.jpg"
        filepath = os.path.join(args.output, filename)

        if not os.path.exists(filepath):
            try:
                stem = os.path.splitext(filepath)[0]
                L.download_pic(stem, post.url, post.date_utc)
                for ext in [".jpg", ".jpeg", ".png", ".webp"]:
                    candidate = stem + ext
                    if os.path.exists(candidate) and candidate != filepath:
                        shutil.move(candidate, filepath)
                        break
            except Exception as e:
                print(f"  Skipped: {e}")
                continue

        if os.path.exists(filepath):
            image_files.append(filename)
            count += 1
            print(f"  [{count}/{args.max}] {filename}")

    print(f"\nDownloaded {len(image_files)} images to {args.output}/")

    # Generate image list JSON for the static site
    all_images = sorted(
        f for f in os.listdir(args.output)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    )

    with open("image-list.json", "w") as f:
        json.dump(all_images, f, indent=2)

    print(f"Wrote image-list.json with {len(all_images)} entries.")


if __name__ == "__main__":
    main()
