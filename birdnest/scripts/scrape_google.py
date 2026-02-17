#!/usr/bin/env python3
"""
scrape_google.py — Download bird nest images using icrawler (Bing backend).

Usage:
    python scrape_google.py --query "bird nest" --count 50 --output-dir dataset/birdnest
    python scrape_google.py --query "empty tree branch" --count 50 --output-dir dataset/not_birdnest

Suggested queries for bird nests:
    "bird nest", "bird nest in tree", "bird nest eggs", "bird nest close up"

Suggested queries for non-bird-nests:
    "empty tree branch", "basket weaving", "bowl on table", "tangled rope",
    "haystack", "bush no nest", "tree hollow"
"""

import argparse
from pathlib import Path

from icrawler.builtin import BingImageCrawler


def main():
    parser = argparse.ArgumentParser(description="Download images via Bing image search")
    parser.add_argument("--query", required=True, help="Search query string")
    parser.add_argument("--count", type=int, default=50, help="Number of images to download")
    parser.add_argument("--output-dir", required=True, help="Output directory for images")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    crawler = BingImageCrawler(
        storage={"root_dir": str(output)},
        log_level="WARNING",
    )
    crawler.crawl(keyword=args.query, max_num=args.count)

    count = len(list(output.glob("*")))
    print(f"Downloaded {count} images to {output}")


if __name__ == "__main__":
    main()
