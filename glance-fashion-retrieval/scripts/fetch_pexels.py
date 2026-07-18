#!/usr/bin/env python3
"""
Script to fetch images from the Unsplash API to build a fashion dataset.
Loads API keys from the .env file.
Queries Unsplash using a matrix of terms: environment x clothing type x color.
"""

import os
import sys
import csv
import time
import urllib.parse
from pathlib import Path
from typing import Dict, List, Set
import requests
from tqdm import tqdm

# Define path configurations
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_PATH = DATA_DIR / "metadata.csv"

# Matrix definition for queries
ENVIRONMENTS = ["office", "urban street", "park", "home"]
COLORS = ["red", "blue", "yellow", "green", "white", "black", "beige", "navy", "brown", "grey", "orange", "pink"]

# Specific clothing category items mapping to the three requested groups: formal, casual, outerwear
CLOTHING_CATEGORIES = {
    "formal": ["business suit", "blazer", "formal wear"],
    "casual": ["hoodie", "t-shirt", "sweater"],
    "outerwear": ["raincoat", "jacket", "coat"]
}

def load_env():
    """Manually parses .env file to load variables into os.environ."""
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        print(f"Loading environment from {env_path}")
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

def generate_queries() -> List[str]:
    """Generates compound queries combining environment, clothing category, and color."""
    queries = []
    
    # 1. Systematic matrix generation
    import random
    random.seed(42)
    
    # Generate combinations
    for env in ENVIRONMENTS:
        for cat, garments in CLOTHING_CATEGORIES.items():
            for garment in garments:
                # pick a subset of colors randomly to avoid generating too many queries
                selected_colors = random.sample(COLORS, k=3)
                for color in selected_colors:
                    query = f"{color} {garment} in {env}"
                    queries.append(query)
                    
    # 2. Add specific target queries to ensure evaluation patterns are covered in dataset
    target_queries = [
        "person in bright yellow raincoat",
        "professional business attire inside modern office",
        "person wearing blue shirt sitting on park bench",
        "casual weekend outfit city walk",
        "red tie and white shirt formal setting",
        "man in formal suit office",
        "woman casual street style"
    ]
    queries.extend(target_queries)
    
    # Deduplicate queries
    unique_queries = list(dict.fromkeys(queries))
    return unique_queries

def fetch_and_save_images(access_key: str, queries: List[str], target_total: int = 500, images_per_query: int = 12):
    """Fetches images from Unsplash API, saves them, and logs to metadata.csv."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    headers = {"Authorization": f"Client-ID {access_key}"}
    downloaded_ids: Set[str] = set()
    
    # Check what is already downloaded (to resume)
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    downloaded_ids.add(row["image_id"])
            print(f"Loaded existing metadata: {len(downloaded_ids)} images already logged.")
        except Exception as e:
            print(f"Error reading existing metadata: {e}. Will overwrite.")

    pbar = tqdm(total=target_total, desc="Downloading images")
    pbar.update(len(downloaded_ids))
    
    csv_file_exists = METADATA_PATH.exists()
    
    with open(METADATA_PATH, "a" if csv_file_exists else "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_id", "source", "path", "query_used"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not csv_file_exists:
            writer.writeheader()
            
        for query in queries:
            if len(downloaded_ids) >= target_total:
                print("\nReached target image count.")
                break
                
            print(f"\nQuerying Unsplash: '{query}'")
            encoded_query = urllib.parse.quote(query)
            url = f"https://api.unsplash.com/search/photos?query={encoded_query}&per_page={images_per_query}"
            
            try:
                response = requests.get(url, headers=headers, timeout=15)
                
                # Check for rate limiting
                if response.status_code == 403:
                    print("Forbidden/Rate limited by Unsplash API. Sleeping 30s...")
                    time.sleep(30)
                    response = requests.get(url, headers=headers, timeout=15)
                    
                if response.status_code != 200:
                    print(f"Failed to query '{query}': HTTP {response.status_code}")
                    continue
                    
                data = response.json()
                photos = data.get("results", [])
                
                for photo in photos:
                    if len(downloaded_ids) >= target_total:
                        break
                        
                    photo_id = str(photo["id"])
                    if photo_id in downloaded_ids:
                        continue
                        
                    # Get regular sized image URL
                    src_urls = photo.get("urls", {})
                    img_url = src_urls.get("regular") or src_urls.get("small")
                    if not img_url:
                        continue
                        
                    try:
                        img_response = requests.get(img_url, timeout=15)
                        if img_response.status_code == 200:
                            img_path = RAW_DIR / f"{photo_id}.jpg"
                            img_path.write_bytes(img_response.content)
                            
                            rel_path = f"data/raw/{photo_id}.jpg"
                            writer.writerow({
                                "image_id": photo_id,
                                "source": "unsplash",
                                "path": rel_path,
                                "query_used": query
                            })
                            f.flush()
                            
                            downloaded_ids.add(photo_id)
                            pbar.update(1)
                            # Wait a bit to avoid hitting rate limits
                            time.sleep(0.5)
                        else:
                            print(f"Failed to download image {photo_id}: HTTP {img_response.status_code}")
                    except Exception as e:
                        print(f"Error downloading image {photo_id}: {e}")
                        
            except Exception as e:
                print(f"Network error searching '{query}': {e}")
                time.sleep(2)
                
    pbar.close()
    print(f"Finished Unsplash collection. Total images: {len(downloaded_ids)}")

def main():
    load_env()
    access_key = os.environ.get("UNSPLASH_ACCESS_KEY")
    if not access_key:
        print("ERROR: UNSPLASH_ACCESS_KEY environment variable is not set.", file=sys.stderr)
        print("Please check your .env file and run again.", file=sys.stderr)
        sys.exit(1)
        
    queries = generate_queries()
    print(f"Generated {len(queries)} compound queries.")
    
    # We want to pull ~400 images from Unsplash
    fetch_and_save_images(access_key, queries, target_total=400, images_per_query=10)

if __name__ == "__main__":
    main()
