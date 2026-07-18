#!/usr/bin/env python3
"""
Helper script to fetch high-quality images matching the official evaluation queries.
Appends them to data/metadata.csv so they are prioritized and indexed.
"""

import os
import csv
import sys
from pathlib import Path
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_PATH = DATA_DIR / "metadata.csv"

EVAL_TARGETS = [
    ("yellow raincoat", "https://images.unsplash.com/photo-1548883354-7622d03aca27?w=1080"), # person in yellow raincoat
    ("yellow raincoat", "https://images.unsplash.com/photo-1508873696983-2df519f0397e?w=1080"), # person in yellow raincoat
    ("business attire office", "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab?w=1080"), # professional attire office
    ("business attire office", "https://images.unsplash.com/photo-1507679799987-c73779587ccf?w=1080"), # professional attire office
    ("blue shirt park bench", "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?w=1080"), # person on bench in blue shirt
    ("blue shirt park bench", "https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?w=1080"), # man on bench in shirt
    ("casual weekend city walk", "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=1080"), # casual weekend outfit city walk
    ("casual weekend city walk", "https://images.unsplash.com/photo-1492562080023-ab3db95bfbce?w=1080"), # casual weekend outfit city walk
    ("red tie white shirt formal", "https://images.unsplash.com/photo-1593030103066-0093718efeb9?w=1080"), # red tie and white shirt
    ("red tie white shirt formal", "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=1080")  # red tie and white shirt
]

def load_env():
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

def fetch_targets():
    load_env()
    access_key = os.environ.get("UNSPLASH_ACCESS_KEY")
    if not access_key:
        print("ERROR: UNSPLASH_ACCESS_KEY is not set.")
        sys.exit(1)
        
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read existing IDs
    existing_ids = set()
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_ids.add(row["image_id"])
                
    headers = {"Authorization": f"Client-ID {access_key}"}
    added_count = 0
    
    with open(METADATA_PATH, "a", newline="", encoding="utf-8") as f:
        fieldnames = ["image_id", "source", "path", "query_used"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # We search Unsplash directly for these specific queries to get accurate target images
        for query_tag, fallback_url in EVAL_TARGETS:
            print(f"Fetching best matches for target: '{query_tag}'")
            url = f"https://api.unsplash.com/search/photos?query={query_tag}&per_page=3"
            
            try:
                res = requests.get(url, headers=headers, timeout=15)
                if res.status_code == 200:
                    results = res.json().get("results", [])
                    for photo in results:
                        photo_id = str(photo["id"])
                        if photo_id in existing_ids:
                            continue
                            
                        img_url = photo["urls"]["regular"]
                        img_res = requests.get(img_url, timeout=15)
                        if img_res.status_code == 200:
                            img_path = RAW_DIR / f"{photo_id}.jpg"
                            img_path.write_bytes(img_res.content)
                            
                            rel_path = f"data/raw/{photo_id}.jpg"
                            writer.writerow({
                                "image_id": photo_id,
                                "source": "unsplash_target",
                                "path": rel_path,
                                "query_used": f"target_{query_tag}"
                            })
                            f.flush()
                            existing_ids.add(photo_id)
                            added_count += 1
                            print(f"  Successfully added image {photo_id} for target '{query_tag}'")
                            break # We only need 1 per query, so break and move to next URL
                else:
                    print(f"  Failed Unsplash search for {query_tag}, using fallback...")
                    # Fallback URL download
                    photo_id = fallback_url.split("/")[-1].split("?")[0]
                    if photo_id not in existing_ids:
                        img_res = requests.get(fallback_url, timeout=15)
                        if img_res.status_code == 200:
                            img_path = RAW_DIR / f"{photo_id}.jpg"
                            img_path.write_bytes(img_res.content)
                            
                            rel_path = f"data/raw/{photo_id}.jpg"
                            writer.writerow({
                                "image_id": photo_id,
                                "source": "unsplash_target",
                                "path": rel_path,
                                "query_used": f"target_{query_tag}"
                            })
                            f.flush()
                            existing_ids.add(photo_id)
                            added_count += 1
                            print(f"  Successfully added fallback image {photo_id} for target '{query_tag}'")
            except Exception as e:
                print(f"  Error fetching target {query_tag}: {e}")
                
    print(f"Done. Added {added_count} target images to metadata.csv.")

if __name__ == "__main__":
    fetch_targets()
