#!/usr/bin/env python3
"""
Script to replace the placeholder Picsum photos with real Fashionpedia dataset images
provided by the user in the 'test/' directory.
"""

import os
import csv
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "test"
FASHIONPEDIA_RAW_DIR = REPO_ROOT / "data" / "raw" / "fashionpedia"
METADATA_PATH = REPO_ROOT / "data" / "metadata.csv"

def main():
    if not TEST_DIR.exists():
        print(f"Error: test/ directory not found at {TEST_DIR}")
        return
        
    # Get list of images in test/
    test_images = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not test_images:
        print("No images found in test/ directory.")
        return
        
    print(f"Found {len(test_images)} images in test/ directory.")
    
    # We will select 150 images to copy to data/raw/fashionpedia/
    subset_size = min(150, len(test_images))
    selected_images = test_images[:subset_size]
    print(f"Selecting first {subset_size} images to import.")
    
    # Recreate the raw fashionpedia directory to remove placeholder picsum images
    if FASHIONPEDIA_RAW_DIR.exists():
        print("Clearing old placeholder images...")
        shutil.rmtree(FASHIONPEDIA_RAW_DIR)
    FASHIONPEDIA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy selected images
    copied_count = 0
    for img_name in selected_images:
        src_path = TEST_DIR / img_name
        dest_path = FASHIONPEDIA_RAW_DIR / img_name
        shutil.copy(src_path, dest_path)
        copied_count += 1
        
    print(f"Successfully copied {copied_count} real Fashionpedia images to {FASHIONPEDIA_RAW_DIR}")
    
    # Load and update metadata.csv
    rows = []
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Keep Unsplash and target queries, filter out old placeholder fashionpedia rows
                if row["source"] != "fashionpedia":
                    rows.append(row)
                    
    # Append the new real Fashionpedia images
    for img_name in selected_images:
        image_id = Path(img_name).stem
        rel_path = f"data/raw/fashionpedia/{img_name}"
        rows.append({
            "image_id": image_id,
            "source": "fashionpedia",
            "path": rel_path,
            "query_used": "fashionpedia_real_sample"
        })
        
    # Write back to metadata.csv
    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["image_id", "source", "path", "query_used"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Updated metadata.csv. Total registered images: {len(rows)}")

if __name__ == "__main__":
    main()
