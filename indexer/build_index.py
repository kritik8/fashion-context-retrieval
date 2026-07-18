import os
import csv
import json
import numpy as np
from pathlib import Path
from PIL import Image
import chromadb
from tqdm import tqdm
from typing import Optional

from indexer.detect import GarmentDetector
from indexer.embed import FashionCLIPEmbedder
from indexer.color import extract_dominant_color

# Path setup
REPO_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = REPO_ROOT / "data" / "metadata.csv"
CHROMA_DB_DIR = REPO_ROOT / "indexer" / "db" / "chroma"

def load_dataset_metadata() -> list:
    """Loads image paths and IDs from the metadata CSV file."""
    if not METADATA_PATH.exists():
        print(f"Metadata file not found at {METADATA_PATH}. Please run fetch scripts first.")
        return []
        
    images = []
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert relative path in csv to absolute path
            abs_path = REPO_ROOT / row["path"]
            images.append({
                "image_id": row["image_id"],
                "source": row["source"],
                "path": abs_path,
                "query_used": row["query_used"]
            })
    return images

def build_index(limit: Optional[int] = None):
    """Orchestrates the entire indexing process and stores data in ChromaDB."""
    images = load_dataset_metadata()
    if not images:
        print("No images found to index.")
        return
        
    if limit is not None:
        target_queries_set = {
            "person in bright yellow raincoat",
            "professional business attire inside modern office",
            "person wearing blue shirt sitting on park bench",
            "casual weekend outfit city walk",
            "red tie and white shirt formal setting",
            "man in formal suit office",
            "woman casual street style"
        }
        
        target_images = []
        fashionpedia_images = []
        systematic_images = []
        
        for img in images:
            query = img["query_used"].lower()
            is_target = (img["query_used"] in target_queries_set or 
                         any(q in query for q in ["yellow raincoat", "park bench", "red tie", "business attire", "target_"]))
            if is_target:
                target_images.append(img)
            elif img["source"] == "fashionpedia":
                fashionpedia_images.append(img)
            else:
                systematic_images.append(img)
                
        # Sample: up to 10 targets, 40 fashionpedia, and fill the rest (up to 150 total) from systematic
        selected_targets = target_images[:10]
        selected_fashionpedia = fashionpedia_images[:40]
        
        remaining_slots = max(0, limit - len(selected_targets) - len(selected_fashionpedia))
        selected_systematic = systematic_images[:remaining_slots]
        
        images = selected_targets + selected_fashionpedia + selected_systematic
        print(f"Constructed test database subset: {len(selected_targets)} targets, {len(selected_fashionpedia)} real Fashionpedia, {len(selected_systematic)} systematic images.")
        
    print(f"Loaded {len(images)} images from metadata.")
    
    # Initialize ChromaDB persistent client
    CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
    
    # Get or create collections
    # ChromaDB collections take cosine distance as metric (which matches CLIP cosine similarity)
    # Cosine distance = 1 - Cosine similarity.
    scenes_coll = chroma_client.get_or_create_collection(
        name="scenes", 
        metadata={"hnsw:space": "cosine"}
    )
    regions_coll = chroma_client.get_or_create_collection(
        name="regions", 
        metadata={"hnsw:space": "cosine"}
    )
    
    # Identify already indexed images for idempotence/resumability
    try:
        existing_scenes = scenes_coll.get(include=[])
        indexed_image_ids = set(existing_scenes["ids"])
        print(f"Found {len(indexed_image_ids)} images already indexed in ChromaDB. Resuming...")
    except Exception as e:
        print(f"Could not load existing index IDs: {e}. Starting fresh.")
        indexed_image_ids = set()
        
    # Initialize ML components
    detector = GarmentDetector()
    embedder = FashionCLIPEmbedder()
    
    processed_count = 0
    
    for idx, img_info in enumerate(images):
        img_id = img_info["image_id"]
        img_path = img_info["path"]
        
        # Skip if already indexed
        if img_id in indexed_image_ids:
            continue
            
        if not img_path.exists():
            print(f"Warning: Image file not found: {img_path}. Skipping.")
            continue
            
        try:
            # 1. Load image
            img = Image.open(img_path).convert("RGB")
            
            # 2. Detect clothing regions first
            detections = detector.detect_garments(img, threshold=0.3)
            
            # 3. Create all crops and extract dominant colors
            crops = [img] # Global image at index 0
            crop_infos = [] # (crop_idx, bbox, garment_label, detect_conf, dom_color)
            
            for crop_idx, det in enumerate(detections):
                bbox = det["box"]
                crop_img = img.crop(bbox)
                crops.append(crop_img)
                
                # Extract dominant color
                dom_color = extract_dominant_color(crop_img)
                crop_infos.append((crop_idx, bbox, det["label"], det["score"], dom_color))
                
            # 4. Batch compute all embeddings in a single CLIP forward pass
            all_embs = embedder.get_image_embeddings(crops)
            global_emb = all_embs[0]
            region_embs = all_embs[1:]
            
            # 5. Classify scene and style, then save to 'scenes' collection
            scene_l, scene_s, style_l, style_s = embedder.classify_global(global_emb)
            scenes_coll.add(
                ids=[img_id],
                embeddings=[global_emb.tolist()],
                metadatas=[{
                    "image_id": img_id,
                    "image_path": str(img_path.relative_to(REPO_ROOT)),
                    "scene_label": scene_l,
                    "scene_score": scene_s,
                    "style_label": style_l,
                    "style_score": style_s
                }]
            )
            
            # 6. Save region crops if any garments detected
            region_embeddings = []
            region_metadatas = []
            region_ids = []
            
            for sub_idx, (crop_idx, bbox, garment_label, detect_conf, dom_color) in enumerate(crop_infos):
                crop_id = f"{img_id}_crop_{crop_idx}"
                region_ids.append(crop_id)
                region_embeddings.append(region_embs[sub_idx].tolist())
                region_metadatas.append({
                    "image_id": img_id,
                    "image_path": str(img_path.relative_to(REPO_ROOT)),
                    "garment_label": garment_label,
                    "dominant_color": dom_color,
                    "bbox": json.dumps(bbox),
                    "detect_conf": float(detect_conf)
                })
                
            # Batch add regions if any detected
            if region_ids:
                regions_coll.add(
                    ids=region_ids,
                    embeddings=region_embeddings,
                    metadatas=region_metadatas
                )
                
            processed_count += 1
            if processed_count % 25 == 0 or idx == len(images) - 1:
                print(f"Progress: Processed {processed_count} new images. Total processed: {idx + 1}/{len(images)}.")
                
        except Exception as e:
            print(f"Error processing image {img_id} at {img_path}: {e}")
            import traceback
            traceback.print_exc()
            
    print("Indexing complete.")
    print(f"Total images in 'scenes' collection: {scenes_coll.count()}")
    print(f"Total garments in 'regions' collection: {regions_coll.count()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build database index.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to index")
    args = parser.parse_args()
    build_index(limit=args.limit)
