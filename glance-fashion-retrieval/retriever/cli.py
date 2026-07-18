import os
import argparse
import re
import shutil
import json
from pathlib import Path

from retriever.search import FashionSearcher

REPO_ROOT = Path(__file__).resolve().parent.parent

def generate_slug(text: str) -> str:
    """Generates a URL-safe, filesystem-safe lowercase slug from text."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")

def run_cli():
    parser = argparse.ArgumentParser(description="Multimodal Fashion & Context Retrieval CLI Search Engine")
    parser.add_argument("query", type=str, help="Natural language search query")
    parser.add_argument("--k", type=int, default=5, help="Number of top results to retrieve")
    parser.add_argument("--device", type=str, default=None, help="Device to run inference on (cuda/cpu)")
    args = parser.parse_args()
    
    print(f"Query: '{args.query}' (k={args.k})")
    
    # Initialize and execute search
    searcher = FashionSearcher(device=args.device)
    results = searcher.search(args.query, k=args.k)
    
    if not results:
        print("No results returned.")
        return
        
    # Set up directory for saving results
    slug = generate_slug(args.query)
    results_dir = REPO_ROOT / "eval" / "results" / slug
    
    # Clean previous results if folder exists, then create fresh
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(f"SEARCH RESULTS FOR: '{args.query}'")
    print("=" * 80)
    
    console_data = []
    
    for idx, res in enumerate(results):
        rank = idx + 1
        img_id = res["image_id"]
        rel_path = res["image_path"]
        abs_img_path = REPO_ROOT / rel_path
        score = res["final_score"]
        
        # Determine output filename
        # e.g., rank_1_score_0.842_12345.jpg
        dst_filename = f"rank_{rank}_score_{score:.3f}_{img_id}.jpg"
        dst_path = results_dir / dst_filename
        
        # Copy image file (safer than symlinking on Windows which requires elevated rights)
        if abs_img_path.exists():
            shutil.copy(abs_img_path, dst_path)
            copy_status = "Copied to results folder"
        else:
            copy_status = "Original file missing"
            
        # Formulate detail string
        scene_str = f"{res['scene_label']} ({res['scene_score']:.3f})"
        style_str = f"{res['style_label']} ({res['style_score']:.3f})"
        
        garment_breakdown = ", ".join(f"{k}: {v:.3f}" for k, v in res["garment_scores"].items()) if res["garment_scores"] else "N/A"
        
        print(f"Rank {rank} | Score: {score:.4f} | ID: {img_id}")
        print(f"  Path: {rel_path} ({copy_status})")
        print(f"  Context: Scene='{scene_str}' | Style='{style_str}'")
        if res["garment_scores"]:
            print(f"  Garment Scores: {garment_breakdown}")
        if res.get("context_score", 0.0) > 0.0:
            print(f"  Context score: {res['context_score']:.3f}")
        if res.get("fallback_score", 0.0) > 0.0:
            print(f"  Global fallback score: {res['fallback_score']:.3f}")
        print("-" * 80)
        
        console_data.append({
            "rank": rank,
            "image_id": img_id,
            "path": str(rel_path),
            "final_score": score,
            "garment_scores": res["garment_scores"],
            "context_score": res.get("context_score", 0.0),
            "fallback_score": res.get("fallback_score", 0.0),
            "scene_label": res["scene_label"],
            "scene_score": res["scene_score"],
            "style_label": res["style_label"],
            "style_score": res["style_score"]
        })
        
    # Write metadata results.json in the folder
    with open(results_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(console_data, f, indent=2)
        
    print(f"\nSaved top-{len(results)} images and metadata json to: {results_dir.relative_to(REPO_ROOT)}")

if __name__ == "__main__":
    run_cli()
