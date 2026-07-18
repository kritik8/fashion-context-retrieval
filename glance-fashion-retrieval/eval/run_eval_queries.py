import os
import re
import shutil
import json
from pathlib import Path
from tabulate import tabulate

from retriever.search import FashionSearcher

REPO_ROOT = Path(__file__).resolve().parent.parent

EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting."
]

def generate_slug(text: str) -> str:
    """Generates a URL-safe, filesystem-safe lowercase slug from text."""
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")

def run_evaluation():
    print("=" * 80)
    print("RUNNING MULTIMODAL FASHION & CONTEXT RETRIEVAL EVALUATION")
    print("=" * 80)
    
    # Initialize searcher
    searcher = FashionSearcher()
    
    # Check if index is empty
    if not searcher.scenes_coll or searcher.scenes_coll.count() == 0:
        print("\n[ERROR] Search collections are empty or missing. Please ensure database indexing has run successfully.")
        return
        
    all_table_data = []
    
    for q_idx, query in enumerate(EVAL_QUERIES):
        print(f"\nProcessing Query {q_idx+1}/{len(EVAL_QUERIES)}: '{query}'")
        results = searcher.search(query, k=5)
        
        slug = generate_slug(query)
        results_dir = REPO_ROOT / "eval" / "results" / slug
        
        # Clean previous query results and recreate
        if results_dir.exists():
            shutil.rmtree(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        query_records = []
        
        for idx, res in enumerate(results):
            rank = idx + 1
            img_id = res["image_id"]
            rel_path = res["image_path"]
            abs_path = REPO_ROOT / rel_path
            score = res["final_score"]
            
            # File copy destination
            dst_filename = f"rank_{rank}_score_{score:.3f}_{img_id}.jpg"
            dst_path = results_dir / dst_filename
            
            if abs_path.exists():
                shutil.copy(abs_path, dst_path)
            
            # Record for JSON
            query_records.append({
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
            
            # Format row for print table
            # Shorten query for tabular view if needed
            q_short = query if len(query) < 40 else query[:37] + "..."
            
            # Garment breakdown string
            garments_str = ", ".join(f"{k}({v:.2f})" for k, v in res["garment_scores"].items()) if res["garment_scores"] else "N/A"
            
            all_table_data.append([
                q_short,
                rank,
                img_id,
                f"{score:.4f}",
                res["scene_label"],
                garments_str,
                str(Path(rel_path).name)
            ])
            
        # Save JSON metadata for the query
        with open(results_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(query_records, f, indent=2)
            
        print(f"  Query {q_idx+1} complete. Saved 5 results to eval/results/{slug}/")

    # Print summary table
    print("\n" + "=" * 100)
    print("EVALUATION RUN SUMMARY TABLE")
    print("=" * 100)
    
    headers = ["Query", "Rank", "Image ID", "Final Score", "Detected Scene", "Garment Scores", "File Name"]
    print(tabulate(all_table_data, headers=headers, tablefmt="grid"))
    print("=" * 100)
    print("Evaluation completed successfully.")

if __name__ == "__main__":
    run_evaluation()
