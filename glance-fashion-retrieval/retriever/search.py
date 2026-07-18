import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import chromadb
import numpy as np

from indexer.embed import FashionCLIPEmbedder
from retriever.query_parser import parse_query

REPO_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DB_DIR = REPO_ROOT / "indexer" / "db" / "chroma"

class FashionSearcher:
    """
    Search system that parses queries, queries ChromaDB collections,
    and computes a hybrid score combining garment region and global scene matches.
    """
    def __init__(self, device: str = None):
        # Initialize the persistent client and the embedder
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
        self.embedder = FashionCLIPEmbedder(device=device)
        
        # Load or check collections
        try:
            self.scenes_coll = self.chroma_client.get_collection(name="scenes")
            self.regions_coll = self.chroma_client.get_collection(name="regions")
        except Exception as e:
            print(f"Error loading ChromaDB collections: {e}")
            print("Please run build_index.py first to populate the collections.")
            self.scenes_coll = None
            self.regions_coll = None

    def search(self, query_str: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes hybrid search scoring:
        1. Parse query to extract garment claims (garment + color) and context hints.
        2. If claims exist:
           a. Compute similarity for each garment claim (using MAX matching region + color bonus).
           b. Compute context similarity (scene + style hints against global embedding).
           c. Hybrid rank candidate images.
        3. If no claims are parsed, fallback to 100% global similarity with raw query.
        """
        if not self.scenes_coll or not self.regions_coll:
            print("Search collections are unpopulated or uninitialized.")
            return []
            
        total_images = self.scenes_coll.count()
        if total_images == 0:
            print("No images found in database index.")
            return []
            
        # Initialize our universe of candidate images
        scenes_data = self.scenes_coll.get(include=["metadatas"])
        image_universe = {}
        for idx, metadata in enumerate(scenes_data["metadatas"]):
            img_id = metadata["image_id"]
            image_universe[img_id] = {
                "image_id": img_id,
                "image_path": metadata["image_path"],
                "scene_label": metadata["scene_label"],
                "scene_score": float(metadata["scene_score"]),
                "style_label": metadata["style_label"],
                "style_score": float(metadata["style_score"]),
                "garment_scores": {},   # Score per parsed garment-color claim
                "context_score": 0.0,
                "final_score": 0.0,
                "fallback_score": 0.0
            }
            
        # Parse query with embedder fallback enabled
        parsed_query = parse_query(query_str, embedder=self.embedder)
        garment_claims = parsed_query["garment_claims"]
        scene_hint = parsed_query["scene_hint"]
        style_hint = parsed_query["style_hint"]
        
        has_garment_claims = len(garment_claims) > 0
        has_context_hints = (scene_hint is not None) or (style_hint is not None)
        
        # Clever Bit #1: Max-similarity per-region matching
        # When garment claims are present, we query region-by-region and select the MAX score for each candidate image
        # This prevents color mismatching (e.g. searching for red shirt + blue pants and getting blue shirt + red pants)
        if has_garment_claims:
            garment_claim_keys = []
            for claim in garment_claims:
                garment = claim["garment"]
                color = claim["color"]
                
                # Format string to embed: "color garment" or just "garment"
                claim_text = f"{color} {garment}" if color else garment
                claim_key = f"{color or ''} {garment}".strip()
                garment_claim_keys.append(claim_key)
                
                # Get FashionCLIP embedding for this target garment text claim
                claim_emb = self.embedder.get_text_embeddings([claim_text])[0]
                
                # Query region crops that match this specific garment label
                # This limits the search scope to appropriate garments, avoiding false matches on scene background
                try:
                    region_results = self.regions_coll.query(
                        query_embeddings=[claim_emb.tolist()],
                        n_results=min(1000, self.regions_coll.count()),
                        where={"garment_label": garment}
                    )
                    
                    if region_results and region_results["ids"] and region_results["ids"][0]:
                        matched_ids = region_results["ids"][0]
                        distances = region_results["distances"][0]
                        metadatas = region_results["metadatas"][0]
                        
                        for m_id, dist, meta in zip(matched_ids, distances, metadatas):
                            cand_img_id = meta["image_id"]
                            # Cosine similarity is 1.0 - cosine distance
                            sim = 1.0 - float(dist)
                            
                            # Dominant color exact matching bonus (+0.1)
                            color_bonus = 0.0
                            if color and meta.get("dominant_color") == color:
                                color_bonus = 0.10
                                
                            total_sim = min(1.0, sim + color_bonus)
                            
                            # Keep only the MAX similarity score among all matching garment crops in the same image
                            if cand_img_id in image_universe:
                                current_max = image_universe[cand_img_id]["garment_scores"].get(claim_key, 0.0)
                                image_universe[cand_img_id]["garment_scores"][claim_key] = max(current_max, total_sim)
                except Exception as e:
                    print(f"Error querying garment region '{claim_key}': {e}")
                    
            # Set default 0.0 scores for candidate images that didn't match specific garments
            for img_id in image_universe:
                for key in garment_claim_keys:
                    if key not in image_universe[img_id]["garment_scores"]:
                        image_universe[img_id]["garment_scores"][key] = 0.0
                        
        # Get scene/style context score
        if has_context_hints:
            context_query = " ".join(filter(None, [scene_hint, style_hint]))
            context_emb = self.embedder.get_text_embeddings([context_query])[0]
            
            try:
                context_results = self.scenes_coll.query(
                    query_embeddings=[context_emb.tolist()],
                    n_results=min(1000, total_images)
                )
                
                if context_results and context_results["ids"] and context_results["ids"][0]:
                    matched_ids = context_results["ids"][0]
                    distances = context_results["distances"][0]
                    
                    for cand_img_id, dist in zip(matched_ids, distances):
                        if cand_img_id in image_universe:
                            # Cosine similarity
                            image_universe[cand_img_id]["context_score"] = 1.0 - float(dist)
            except Exception as e:
                print(f"Error querying context: {e}")

        # Clever Bit #2: Compositional scoring formula
        # We blend the localized garment claims and the global environmental context
        if has_garment_claims:
            for img_id, item in image_universe.items():
                g_avg = sum(item["garment_scores"].values()) / len(item["garment_scores"])
                
                if has_context_hints:
                    # Combined weight: 60% garment similarity, 40% scene context
                    item["final_score"] = 0.6 * g_avg + 0.4 * item["context_score"]
                else:
                    item["final_score"] = g_avg
        elif has_context_hints:
            # Only context hints matched
            for img_id, item in image_universe.items():
                item["final_score"] = item["context_score"]
        else:
            # Clever Bit #3: Zero-shot fallback path
            # If the parser finds no structured clothing or style words, we use the original natural language query 
            # and compare it directly to the global image embedding (pure zero-shot CLIP fallback).
            # This ensures that query terms like "person sitting on a park bench in sunshine" still retrieve relevant images.
            raw_query_emb = self.embedder.get_text_embeddings([query_str])[0]
            try:
                fallback_results = self.scenes_coll.query(
                    query_embeddings=[raw_query_emb.tolist()],
                    n_results=min(1000, total_images)
                )
                
                if fallback_results and fallback_results["ids"] and fallback_results["ids"][0]:
                    matched_ids = fallback_results["ids"][0]
                    distances = fallback_results["distances"][0]
                    
                    for cand_img_id, dist in zip(matched_ids, distances):
                        if cand_img_id in image_universe:
                            sim = 1.0 - float(dist)
                            image_universe[cand_img_id]["final_score"] = sim
                            image_universe[cand_img_id]["fallback_score"] = sim
            except Exception as e:
                print(f"Error executing zero-shot fallback: {e}")
                
        # Sort candidate images by final score descending
        sorted_candidates = sorted(
            image_universe.values(),
            key=lambda x: x["final_score"],
            reverse=True
        )
        
        # Return top k results
        return sorted_candidates[:k]

if __name__ == "__main__":
    import sys
    searcher = FashionSearcher()
    
    query = sys.argv[1] if len(sys.argv) > 1 else "A person in a bright yellow raincoat."
    print(f"Running search for: '{query}'")
    
    results = searcher.search(query, k=3)
    for idx, res in enumerate(results):
        print(f"\n[{idx+1}] Score: {res['final_score']:.4f} (ID: {res['image_id']})")
        print(f"    Path: {res['image_path']}")
        print(f"    Scene: {res['scene_label']} ({res['scene_score']:.3f}) | Style: {res['style_label']} ({res['style_score']:.3f})")
        if res['garment_scores']:
            print(f"    Garment breakdowns: { {k: f'{v:.3f}' for k, v in res['garment_scores'].items()} }")
        if res['context_score'] > 0:
            print(f"    Context score: {res['context_score']:.3f}")
        if res['fallback_score'] > 0:
            print(f"    Fallback score: {res['fallback_score']:.3f}")
