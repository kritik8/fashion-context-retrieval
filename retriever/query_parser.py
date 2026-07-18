import re
from typing import Dict, List, Any, Optional
import numpy as np

# Standard vocabularies matching the indexer
GARMENTS = [
    "shirt", "t-shirt", "blazer", "jacket", "coat", 
    "raincoat", "hoodie", "dress", "pants", "jeans", 
    "skirt", "tie", "sweater"
]

COLORS = [
    "red", "blue", "yellow", "green", "white", "black", "beige", "navy", 
    "brown", "grey", "orange", "pink", "purple", "teal", "maroon", 
    "olive", "gold", "silver", "cream", "khaki"
]

# Synonyms dictionary mapping user terms to the standard indexer vocabulary
SYNONYMS = {
    "button-down": "shirt",
    "buttondown": "shirt",
    "blouse": "shirt",
    "trousers": "pants",
    "slacks": "pants",
    "suit": "blazer",
    "mac": "coat",
    "trench": "coat",
    "trenchcoat": "coat",
    "tee": "t-shirt",
    "tshirt": "t-shirt",
    "pullover": "sweater",
    "cardigan": "sweater",
    "blazers": "blazer",
    "jackets": "jacket",
    "coats": "coat",
    "raincoats": "raincoat",
    "hoodies": "hoodie",
    "dresses": "dress",
    "skirts": "skirt",
    "ties": "tie",
    "sweaters": "sweater"
}

def parse_query(query: str, embedder: Optional[Any] = None) -> Dict[str, Any]:
    """
    Parses a natural language query into structured garment claims and scene/style hints.
    Utilizes keyword/synonym matching and falls back to FashionCLIP semantic similarity.
    
    Returns a dictionary of the form:
    {
        "garment_claims": [{"garment": str, "color": Optional[str]}],
        "scene_hint": Optional[str],
        "style_hint": Optional[str]
    }
    """
    query_clean = query.strip().lower()
    
    # 1. Map synonyms to clean the query text
    for syn, vocab_word in SYNONYMS.items():
        query_clean = re.sub(rf"\b{syn}\b", vocab_word, query_clean)
        
    # Split query into words/tokens for proximity analysis
    tokens = re.findall(r"\b[\w-]+\b", query_clean)
    
    # Find all occurrences of colors and garments with their index in tokens list
    found_colors = []
    found_garments = []
    
    for idx, token in enumerate(tokens):
        if token in COLORS:
            found_colors.append((token, idx))
        if token in GARMENTS:
            found_garments.append((token, idx))
            
    # Pair colors and garments using proximity rules
    garment_claims = []
    paired_color_indices = set()
    
    for g_word, g_idx in found_garments:
        matched_color = None
        
        # Look back up to 3 tokens for a color (e.g. "bright yellow raincoat")
        for color, c_idx in found_colors:
            if c_idx in paired_color_indices:
                continue
            if 0 < g_idx - c_idx <= 3:
                matched_color = color
                paired_color_indices.add(c_idx)
                break
                
        # If no preceding color, look forward for patterns like "shirt in blue" or "jacket of black"
        if not matched_color:
            for color, c_idx in found_colors:
                if c_idx in paired_color_indices:
                    continue
                # check if there is "in", "of", "with", "wearing" between garment and color
                if 0 < c_idx - g_idx <= 3:
                    # check intermediate tokens
                    int_tokens = tokens[g_idx+1:c_idx]
                    if any(t in ["in", "of", "with", "wearing", "is"] for t in int_tokens):
                        matched_color = color
                        paired_color_indices.add(c_idx)
                        break
                        
        garment_claims.append({
            "garment": g_word,
            "color": matched_color
        })
        
    # 2. Extract scene and style hints via substring rule-based matching
    scene_hint = None
    style_hint = None
    
    # Substring scene matching
    if "office" in query_clean:
        scene_hint = "office interior"
    elif any(x in query_clean for x in ["street", "city", "urban", "walk", "sidewalk"]):
        scene_hint = "urban street"
    elif any(x in query_clean for x in ["park", "bench", "garden", "lawn", "forest"]):
        scene_hint = "park"
    elif any(x in query_clean for x in ["home", "house", "room", "apartment", "living", "interior"]):
        scene_hint = "home interior"
        
    # Substring style matching
    if "formal" in query_clean or "suit" in query_clean or "business attire" in query_clean:
        style_hint = "professional business attire"
    elif "casual" in query_clean or "weekend" in query_clean:
        style_hint = "casual weekend outfit"
    elif "professional" in query_clean:
        style_hint = "professional business attire"
        
    # 3. Fallback to FashionCLIP embeddings if substring checks failed and embedder is available
    if embedder is not None:
        try:
            # Embed query text
            query_emb = embedder.get_text_embeddings([query])[0]
            
            if not scene_hint:
                # Compare query text embedding with precomputed scene label features
                scene_sims = np.dot(query_emb, embedder.scene_features.T)
                scene_hint = embedder.scene_labels[np.argmax(scene_sims)]
                
            if not style_hint:
                # Compare query text embedding with precomputed style label features
                style_sims = np.dot(query_emb, embedder.style_features.T)
                style_hint = embedder.style_labels[np.argmax(style_sims)]
        except Exception as e:
            print(f"Warning: Failed embedding-based query parsing fallback: {e}")
            
    # Default fallbacks if both rules and embeddings fail (e.g. no embedder)
    if not scene_hint:
        # Default to a safe context or None
        scene_hint = None
    if not style_hint:
        style_hint = None
        
    return {
        "garment_claims": garment_claims,
        "scene_hint": scene_hint,
        "style_hint": style_hint
    }

if __name__ == "__main__":
    # Test cases
    queries = [
        "a red tie and white shirt",
        "bright yellow raincoat in a park",
        "professional business attire in a modern office",
        "someone wearing a blue shirt sitting on a park bench",
        "casual weekend outfit for a city walk",
        "a person with trousers and a button-down blouse"
    ]
    for q in queries:
        print(f"Query: '{q}'")
        parsed = parse_query(q)
        print(f"  Parsed: {parsed}")
