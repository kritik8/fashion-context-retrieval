import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from typing import List, Dict, Any, Tuple
from pathlib import Path

class FashionCLIPEmbedder:
    """
    FashionCLIP embedder utilizing patrickjohncyh/fashion-clip.
    Generates normalized image and text embeddings, and performs
    zero-shot classification for environmental context and clothing styles.
    """
    def __init__(self, model_id: str = "patrickjohncyh/fashion-clip", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing FashionCLIPEmbedder on device: {self.device}")
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        # Predefined labels
        self.scene_labels = ["office interior", "urban street", "park", "home interior"]
        self.style_labels = ["formal", "casual", "professional business attire", "casual weekend outfit"]
        
        # Cache the text embeddings for scene and style labels to boost processing speed
        self._precompute_text_caches()

    def _precompute_text_caches(self):
        """Precomputes and stores normalized text embeddings for the fixed vocab lists."""
        self.scene_features = self.get_text_embeddings(self.scene_labels)
        self.style_features = self.get_text_embeddings(self.style_labels)

    def get_image_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        """
        Computes L2-normalized image embeddings for a list of PIL Images.
        Returns a numpy array of shape (num_images, embedding_dim).
        """
        processed_images = []
        for img in images:
            if img.mode != "RGB":
                processed_images.append(img.convert("RGB"))
            else:
                processed_images.append(img)
                
        inputs = self.processor(images=processed_images, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            # L2 Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
        return image_features.cpu().numpy()

    def get_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Computes L2-normalized text embeddings for a list of query strings.
        Returns a numpy array of shape (num_texts, embedding_dim).
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            # L2 Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy()

    def classify_global(self, global_embedding: np.ndarray) -> Tuple[str, float, str, float]:
        """
        Performs zero-shot classification using pre-computed scene and style label embeddings.
        Returns (top_scene_label, scene_score, top_style_label, style_score).
        """
        # Ensure global_embedding has shape (1, embedding_dim)
        if len(global_embedding.shape) == 1:
            global_embedding = global_embedding[np.newaxis, :]
            
        # Cosine similarities (dot products since both are L2 normalized)
        scene_sims = np.dot(global_embedding, self.scene_features.T)[0]
        style_sims = np.dot(global_embedding, self.style_features.T)[0]
        
        # Get top indices
        top_scene_idx = np.argmax(scene_sims)
        top_style_idx = np.argmax(style_sims)
        
        return (
            self.scene_labels[top_scene_idx],
            float(scene_sims[top_scene_idx]),
            self.style_labels[top_style_idx],
            float(style_sims[top_style_idx])
        )

if __name__ == "__main__":
    # Quick self-test block
    import sys
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        if img_path.exists():
            img = Image.open(img_path)
            embedder = FashionCLIPEmbedder()
            
            # Embed global
            glob_emb = embedder.get_image_embeddings([img])[0]
            print(f"Global embedding shape: {glob_emb.shape}")
            
            # Classify
            scene_l, scene_s, style_l, style_s = embedder.classify_global(glob_emb)
            print(f"Scene classification: {scene_l} (score={scene_s:.3f})")
            print(f"Style classification: {style_l} (score={style_s:.3f})")
        else:
            print(f"File not found: {img_path}")
