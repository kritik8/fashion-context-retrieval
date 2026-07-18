import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from typing import List, Dict, Any, Tuple
from pathlib import Path

class GarmentDetector:
    """
    Zero-shot garment detector utilizing the Grounding DINO tiny model.
    Localizes clothing items based on a predefined text vocabulary.
    """
    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Initializing GarmentDetector on device: {self.device}")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        # Predefined text vocabulary for zero-shot object detection
        self.vocab = [
            "shirt", "t-shirt", "blazer", "jacket", "coat", 
            "raincoat", "hoodie", "dress", "pants", "jeans", 
            "skirt", "tie", "sweater"
        ]
        
        # Standard Grounding DINO prompt formatting
        # Join words with " . " and end with a period
        self.text_prompt = " . ".join(self.vocab) + " ."

    def detect_garments(self, image: Image.Image, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Detects garments in a PIL image.
        Returns a list of detections, where each detection is a dictionary:
        {
            'label': str,         # matched vocabulary garment label
            'box': [xmin, ymin, xmax, ymax],  # absolute coordinates
            'score': float        # confidence score
        }
        """
        width, height = image.size
        
        # Grounding DINO requires RGB image
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = self.processor(images=image, text=self.text_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Post-process detections
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=threshold,
            text_threshold=threshold,
            target_sizes=[(height, width)]
        )
        
        detections = []
        if not results:
            return detections
            
        result = results[0]
        scores = result["scores"].cpu().tolist()
        labels = result["labels"]  # List of string labels
        boxes = result["boxes"].cpu().tolist()
        
        for score, label, box in zip(scores, labels, boxes):
            cleaned_label = label.strip().lower()
            
            # Map the predicted label to our strict vocabulary list
            matched_vocab_label = None
            for vocab_item in self.vocab:
                if vocab_item in cleaned_label or cleaned_label in vocab_item:
                    matched_vocab_label = vocab_item
                    break
                    
            # If no direct match is found in our vocabulary, skip it to keep the index clean
            if not matched_vocab_label:
                continue
                
            # Clamp box coordinates to image dimensions
            xmin = max(0.0, float(box[0]))
            ymin = max(0.0, float(box[1]))
            xmax = min(float(width), float(box[2]))
            ymax = min(float(height), float(box[3]))
            
            # Avoid invalid empty boxes
            if xmax <= xmin or ymax <= ymin:
                continue
                
            detections.append({
                "label": matched_vocab_label,
                "box": [xmin, ymin, xmax, ymax],
                "score": float(score)
            })
            
        return detections

if __name__ == "__main__":
    # Quick self-test block
    import sys
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        if img_path.exists():
            img = Image.open(img_path)
            detector = GarmentDetector()
            dets = detector.detect_garments(img)
            print(f"Detected {len(dets)} garments:")
            for det in dets:
                print(f"  {det['label']}: score={det['score']:.3f}, box={[int(coord) for coord in det['box']]}")
        else:
            print(f"File not found: {img_path}")
