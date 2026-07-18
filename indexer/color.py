import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from typing import Tuple, Dict

# Named CSS3-ish color palette with RGB values
COLOR_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "red": (220, 20, 60),        # Crimson / Red
    "blue": (30, 144, 255),      # Dodger Blue
    "yellow": (255, 215, 0),     # Gold / Yellow
    "green": (34, 139, 34),      # Forest Green
    "white": (250, 250, 250),    # Off-white
    "black": (15, 15, 15),       # Near black
    "beige": (245, 245, 220),    # Beige
    "navy": (0, 0, 128),         # Navy Blue
    "brown": (139, 69, 19),      # Saddle Brown
    "grey": (128, 128, 128),     # Grey
    "orange": (255, 140, 0),     # Dark Orange
    "pink": (255, 192, 203),     # Pink
    "purple": (128, 0, 128),     # Purple
    "teal": (0, 128, 128),       # Teal
    "maroon": (128, 0, 0),       # Maroon
    "olive": (128, 128, 0),      # Olive
    "gold": (218, 165, 32),      # Goldenrod
    "silver": (192, 192, 192),   # Silver
    "cream": (255, 253, 208),    # Cream
    "khaki": (240, 230, 140)     # Khaki
}

def extract_dominant_color(crop: Image.Image) -> str:
    """
    Extracts the dominant color of a garment crop using K-Means (k=3).
    Maps the dominant RGB cluster center to the closest named color in the palette.
    """
    # 1. Resize image to 50x50 to speed up K-means computation significantly
    resized_crop = crop.resize((50, 50), Image.Resampling.BILINEAR)
    img_arr = np.array(resized_crop)
    
    # Ensure it's RGB
    if len(img_arr.shape) == 2:  # Grayscale
        img_arr = np.stack([img_arr] * 3, axis=-1)
    elif img_arr.shape[2] == 4:  # RGBA
        img_arr = img_arr[:, :, :3]
        
    pixels = img_arr.reshape(-1, 3)
    
    # 2. Run K-Means with 3 clusters
    kmeans = KMeans(n_clusters=3, n_init=1, random_state=42)
    kmeans.fit(pixels)
    
    # 3. Find the most frequent cluster label
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    counts = np.bincount(labels)
    dominant_cluster_idx = np.argmax(counts)
    dominant_rgb = centers[dominant_cluster_idx]
    
    # Convert dominant_rgb elements to integers
    dom_r, dom_g, dom_b = int(dominant_rgb[0]), int(dominant_rgb[1]), int(dominant_rgb[2])
    
    # 4. Map the dominant RGB to the nearest named color using Euclidean distance
    closest_color_name = "grey"
    min_distance = float("inf")
    
    for name, palette_rgb in COLOR_PALETTE.items():
        # Weighted Euclidean distance to align closer to human perception:
        # Red is weighted slightly more, green is weighted more, blue is weighted less
        # Formula: d = sqrt( 2*dR^2 + 4*dG^2 + 3*dB^2 )
        # A simpler standard Euclidean distance is also robust, but weighted helps
        dr = dom_r - palette_rgb[0]
        dg = dom_g - palette_rgb[1]
        db = dom_b - palette_rgb[2]
        
        dist = np.sqrt(2 * (dr**2) + 4 * (dg**2) + 3 * (db**2))
        
        if dist < min_distance:
            min_distance = dist
            closest_color_name = name
            
    return closest_color_name

if __name__ == "__main__":
    # Quick self-test block
    import sys
    from pathlib import Path
    if len(sys.argv) > 1:
        img_path = Path(sys.argv[1])
        if img_path.exists():
            img = Image.open(img_path)
            dom_color = extract_dominant_color(img)
            print(f"Dominant color: {dom_color}")
        else:
            print(f"File not found: {img_path}")
