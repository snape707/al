import io
import json
from typing import Tuple, List
import numpy as np
from PIL import Image

def load_labels(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return labels

def preprocess_pil_image(img: Image.Image, target_size: int) -> np.ndarray:
    img = img.convert("RGB")
    img = img.resize((target_size, target_size))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, 3)
    return arr
