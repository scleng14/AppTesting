# landmark.py
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import streamlit as st

# 地标关键字与坐标映射
LANDMARK_KEYWORDS = {
    "petronas towers": ("Petronas Twin Towers", "Kuala Lumpur", 3.1579, 101.7116),
    "kl tower": ("KL Tower", "Kuala Lumpur", 3.1528, 101.7037),
    "pyramid sunway": ("Sunway Pyramid", "Selangor", 3.0731, 101.6078),
    "penang bridge": ("Penang Bridge", "Penang", 5.3363, 100.3076),
    "malacca straits mosque": ("Malacca Straits Mosque", "Melaka", 2.1896, 102.2501),
}

CLIP_MODEL_NAME = "geolocal/StreetCLIP"
CLIP_THRESHOLD = 0.6

@st.cache_resource
def load_clip():
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return model, processor

clip_model, clip_processor = load_clip()

def detect_landmark(image: Image.Image):
    try:
        inputs = clip_processor(
            text=list(LANDMARK_KEYWORDS.keys()),
            images=image,
            return_tensors="pt",
            padding=True
        )
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze()
        max_prob = torch.max(probs).item()
        if max_prob > CLIP_THRESHOLD:
            match = list(LANDMARK_KEYWORDS.keys())[torch.argmax(probs).item()]
            coords = LANDMARK_KEYWORDS[match][2:]
            return coords
        return None
    except Exception:
        return None
