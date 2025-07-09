import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import streamlit as st

LANDMARK_KEYWORDS = {
    "petronas towers": ("Petronas Twin Towers", "Kuala Lumpur", 3.1579, 101.7116),
    "kl tower": ("KL Tower", "Kuala Lumpur", 3.1528, 101.7039),
    "georgetown mural": ("Georgetown Street Art", "Penang", 5.4170, 100.3380),
    "putrajaya mosque": ("Putra Mosque", "Putrajaya", 2.9360, 101.6910),
    "malacca dutch square": ("Dutch Square", "Malacca", 2.1944, 102.2496),
    "komtar": ("Komtar Tower", "Penang", 5.4143, 100.3288),
    "a famosa": ("A Famosa", "Melaka", 2.1912, 102.2501),
    "sabah state mosque": ("Sabah State Mosque", "Kota Kinabalu", 5.9576, 116.0654),
    "genting highlands": ("Genting Highlands", "Pahang", 3.4221, 101.7934)
}

CLIP_MODEL_NAME = "geolocal/StreetCLIP"
CLIP_THRESHOLD = 0.6

@st.cache_resource
def load_clip_model():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained("geolocal/StreetCLIP")
    processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
    return model, processor

clip_model, clip_processor = load_clip_model()

def detect_landmark(image_or_path):
    clip_model, clip_processor = load_clip_model()
    try:
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path).convert("RGB")
        else:
            image = image_or_path.convert("RGB")

        inputs = clip_processor(
            text=list(LANDMARK_KEYWORDS.keys()),
            images=image,
            return_tensors="pt",
        )
        outputs = clip_model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1).squeeze()
        max_prob = torch.max(probs).item()

        if max_prob > CLIP_THRESHOLD:
            best_match_key = list(LANDMARK_KEYWORDS.keys())[torch.argmax(probs).item()]
            best_match_info = LANDMARK_KEYWORDS[best_match_key]
            return {
                "name": best_match_info[0],
                "city": best_match_info[1],
                "lat": best_match_info[2],
                "lon": best_match_info[3],
                "source": "Landmark"
            }
        return None
    except Exception:
        return None
