import torch
from transformers import CLIPProcessor, CLIPModel
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import requests

# 预定义地标数据库
LANDMARK_KEYWORDS = {
    "petronas towers": ("Petronas Twin Towers", "Kuala Lumpur", 3.1579, 101.7116),
    # ... (你的完整地标列表)
}

# 初始化模型
CLIP_MODEL_NAME = "geolocal/StreetCLIP"
CLIP_THRESHOLD = 0.6
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

@st.cache_resource
def load_models():
    return (
        CLIPModel.from_pretrained(CLIP_MODEL_NAME),
        CLIPProcessor.from_pretrained(CLIP_MODEL_NAME),
        RateLimiter(Nominatim(user_agent="geo_locator").reverse, min_delay_seconds=2)
    )

clip_model, clip_processor, reverse_geocode = load_models()

def detect_landmark(image):
    """使用 CLIP 模型识别地标"""
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
            return list(LANDMARK_KEYWORDS.keys())[torch.argmax(probs).item()]
        return None
    except Exception:
        return None

def query_landmark_coords(landmark_name):
    """查询地标坐标"""
    if landmark_name in LANDMARK_KEYWORDS:
        return LANDMARK_KEYWORDS[landmark_name][2:], "Predefined"
    
    try:
        response = requests.get(f"https://nominatim.openstreetmap.org/search?q={landmark_name}&format=json")
        if response.json():
            lat = float(response.json()[0]['lat'])
            lon = float(response.json()[0]['lon'])
            return (lat, lon), "OpenStreetMap"
    except Exception:
        pass
    
    return None, "Failed"
