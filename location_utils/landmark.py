import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
import streamlit as st

# 地标数据库（可扩展）
LANDMARK_KEYWORDS = {
    "petronas towers": ("Petronas Twin Towers", "Kuala Lumpur", 3.1579, 101.7116),
    "kl tower": ("KL Tower", "Kuala Lumpur", 3.1528, 101.7039),
    "georgetown mural": ("Georgetown Street Art", "Penang", 5.4170, 100.3380),
    "putrajaya mosque": ("Putra Mosque", "Putrajaya", 2.9360, 101.6910)
}

# 模型参数
CLIP_MODEL_NAME = "geolocal/StreetCLIP"
CLIP_THRESHOLD = 0.6

# 缓存模型加载
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return model, processor

clip_model, clip_processor = load_clip_model()

def detect_landmark(image_path):
    """
    用 CLIP 模型识别图像中的已知地标。
    参数:
        image_path: 图片路径
    返回:
        匹配到的 landmark 名称（str） 或 None
    """
    try:
        image = Image.open(image_path).convert("RGB")
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
            best_match = list(LANDMARK_KEYWORDS.keys())[torch.argmax(probs).item()]
            return best_match
        else:
            return None
    except Exception:
        return None

def query_landmark_coords(landmark_name):
    """
    获取 landmark 名称对应的坐标和来源
    返回:
        ((lat, lon), source) 或 (None, "Failed")
    """
    if landmark_name in LANDMARK_KEYWORDS:
        lat = LANDMARK_KEYWORDS[landmark_name][2]
        lon = LANDMARK_KEYWORDS[landmark_name][3]
        return (lat, lon), "CLIP Landmark Match"

    try:
        # fallback：尝试用 OSM 查询地址
        response = requests.get(
            f"https://nominatim.openstreetmap.org/search?q={landmark_name}&format=json"
        )
        if response.ok and response.json():
            lat = float(response.json()[0]["lat"])
            lon = float(response.json()[0]["lon"])
            return (lat, lon), "OSM Search"
    except Exception:
        pass

    return None, "Failed"
