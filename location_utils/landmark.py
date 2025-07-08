import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import streamlit as st

# 内置 landmark 数据库，可按需扩展
LANDMARK_KEYWORDS = {
    "petronas towers": ("Petronas Twin Towers", "Kuala Lumpur", 3.1579, 101.7116),
    "kl tower": ("KL Tower", "Kuala Lumpur", 3.1528, 101.7039),
    "georgetown mural": ("Georgetown Street Art", "Penang", 5.4170, 100.3380),
    "putrajaya mosque": ("Putra Mosque", "Putrajaya", 2.9360, 101.6910),
    "malacca dutch square": ("Dutch Square", "Malacca", 2.1944, 102.2496)
}

# CLIP 模型设置
CLIP_MODEL_NAME = "geolocal/StreetCLIP"
CLIP_THRESHOLD = 0.6  # 可调整匹配阈值

# 缓存模型加载
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return model, processor

clip_model, clip_processor = load_clip_model()

def detect_landmark(image_or_path):
    """
    使用 CLIP 模型识别上传图片中的已知地标。
    参数:
        image_or_path: PIL.Image 对象或图像路径（str）
    返回:
        地标名称字符串 或 None
    """
    try:
        # 支持 PIL.Image 或文件路径
        if isinstance(image_or_path, str):
            image = Image.open(image_or_path).convert("RGB")
        else:
            image = image_or_path.convert("RGB")

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
            best_match_key = list(LANDMARK_KEYWORDS.keys())[torch.argmax(probs).item()]
            best_match_info = LANDMARK_KEYWORDS[best_match_key]
            return best_match_info[0]  # 返回地标名称
        return None
    except Exception as e:
        return None
