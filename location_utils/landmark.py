import streamlit as st
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
import numpy as np

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_THRESHOLD = 0.6
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

@st.cache_resource
def load_models():
    return (
        CLIPModel.from_pretrained(CLIP_MODEL_NAME),
        CLIPProcessor.from_pretrained(CLIP_MODEL_NAME),
    )

model, processor = load_models()

# Example landmark list
landmark_list = [
    "Eiffel Tower", "Statue of Liberty", "Colosseum", "Big Ben", "KLCC", "Petronas Towers",
    "Pyramids of Giza", "Taj Mahal", "Mount Fuji", "Burj Khalifa"
]

def detect_landmark(image):
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    inputs = processor(text=landmark_list, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).numpy().flatten()

    best_idx = int(np.argmax(probs))
    if probs[best_idx] > CLIP_THRESHOLD:
        return landmark_list[best_idx]
    return None

def query_landmark_coords(landmark_name):
    query = f"""
    [out:json];
    node["name"="{landmark_name}"];
    out center;
    """
    try:
        response = requests.post(OVERPASS_URL, data=query.encode("utf-8"), timeout=30)
        data = response.json()
        if data.get("elements"):
            node = data["elements"][0]
            lat = node.get("lat") or node.get("center", {}).get("lat")
            lon = node.get("lon") or node.get("center", {}).get("lon")
            return (lat, lon)
    except Exception:
        return None
    return None
