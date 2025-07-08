# landmark.py
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import requests
import json

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

landmark_list = [
    "Petronas Twin Towers", "Kuala Lumpur Tower", "Sultan Abdul Samad Building",
    "Malacca Christ Church", "Penang Kek Lok Si Temple", "Mount Kinabalu",
    "Langkawi Sky Bridge", "Putra Mosque", "Batu Caves", "George Town UNESCO Site"
]

def detect_landmark(image: Image.Image):
    try:
        inputs = processor(text=landmark_list, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).tolist()[0]
        best_idx = int(torch.argmax(logits_per_image))
        return {
            "name": landmark_list[best_idx],
            "score": probs[best_idx]
        }
    except:
        return None

def query_landmark_coords(name):
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={name}&format=json"
        res = requests.get(url, headers={"User-Agent": "emotion-location-app"})
        data = res.json()
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return (lat, lon)
        return None
    except:
        return None
