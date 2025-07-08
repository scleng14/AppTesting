from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Add more Malaysian or global landmarks below
landmark_dict = {
    "KLCC Twin Towers": "Kuala Lumpur, Malaysia",
    "Petronas Towers": "Kuala Lumpur, Malaysia",
    "KL Tower": "Kuala Lumpur, Malaysia",
    "Menara Kuala Lumpur": "Kuala Lumpur, Malaysia",
    "Sultan Abdul Samad Building": "Kuala Lumpur, Malaysia",
    "Mount Fuji": "Japan",
    "Eiffel Tower": "Paris, France",
    "Great Wall of China": "China",
    "Statue of Liberty": "New York, USA",
    "Taj Mahal": "Agra, India",
    "Colosseum": "Rome, Italy",
    "Sydney Opera House": "Sydney, Australia"
}

def detect_landmark(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        texts = list(landmark_dict.keys())

        inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)[0]

        top_index = torch.argmax(probs).item()
        top_landmark = texts[top_index]
        top_confidence = probs[top_index].item()

        if top_confidence > 0.3:
            return top_landmark, landmark_dict[top_landmark], round(top_confidence * 100, 2)
        else:
            return None, None, 0.0

    except Exception as e:
        print(f"[LANDMARK ERROR] {str(e)}")
        return None, None, 0.0
