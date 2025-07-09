def get_config():
    return {
        "title": "AI Emotion Detector",
        "language_selector": {
            "label": "\U0001F310 Select Language"
        },
        "color_map": {
            "happy": (0, 255, 0),      # Green
            "neutral": (255, 255, 0),   # Yellow
            "sad": (0, 0, 255),         # Red
            "angry": (0, 165, 255),     # Orange
            "fear": (128, 0, 128),      # Purple
            "surprise": (255, 0, 255),  # Pink
            "disgust": (0, 128, 0)      # Dark Green
        }
    }
