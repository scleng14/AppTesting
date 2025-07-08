import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import os
import plotly.express as px
from emotion_utils.detector import EmotionDetector
from location_utils.extract_gps import extract_gps, convert_gps
from location_utils.landmark import detect_landmark
from location_utils.geocoder import get_address_from_coords, query_landmark_coords

# ----------------- 原有情绪检测代码完全不变 -----------------
@st.cache_resource
def get_detector():
    return EmotionDetector()

detector = get_detector()

def save_history(username, emotion, confidence, location="Unknown", coords=None):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[username, emotion, confidence, location, now, coords]],
                     columns=["Username", "Emotion", "Confidence", "Location", "timestamp", "Coordinates"])
    try:
        if os.path.exists("history.csv"):
            prev = pd.read_csv("history.csv")
            df = pd.concat([prev, df])
        df.to_csv("history.csv", index=False)
    except Exception as e:
        st.error(f"Failed to save history: {e}")

# ... (其余原有函数如 show_detection_guide, sidebar_design 完全不变)

def get_location(image):
    """整合三级定位逻辑"""
    # 1. GPS 定位
    gps_data = extract_gps(image)
    if gps_data:
        coords = convert_gps(gps_data)
        if coords:
            address = get_address_from_coords(coords)
            if address:
                return address, "GPS (EXIF)", coords
    
    # 2. 地标识别
    landmark_name = detect_landmark(image)
    if landmark_name:
        coords, source = query_landmark_coords(landmark_name)
        if coords:
            address = get_address_from_coords(coords)
            if address:
                return address, f"Landmark recognition ({source})", coords
        return landmark_name, "Landmark detected", coords
    
    return "Unknown", "No location data", None

def main():
    # ... (原有界面代码完全不变，仅在文件上传部分添加位置检测)
    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # 原有情绪检测
            detections = detector.detect_emotions(img)
            detected_img = detector.draw_detections(img, detections)
            
            # 新增位置检测
            location, method, coords = get_location(image)
            
            # 在原有结果中显示位置信息
            if detections:
                st.success(f"📍 Location: {location} (Method: {method})")
                save_history(username, emotions[0], confidences[0], location, str(coords) if coords else None)

        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
