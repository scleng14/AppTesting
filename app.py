# app.py
import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import random
import plotly.express as px
from emotion_utils.detector import EmotionDetector
from location_utils.extract_gps import extract_gps_from_image
from location_utils.landmark import detect_landmark
from location_utils.geocoder import reverse_geocode

# ----------------- App Configuration -----------------
st.set_page_config(
    page_title="AI Emotion & Location Detector",
    page_icon="👁‍🗨",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def get_detector():
    return EmotionDetector()

detector = get_detector()

def save_history(username, emotion, confidence, location):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[username, emotion, confidence, location, now]],
                     columns=["Username","Emotion","Confidence","Location","timestamp"])
    try:
        if os.path.exists("history.csv"):
            prev = pd.read_csv("history.csv")
            df = pd.concat([prev, df])
        df.to_csv("history.csv", index=False)
    except Exception as e:
        st.error(f"Failed to save history: {e}")

def show_detection_guide():
    with st.expander("ℹ️ How Emotion Detection Works", expanded=False):
        st.markdown("""
        *Detection Logic Explained:*
        - 😊 Happy: Smile present, cheeks raised
        - 😠 Angry: Eyebrows lowered, eyes wide open
        - 😐 Neutral: No strong facial movements
        - 😢 Sad: Eyebrows raised, lip corners down
        - 😲 Surprise: Eyebrows raised, mouth open
        - 😨 Fear: Eyes tense, lips stretched
        - 🤢 Disgust: Nose wrinkled, upper lip raised

        *Tips for Better Results:*
        - Use clear, front-facing images
        - Ensure good lighting
        - Avoid obstructed faces
        """)

def sidebar_design(username):
    if username:
        st.sidebar.success(f"👤 Logged in as: {username}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Quick Navigation")
    st.sidebar.markdown("- Upload and detect emotions")
    st.sidebar.markdown("- View and filter upload history")
    st.sidebar.markdown("- Visualize your emotion distribution")
    st.sidebar.divider()
    st.sidebar.info("Enhance your experience by ensuring clear, well-lit facial images.")

def main():
    st.title("👁‍🗨 AI Emotion & Location Detector")
    st.caption("Upload a photo to detect facial emotions and estimate location.")
    tabs = st.tabs(["🏠 Home", "🗺️ Location Map", "📜 Upload History", "📊 Emotion Analysis Chart"])

    with tabs[0]:
        username = st.text_input("👤 Enter your username")
        sidebar_design(username)
        if username:
            uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "png"])
            if uploaded_file:
                try:
                    image = Image.open(uploaded_file)
                    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    detections = detector.detect_emotions(img)
                    detected_img = detector.draw_detections(img, detections)

                    coords, method = extract_gps_from_image(image)
                    location_display = "Unknown"
                    if coords:
                        location_display = reverse_geocode(coords)
                    else:
                        landmark_info = detect_landmark(image)
                        if landmark_info:
                            location_display = f"{landmark_info['name']}, {landmark_info['city']}"
                            method = "Landmark"

                    st.session_state["uploaded_image"] = uploaded_file

                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.subheader("🔍 Detection Results")
                        if detections:
                            emotions = [d["emotion"] for d in detections]
                            confidences = [d["confidence"] for d in detections]
                            st.success(f"🎭 {len(detections)} face(s) detected")
                            for i, (emo, conf) in enumerate(zip(emotions, confidences)):
                                st.write(f"- Face {i + 1}: {emo} ({conf}%)")
                            show_detection_guide()
                            save_history(username, emotions[0], confidences[0], location_display)
                            st.info(f"📍 Location: {location_display}")
                        else:
                            st.warning("No faces were detected in the uploaded image.")
                    with col2:
                        t1, t2 = st.tabs(["Original Image", "Processed Image"])
                        with t1:
                            st.image(image, use_container_width=True)
                        with t2:
                            st.image(detected_img, channels="BGR", use_container_width=True,
                                     caption=f"Detected {len(detections)} face(s)")
                except Exception as e:
                    st.error(f"Error while processing the image: {e}")

    with tabs[1]:
        st.header("📍 Location Detection")

        if "uploaded_image" in st.session_state:
            loc_file = st.session_state["uploaded_image"]
            image = Image.open(loc_file)
            coords, method = extract_gps_from_image(image)
            location_text = "Unknown"
            if coords:
                location_text = reverse_geocode(coords)
            else:
                landmark_info = detect_landmark(image)
                if landmark_info:
                    location_text = f"{landmark_info['name']}, {landmark_info['city']}"
                    method = "Landmark"

            st.success("Detection completed!")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown(f"**Detected Location:** {location_text}")
            st.markdown(f"**Detection Method:** {method if method else 'Unknown'}")
            st.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.info("No image found. Please upload an image in the Home tab first.")

    with tabs[2]:
        st.subheader("📜 Upload History")
        if username:
            try:
                if os.path.exists("history.csv"):
                    df = pd.read_csv("history.csv")
                    if df.empty:
                        st.info("No upload records found.")
                    else:
                        df_filtered = df[df["Username"].str.contains(username, case=False)]
                        df_filtered = df_filtered.sort_values("timestamp", ascending=False).tail(100).reset_index(drop=True)
                        df_filtered.index = range(1, len(df_filtered)+1)
                        st.dataframe(df_filtered)
                        st.caption(f"Total records found for {username}: {len(df_filtered)}")
                else:
                    st.info("No history file found.")
            except:
                st.warning("Error loading history records.")
        else:
            st.warning("Please enter your username to view your upload history.")

    with tabs[3]:
        st.subheader("📊 Emotion Analysis Chart")
        if username:
            try:
                if os.path.exists("history.csv"):
                    df = pd.read_csv("history.csv")
                    df_filtered = df[df["Username"].str.contains(username, case=False)]
                    if not df_filtered.empty:
                        fig = px.pie(df_filtered, names="Emotion", title=f"Emotion Distribution for {username}")
                        st.plotly_chart(fig)
                        st.caption("Chart is based on your personal upload history.")
                    else:
                        st.info("No emotion records found for this username.")
                else:
                    st.info("History file not found.")
            except Exception as e:
                st.error(f"Error generating chart: {e}")
        else:
            st.warning("Please enter your username to generate your emotion chart.")

if __name__ == "__main__":
    main()
