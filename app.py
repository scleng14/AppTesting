import streamlit as st
import cv2
import numpy as np
from PIL import Image, ExifTags
import pandas as pd
from datetime import datetime
import os
import plotly.express as px
from emotion_utils.detector import EmotionDetector
from location_utils.extract_gps import extract_gps, convert_gps
from location_utils.landmark import detect_landmark, query_landmark_coords
from location_utils.geocoder import get_address_from_coords

# ----------------- App Configuration -----------------
st.set_page_config(
    page_title="AI Emotion & Location Detector",
    page_icon="👁‍🗨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- Cached Resources -----------------
@st.cache_resource
def get_detector():
    return EmotionDetector()

detector = get_detector()

# ----------------- Core Functions -----------------
def save_history(username, emotion, confidence, location="Unknown", coords=None):
    """Save results to history.csv (added coords column)"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[username, emotion, confidence, location, now, coords]],
                     columns=["Username","Emotion","Confidence","Location","timestamp","Coordinates"])
    try:
        if os.path.exists("history.csv"):
            prev = pd.read_csv("history.csv")
            df = pd.concat([prev, df])
        df.to_csv("history.csv", index=False)
    except Exception as e:
        st.error(f"Failed to save history: {e}")

def show_detection_guide():
    """Original emotion detection guide (unchanged)"""
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
        """)

def sidebar_design(username):
    """Original sidebar (unchanged)"""
    if username:
        st.sidebar.success(f"👤 Logged in as: {username}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("## Quick Navigation")
    st.sidebar.markdown("- Upload and detect emotions")
    st.sidebar.markdown("- View and filter upload history")
    st.sidebar.markdown("- Visualize your emotion distribution")

def get_location(image):
    """Integrated location detection pipeline"""
    # 1. Try GPS EXIF data
    gps_data = extract_gps(image)
    if gps_data:
        coords = convert_gps(gps_data)
        if coords:
            address = get_address_from_coords(coords)
            if address:
                return address, "GPS (EXIF)", coords
    
    # 2. Try landmark recognition
    landmark_name = detect_landmark(image)
    if landmark_name:
        coords, source = query_landmark_coords(landmark_name)
        if coords:
            address = get_address_from_coords(coords)
            if address:
                return address, f"Landmark recognition ({source})", coords
        return landmark_name, "Landmark detected", coords
    
    return "Unknown", "No location data", None

# ----------------- Main App -----------------
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
                    
                    # Original emotion detection
                    detections = detector.detect_emotions(img)
                    detected_img = detector.draw_detections(img, detections)
                    
                    # New location detection
                    location, method, coords = get_location(image)
                    
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.subheader("🔍 Detection Results")
                        if detections:
                            emotions = [d["emotion"] for d in detections]
                            confidences = [d["confidence"] for d in detections]
                            st.success(f"🎭 {len(detections)} face(s) detected")
                            for i, (emo, conf) in enumerate(zip(emotions, confidences)):
                                st.write(f"- Face {i + 1}: {emo} ({conf}%)")
                            
                            # Display location info
                            st.success(f"📍 Location: {location}")
                            st.caption(f"Method: {method}")
                            
                            save_history(username, emotions[0], confidences[0], location, str(coords) if coords else None)
                            show_detection_guide()
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
                    st.error(f"Error processing image: {str(e)}")

    with tabs[1]:  # Enhanced map tab
        st.subheader("🗺️ Detected Locations")
        if username:
            try:
                if os.path.exists("history.csv"):
                    df = pd.read_csv("history.csv")
                    df_filtered = df[df["Username"] == username]
                    
                    if not df_filtered.empty:
                        # Process coordinates
                        df_mapped = df_filtered.dropna(subset=["Coordinates"])
                        if not df_mapped.empty:
                            df_mapped[['lat', 'lon']] = (
                                df_mapped['Coordinates']
                                .str.strip('()')
                                .str.split(',', expand=True)
                                .astype(float)
                            )
                            st.map(df_mapped)
                        else:
                            st.info("No geotagged images in your history.")
                    else:
                        st.info("No location records found for this username.")
                else:
                    st.info("No history file found.")
            except Exception as e:
                st.error(f"Map error: {str(e)}")
        else:
            st.warning("Please enter your username to view locations.")

    with tabs[2]:  # Original history tab (with location column)
        st.subheader("📜 Upload History")
        if username:
            try:
                if os.path.exists("history.csv"):
                    df = pd.read_csv("history.csv")
                    if df.empty:
                        st.info("No upload records found.")
                    else:
                        df_filtered = df[df["Username"].str.contains(username, case=False)]
                        df_filtered = df_filtered.sort_values("timestamp", ascending=False)
                        st.dataframe(
                            df_filtered[["Username", "Emotion", "Confidence", "Location", "timestamp"]],
                            hide_index=True
                        )
                else:
                    st.info("No history file found.")
            except Exception as e:
                st.error(f"History error: {str(e)}")
        else:
            st.warning("Please enter your username to view history.")

    with tabs[3]:  # Original chart tab (unchanged)
        st.subheader("📊 Emotion Analysis Chart")
        if username:
            try:
                if os.path.exists("history.csv"):
                    df = pd.read_csv("history.csv")
                    df_filtered = df[df["Username"].str.contains(username, case=False)]
                    if not df_filtered.empty:
                        fig = px.pie(df_filtered, names="Emotion", title=f"Emotion Distribution for {username}")
                        st.plotly_chart(fig)
                    else:
                        st.info("No emotion records found.")
                else:
                    st.info("No history file found.")
            except Exception as e:
                st.error(f"Chart error: {str(e)}")
        else:
            st.warning("Please enter your username to view charts.")

if __name__ == "__main__":
    main()
