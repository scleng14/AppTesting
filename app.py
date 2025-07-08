# app.py
import streamlit as st
from datetime import datetime
from PIL import Image
from utils.emotion import EmotionDetector
from location_utils.extract_gps import get_location
import pandas as pd
import os

# ========== Setup ==========
st.set_page_config(page_title="Emotion and Location Analyzer", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

# ========== Sidebar ==========
with st.sidebar:
    st.title("📷 Analyzer Options")
    st.markdown("*Tips for Better Results:*\n- Use clear, front-facing images\n- Ensure good lighting\n- Avoid obstructed faces")
    st.info("💡 Tip: You can switch tabs to view emotion, location, and history.")

# ========== Tabs ==========
tabs = st.tabs(["Home", "Location", "History"])

# ========== Tab 0: Emotion ==========
with tabs[0]:
    st.header("Face Emotion Detection")
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="emotion")

    if uploaded_image is not None:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        detector = EmotionDetector()
        result = detector.analyze_emotion(img)

        st.subheader("Emotion Results")
        st.write(result)

# ========== Tab 1: Location ==========
with tabs[1]:
    st.header("📍 Location Recognition")
    location_image = st.file_uploader("Upload an image with GPS or landmark", type=["jpg", "jpeg", "png"], key="location")
    username = st.text_input("Your Name", max_chars=50)

    if location_image is not None:
        img = Image.open(location_image).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if not username.strip():
            st.warning("Please enter your name to continue.")
        else:
            with st.spinner("Analyzing location..."):
                address, method = get_location(location_image)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                st.success(f"📍 Detected Location: {address}")
                st.info(f"Method Used: {method}")
                st.caption(f"Timestamp: {timestamp}")

                st.session_state.history.append({
                    "username": username,
                    "location": address,
                    "method": method,
                    "timestamp": timestamp
                })

# ========== Tab 2: History ==========
with tabs[2]:
    st.header("🕓 Analysis History")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "history.csv", "text/csv")
    else:
        st.info("No history yet. Try uploading an image first.")
