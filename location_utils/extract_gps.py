import streamlit as st
from modules.emotion_detector import EmotionDetector
from location_utils.extract_gps import get_location
from datetime import datetime
import pandas as pd
import os

st.set_page_config(page_title="Emotion & Location Detector", layout="wide")

# ===== Sidebar =====
st.sidebar.title("Emotion App")
username = st.sidebar.text_input("Enter your name:", value="Guest")

st.sidebar.markdown("---")
st.sidebar.info("💬 Note: This demo analyzes face emotions and estimates photo location based on GPS or landmark features.")

# ===== Tabs =====
tabs = st.tabs(["Home", "Location", "History", "Chart"])

# ===== Tab 0: Emotion Detection =====
with tabs[0]:
    st.header("😊 Emotion Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        detector = EmotionDetector(uploaded_file)
        detector.run(username)

# ===== Tab 1: Location Detection =====
with tabs[1]:
    st.header("📍 Location Detection")
    loc_file = st.file_uploader("Upload an image for location detection", type=["jpg", "jpeg", "png"], key="loc")

    if loc_file is not None:
        with open("temp_location.jpg", "wb") as f:
            f.write(loc_file.read())

        with st.spinner("Analyzing location..."):
            location_result, method = get_location("temp_location.jpg")

        st.success("Detection completed!")
        st.image("temp_location.jpg", caption="Uploaded Image", use_column_width=True)
        st.markdown(f"**Detected Location:** {location_result}")
        st.markdown(f"**Detection Method:** {method}")
        st.markdown(f"**Username:** {username}")
        st.markdown(f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ===== Tab 2: History =====
with tabs[2]:
    st.header("📚 History Records")
    if os.path.exists("history.csv"):
        df = pd.read_csv("history.csv")
        st.dataframe(df)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download History", csv, "history.csv", "text/csv")
    else:
        st.info("No history available yet.")

# ===== Tab 3: Chart =====
with tabs[3]:
    st.header("📊 Emotion Chart")
    if os.path.exists("history.csv"):
        df = pd.read_csv("history.csv")
        emotion_counts = df["emotion"].value_counts()
        st.bar_chart(emotion_counts)
    else:
        st.info("No data to display chart.")
