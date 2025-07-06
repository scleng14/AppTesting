import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from datetime import datetime
import random
import logging
import os
from io import BytesIO

# ----------------- 初始化设置 -----------------
st.set_page_config(
    page_title="AI Emotion & Location Detector",
    page_icon="👁‍🗨",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- 多语言支持 -----------------
LANGUAGES = ["中文", "English", "Malay"]
lang = st.sidebar.selectbox("🌐 Select Language / 选择语言 / Pilih Bahasa", LANGUAGES)

TRANSLATIONS = {
    "中文": {
        "title": "AI情绪与位置检测系统",
        "upload_guide": "上传照片分析面部表情并推测位置",
        "username": "用户名",
        "user_auth": "用户认证",
        "enter_username": "输入用户名",
        "welcome": "欢迎",
        "upload_image": "上传图片 (JPG/PNG)",
        "analysis_results": "分析结果",
        "detected_emotion": "检测到的情绪",
        "estimated_location": "估计位置",
        "download_results": "下载结果",
        "original_image": "原始图片",
        "processed_image": "处理后的图片",
        "no_faces": "未检测到人脸",
        "error_processing": "图片处理错误",
        "debug_info": "调试信息",
        "input_username_continue": "请输入用户名继续",
        "nav_home": "主页",
        "nav_location_map": "位置地图",
        "nav_history": "历史记录",
        "nav_emotion_chart": "情绪分析图表",  
        "upload_history": "上传历史",
        "no_history": "暂无历史记录",
        "filter_user": "按用户名筛选",
        "records_shown": "条记录已显示",
        "no_record_found": "未找到记录",
        "enter_username_history": "请输入用户名查看历史"
    },
    "English": {
        "title": "AI Emotion & Location Detector",
        "upload_guide": "Upload a photo to analyze facial expressions and estimate location",
        "username": "Username",
        "user_auth": "User Authentication",
        "enter_username": "Enter your username",
        "welcome": "Welcome",
        "upload_image": "Upload an image (JPG/PNG)",
        "analysis_results": "Analysis Results",
        "detected_emotion": "Detected Emotion",
        "estimated_location": "Estimated Location",
        "download_results": "Download Results",
        "original_image": "Original Image",
        "processed_image": "Processed Image",
        "no_faces": "No faces detected",
        "error_processing": "Error processing image",
        "debug_info": "Debug Info",
        "input_username_continue": "Please enter a username to continue",
        "nav_home": "Home",
        "nav_location_map": "Location Map",
        "nav_history": "History",
        "nav_emotion_chart": "Emotion Chart",  
        "upload_history": "Upload History",
        "no_history": "No history available",
        "filter_user": "Filter by username",
        "records_shown": "records shown",
        "no_record_found": "No records found",
        "enter_username_history": "Please enter username to view history"
    },
    "Malay": {
        "title": "Sistem Pengesanan Emosi & Lokasi AI",
        "upload_guide": "Muat naik foto untuk analisis ekspresi muka dan anggaran lokasi",
        "username": "Nama pengguna",
        "user_auth": "Pengesahan Pengguna",
        "enter_username": "Masukkan nama pengguna",
        "welcome": "Selamat datang",
        "upload_image": "Muat naik imej (JPG/PNG)",
        "analysis_results": "Keputusan Analisis",
        "detected_emotion": "Emosi yang Dikesan",
        "estimated_location": "Lokasi Dianggarkan",
        "download_results": "Muat Turun Keputusan",
        "original_image": "Imej Asal",
        "processed_image": "Imej Diproses",
        "no_faces": "Tiada muka dikesan",
        "error_processing": "Ralat memproses imej",
        "debug_info": "Maklumat Debug",
        "input_username_continue": "Masukkan nama pengguna untuk meneruskan",
        "nav_home": "Halaman Utama",
        "nav_location_map": "Peta Lokasi",
        "nav_history": "Sejarah",
        "nav_emotion_chart": "Carta Emosi",  
        "upload_history": "Sejarah Muat Naik",
        "no_history": "Tiada sejarah tersedia",
        "filter_user": "Tapis mengikut nama pengguna",
        "records_shown": "rekod dipaparkan",
        "no_record_found": "Tiada rekod dijumpai",
        "enter_username_history": "Sila masukkan nama pengguna untuk melihat sejarah"
    }
}
T = TRANSLATIONS[lang]

# ----------------- 加载模型 -----------------
@st.cache_resource
def load_models():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    return face_cascade, eye_cascade, smile_cascade
    
face_cascade, eye_cascade, smile_cascade = load_models()

# ----------------- 核心功能 -----------------
def detect_emotion(image):
    """Detect emotions using OpenCV (happy, neutral, sad, angry)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    emotions = []
    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        
        # Detect smiles
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        # Emotion detection logic
        emotion = "neutral"  # default
        
        # Anger detection
        if len(eyes) >= 2:
            eye_centers = [y + ey + eh/2 for (ex,ey,ew,eh) in eyes[:2]]
            avg_eye_height = np.mean(eye_centers)
            eye_sizes = [eh for (ex,ey,ew,eh) in eyes[:2]]
            avg_eye_size = np.mean(eye_sizes)
            
            if avg_eye_size > h/5 and avg_eye_height < h/2.5:
                emotion = "angry"
            elif avg_eye_height < h/3:
                emotion = "sad"
        
        # Happiness detection (priority)
        if len(smiles) > 0:
            emotion = "happy"
        
        emotions.append(emotion)
    
    return emotions, faces

def draw_detections(img, emotions, faces):
    """Draw detection boxes with English labels"""
    output_img = img.copy()
    
    # Color mapping
    color_map = {
        "happy": (0, 255, 0),     # green
        "neutral": (255, 255, 0), # yellow
        "sad": (0, 0, 255),       # red
        "angry": (0, 165, 255)    # orange
    }
    
    for i, ((x,y,w,h), emotion) in enumerate(zip(faces, emotions)):
        color = color_map.get(emotion, (255, 255, 255))
        
        # Draw face rectangle
        cv2.rectangle(output_img, (x,y), (x+w,y+h), color, 3)
        
        # Add emotion label
        cv2.putText(output_img, 
                   emotion.upper(), 
                   (x+5, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, 
                   color, 
                   2)
    
    return output_img

def show_detection_guide(show_full_guide=True):
    """Show detection guide in expandable section"""
    with st.expander("ℹ️ How Emotion Detection Works", expanded=False):
        if show_full_guide:
            st.markdown("""
            *Detection Logic Explained:*
            
            - 😊 *Happy*: Detected when smile is present
            - 😠 *Angry*: Detected when eyes are wide open and positioned in upper face
            - 😐 *Neutral*: Default state when no strong indicators found
            - 😢 *Sad*: Detected when eyes are positioned higher than normal
            
            *Tips for Better Results:*
            - Use clear, front-facing images
            - Ensure good lighting
            - Avoid obstructed faces
            """)
        else:
            st.markdown("""
            *Tips for Better Results:*
            - Use clear, front-facing images
            - Ensure good lighting
            - Avoid obstructed faces
            """)

def save_history(username, emotion, location):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_record = pd.DataFrame([[username, emotion, location, now]], 
                            columns=["Username", "Emotion", "Location", "timestamp"])
    
    try:
        if os.path.exists("history.csv"):
            history_df = pd.read_csv("history.csv")
            history_df = pd.concat([history_df, new_record])
        else:
            history_df = new_record
        history_df.to_csv("history.csv", index=False)
    except Exception as e:
        logger.error(f"保存历史失败: {e}")

# ----------------- 主程序 -----------------
def main():
    st.title(f"🌍 {T['title']}")
    st.caption(T["upload_guide"])

    # 初始化标签页
    tabs = st.tabs([
        f"🏠 {T['nav_home']}",
        f"🗺️ {T['nav_location_map']}",
        f"📜 {T['nav_history']}",
        f"📊 {T['nav_emotion_chart'']}"
    ])

    # 主页标签
    with tabs[0]:
        username = st.text_input(f"👤 {T['enter_username']}")
        if username:
            st.sidebar.success(f"👤 {T['welcome']} {username}")
            uploaded_file = st.file_uploader("Upload Image (JPG/PNG)", type=["jpg", "png"])
            try:
            # Convert image format
            image = Image.open(uploaded_file)
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect emotions
            emotions, faces = detect_emotion(img)
            detected_img = draw_detections(img.copy(), emotions, faces)
            
            # Two-column layout
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("🔍 Detection Results")
                if emotions:
                    # Count each emotion type
                    emotion_count = {}
                    for emo in emotions:
                        emotion_count[emo] = emotion_count.get(emo, 0) + 1
                    
                    # Format the result string
                    result_parts = []
                    for emo, count in emotion_count.items():
                        result_parts.append(f"{count} {emo.capitalize()}")
                    
                    st.success(f"🎭 Detected {len(faces)} face(s): " + ", ".join(result_parts))
                    
                    # Show full detection guide
                    show_detection_guide(show_full_guide=True)
                else:
                    st.warning("No faces detected")
                    # Show only tips when no faces detected
                    show_detection_guide(show_full_guide=False)
            
            with col2:
                tab1, tab2 = st.tabs(["Original Image", "Analysis Result"])
                with tab1:
                    st.image(image, use_container_width=True)
                with tab2:
                    st.image(detected_img, channels="BGR", use_container_width=True,
                           caption=f"Detected {len(faces)} faces")
                
        except Exception as e:
            st.error(f"Processing error: {str(e)}")
    # 位置地图标签
    with tabs[1]:
        st.map(pd.DataFrame({
            'lat': [3.1390 + random.uniform(-0.01, 0.01)],
            'lon': [101.6869 + random.uniform(-0.01, 0.01)]
        }))

    # 历史记录标签
    with tabs[2]:
        st.header(f"📜 {T['upload_history']}")
        if username:
            try:
                history_df = pd.read_csv("history.csv") 
                if history_df.empty:
                st.info(T["no_history"])
            else:
                keyword = st.text_input(T["filter_user"]).strip()
                if keyword:
                    filtered_df = history_df[history_df["Username"].str.lower() == keyword.lower()]
                else:
                    filtered_df = history_df
                st.dataframe(filtered_df)
                st.caption(f"{len(filtered_df)} {T['records_shown']}")
        except FileNotFoundError:
            st.info(T["no_record_found"])
    else:
        st.warning(T["enter_username_history"])

    # 筛选标签
    with tabs[3]:
        st.subheader(f"📊 {T['nav_emotion_chart']}")
        
        try:
            if os.path.exists("history.csv"):
                df = pd.read_csv("history.csv")
                if not df.empty:
                    fig = px.pie(df, names="Emotion", title=T["nav_emotion_chart"])
                    st.plotly_chart(fig)
                else:
                    st.warning(T["no_history"])
            else:
                st.warning(T["no_record_found"])
        except Exception as e:
            st.error(f"生成图表错误: {e}")

if __name__ == "__main__":
    main()
