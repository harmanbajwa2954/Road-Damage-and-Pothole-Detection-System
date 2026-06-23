import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile

# ========================================
# 🛠️ CONFIGURATION & MODEL LOADING
# ========================================
st.set_page_config(page_title="Road Defect Detector", layout="wide")

MODEL_PATH = "best.pt"  # Update this to your path if different

@st.cache_resource
def load_model():
    try:
        return YOLO(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}. Ensure the file exists.")
        return None

model = load_model()

# RDD 2022 Class mapping from your training script
CLASS_NAMES = ['Longitudinal_Crack', 'Transverse_Crack', 'Alligator_Crack', 'Pothole']

# ========================================
# 🎨 UI LAYOUT
# ========================================
st.title("🛣️ Road Defect & Pothole Detection")
st.write("Real-time automated road damage assessment using YOLOv8 trained on RDD 2022.")

# Sidebar sidebar options
st.sidebar.header("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

# Mode selection
app_mode = st.sidebar.selectbox("Choose Detection Mode", ["About Dataset", "Upload Video", "Live Webcam"])

# --- MODE 1: ABOUT ---
if app_mode == "About Dataset":
    st.subheader("RDD 2022 Model Details")
    st.markdown("""
    This model detects **4 specific types** of road distress:
    * **Longitudinal Crack**
    * **Transverse Crack**
    * **Alligator Crack**
    * **Pothole**
    
    Select a mode from the sidebar to begin processing video data.
    """)

# --- MODE 2: UPLOAD VIDEO ---
elif app_mode == "Upload Video":
    st.subheader("📹 Video Upload Detection")
    uploaded_file = st.file_uploader("Upload a road survey video (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save uploaded file to a temporary file path
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()  # Placeholder for the video stream
        
        stop_btn = st.button("Stop Processing")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or stop_btn:
                break
                
            # Run YOLO inference
            results = model.predict(frame, conf=confidence_threshold, verbose=False)
            
            # Annotate frame
            annotated_frame = results[0].plot()
            
            # Streamlit needs RGB; OpenCV produces BGR
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st_frame.image(annotated_frame, channels="RGB", use_container_width=True)
            
        cap.release()
        st.success("Video processing finished.")

# --- MODE 3: LIVE WEBCAM ---
elif app_mode == "Live Webcam":
    st.subheader("🎥 Live Detection Window")
    st.info("Ensure webcam permissions are granted. Click 'Stop' below to halt the feed.")
    
    run_webcam = st.checkbox("Start Webcam", value=True)
    st_frame = st.empty()
    
    # 0 is usually the default built-in webcam
    cap = cv2.VideoCapture(0)
    
    while run_webcam and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame from webcam.")
            break
            
        # Run YOLO inference
        results = model.predict(frame, conf=confidence_threshold, verbose=False)
        
        # Annotate frame
        annotated_frame = results[0].plot()
        
        # Convert BGR to RGB
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(annotated_frame, channels="RGB", use_container_width=True)
        
    cap.release()
    st.write("Webcam feed stopped.")