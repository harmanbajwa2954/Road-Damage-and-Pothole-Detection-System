# 🛣️ Road Defect & Pothole Detection

A real-time automated road damage assessment application built using **Streamlit**, **OpenCV**, and **YOLOv8**. 

This application uses a YOLOv8 model trained on the **RDD 2022 dataset** to detect various types of road distress, helping with infrastructure maintenance and safety.

## ✨ Features

- **Real-Time Detection:** Run inference on live webcam feeds or uploaded videos.
- **Multiple Damage Types:** Detects 4 specific types of road distress:
  - Longitudinal Crack
  - Transverse Crack
  - Alligator Crack
  - Pothole
- **Adjustable Confidence:** Easily adjust the confidence threshold using the sidebar slider to filter out less certain predictions.
- **Streamlit UI:** A clean, easy-to-use web interface for interacting with the model.

## 🚀 Installation & Setup

1. **Clone the repository** (or download the source code).
2. **Set up a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   ```
3. **Activate the virtual environment**:
   - **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
   - **Windows (Command Prompt):** `.venv\Scripts\activate.bat`
   - **Mac/Linux:** `source .venv/bin/activate`
4. **Install the dependencies**:
   ```bash
   pip install streamlit ultralytics opencv-python-headless
   ```
5. **Ensure the model exists**: 
   Make sure you have your trained YOLOv8 model file named `best.pt` in the root directory.

## 💻 Usage

To run the Streamlit app, make sure your virtual environment is activated and run:

```bash
streamlit run app.py
```

This will start a local server and open the app in your default web browser (usually at `http://localhost:8501`).

### Detection Modes
- **About Dataset:** View information about the supported classes.
- **Upload Video:** Upload a survey video (MP4, AVI, MOV) to see frame-by-frame annotations of detected defects.
- **Live Webcam:** Start your webcam and run inference in real-time.

## 🛠️ Technologies Used

- **[Streamlit](https://streamlit.io/):** For building the interactive frontend.
- **[YOLOv8 (Ultralytics)](https://github.com/ultralytics/ultralytics):** The core object detection model.
- **[OpenCV](https://opencv.org/):** For video and image processing operations.
