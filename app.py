import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import time
import threading
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import queue
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Anti-Spoofing Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .feature-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    
    .status-live {
        color: #28a745;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .status-spoof {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.5rem;
    }
    
    .confidence-score {
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .detection-stats {
        background: #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .realtime-indicator {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Navigation Styling */
    .nav-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .nav-title {
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .stRadio > div {
        flex-direction: row;
        align-items: center;
        justify-content: center;
        gap: 2rem;
    }
    
    .stRadio > div > label {
        background: rgba(255, 255, 255, 0.1);
        padding: 0.8rem 1.5rem;
        border-radius: 25px;
        color: white !important;
        font-weight: bold;
        border: 2px solid rgba(255, 255, 255, 0.3);
        transition: all 0.3s ease;
        cursor: pointer;
        backdrop-filter: blur(10px);
    }
    
    .stRadio > div > label:hover {
        background: rgba(255, 255, 255, 0.2);
        border-color: rgba(255, 255, 255, 0.6);
        transform: translateY(-2px);
    }
    
    .stRadio > div > label[data-checked="true"] {
        background: rgba(255, 255, 255, 0.9);
        color: #667eea !important;
        border-color: white;
        box-shadow: 0 4px 15px rgba(255, 255, 255, 0.3);
    }
    
    /* Hide radio button circles */
    .stRadio > div > label > div:first-child {
        display: none;
    }
    
    .detection-method-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .method-radio {
        background: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 2px solid #667eea;
        margin: 0.3rem;
        transition: all 0.3s ease;
    }
    
    .method-radio:hover {
        background: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Global variables for real-time detection
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = []
if 'detection_count' not in st.session_state:
    st.session_state.detection_count = {'live': 0, 'spoof': 0}
if 'current_prediction' not in st.session_state:
    st.session_state.current_prediction = "Waiting..."
if 'current_confidence' not in st.session_state:
    st.session_state.current_confidence = 0.0

@st.cache_resource
def load_model():
    """Load the pre-trained MobileNetV2 model"""
    try:
        # Create MobileNetV2 model architecture
        model = models.mobilenet_v2(pretrained=False)
        # Modify the classifier for binary classification (live vs spoof)
        model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.last_channel, 2)
        )
        
        # Load the trained weights
        model.load_state_dict(torch.load('model/mobilenet_v2.pth', map_location='cpu'))
        model.eval()
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def preprocess_image(image):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    return transform(image).unsqueeze(0)

def predict_liveness(model, image):
    """Predict if the image is live or spoofed"""
    if model is None:
        return "Error", 0.0
    
    with torch.no_grad():
        preprocessed = preprocess_image(image)
        outputs = model(preprocessed)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Assuming class 0 = live, class 1 = spoof
        live_confidence = probabilities[0][0].item()
        spoof_confidence = probabilities[0][1].item()
        
        if live_confidence > spoof_confidence:
            return "LIVE", live_confidence
        else:
            return "SPOOF", spoof_confidence

class VideoProcessor:
    def __init__(self):
        self.model = load_model()
        self.frame_count = 0
        self.detection_interval = 5  # Process every 5th frame for performance
        
    def recv(self, frame):
        """Process video frame for real-time detection"""
        img = frame.to_ndarray(format="bgr24")
        
        # Process detection every nth frame
        if self.frame_count % self.detection_interval == 0:
            try:
                prediction, confidence = predict_liveness(self.model, img)
                
                # Update session state
                st.session_state.current_prediction = prediction
                st.session_state.current_confidence = confidence
                
                # Update detection counts
                if prediction == "LIVE":
                    st.session_state.detection_count['live'] += 1
                elif prediction == "SPOOF":
                    st.session_state.detection_count['spoof'] += 1
                
                # Store results
                st.session_state.detection_results.append({
                    'prediction': prediction,
                    'confidence': confidence,
                    'timestamp': time.time()
                })
                
                # Keep only last 50 results
                if len(st.session_state.detection_results) > 50:
                    st.session_state.detection_results.pop(0)
                    
            except Exception as e:
                print(f"Detection error: {e}")
        
        self.frame_count += 1
        
        # Draw detection result on frame
        try:
            prediction = st.session_state.current_prediction
            confidence = st.session_state.current_confidence
            
            # Choose color based on prediction
            if prediction == "LIVE":
                color = (0, 255, 0)  # Green
            elif prediction == "SPOOF":
                color = (0, 0, 255)  # Red
            else:
                color = (128, 128, 128)  # Gray
            
            # Draw rectangle and text
            cv2.rectangle(img, (10, 10), (400, 100), color, 2)
            cv2.putText(img, f"Status: {prediction}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(img, f"Confidence: {confidence:.2%}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                       
        except Exception as e:
            pass
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# RTC Configuration for better connection
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ]
})

# Sidebar Navigation
st.sidebar.markdown("## Page")

# Navigation buttons in sidebar
if st.sidebar.button("📖 Documentation", use_container_width=True, type="primary" if st.session_state.get('current_page', 'Documentation') == 'Documentation' else "secondary"):
    st.session_state.current_page = 'Documentation'

if st.sidebar.button("🔍 Detection", use_container_width=True, type="primary" if st.session_state.get('current_page', 'Documentation') == 'Detection' else "secondary"):
    st.session_state.current_page = 'Detection'

# Set default page if not set
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Documentation'

page = st.session_state.current_page

# Main header
st.markdown('<h1 class="main-header">🛡️ Anti-Spoofing Detection System</h1>', unsafe_allow_html=True)

if page == "Documentation":
    st.markdown("## 🎯 Overview")
    
    st.markdown("""
    Sistem deteksi anti-spoofing ini menggunakan deep learning untuk membedakan antara wajah manusia asli (live) 
    dan serangan spoofing secara real-time.
    """)
    
    # Feature Overview
    st.markdown("---")
    st.markdown("## 🚀 Architecture & Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
        <h3>🤖 Model Architecture</h3>
        <ul>
        <li><strong>Base Model:</strong> MobileNetV2</li>
        <li><strong>Input Size:</strong> 224x224 pixels</li>
        <li><strong>Classes:</strong> Live (0) & Spoof (1)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
        <h3>🛡️ Security Features</h3>
        <ul>
        <li>Deteksi Wajah Asli</li>
        <li>Deteksi Print Attacks</li>
        <li>Deteksi Replay Attacks</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
    # How to Use Section
    st.markdown("---")
    st.markdown("## 🚀 How to Use")
    
    st.markdown("""
    1. Gunakan sidebar dan pilih menu **Detection**
    2. Pilih metode deteksi yang ingin digunakan
    3. **Allow camera access** ketika diminta browser
    4. **Use control buttons** untuk start/stop detection
    """)
    
    # Installation
    st.markdown("---")
    st.markdown("## 📦 Local Installation")
    
    
    st.code("""
        # Clone repository
        git clone https://github.com/riofardinan/anti-spoofing.git
        cd anti-spoofing

        # Setup conda environment
        conda env create -f environment.yml
        conda activate anti-spoofing

        # Or setup with pip
        pip install -r requirements.txt

        # Run application
        streamlit run app.py
    """, language="bash")

    # Installation
    st.markdown("---")
    st.markdown("## 📦 About MobileNetV2")
    
    st.markdown("""
    MobileNetV2 adalah model konvolusional yang dirancang untuk efisiensi dan performa tinggi. 
    Model ini menggunakan arsitektur Depthwise Separable Convolutions untuk mengurangi kompleksitas komputasi.
    """)
    
  
  

elif page == "Detection":
    # Load model
    model = load_model()
    if model is None:
        st.error("❌ Failed to load model. Please check if mobilenet_v2.pth exists and is valid.")
        st.stop()
    
    st.markdown("## 🚀 Start Detection")
    
    st.success("✅ Model loaded successfully!")
    
    st.markdown("### 🎯 Select Method")
    
    # Initialize detection method in session state if not exists
    if 'selected_method' not in st.session_state:
        st.session_state.selected_method = "🎥 Real-time Detection"
    
    # Method selection with buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("🎥 Real-time Detection", 
                    type="primary" if st.session_state.selected_method == "🎥 Real-time Detection" else "secondary",
                    use_container_width=True):
            st.session_state.selected_method = "🎥 Real-time Detection"
            st.rerun()
    
    with col_btn2:
        if st.button("📷 Camera Capture", 
                    type="primary" if st.session_state.selected_method == "📷 Camera Capture" else "secondary",
                    use_container_width=True):
            st.session_state.selected_method = "📷 Camera Capture"
            st.rerun()
    
    with col_btn3:
        if st.button("📁 Upload Image", 
                    type="primary" if st.session_state.selected_method == "📁 Upload Image" else "secondary",
                    use_container_width=True):
            st.session_state.selected_method = "📁 Upload Image"
            st.rerun()
    
    detection_method = st.session_state.selected_method
    
    if detection_method == "🎥 Real-time Detection":

        col1, col2 = st.columns([2, 1])
        
        with col1:
            # WebRTC video streamer
            webrtc_ctx = webrtc_streamer(
                key="live-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=VideoProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )

        with col2:
            st.info("💡 Notes:")
            st.markdown("""
            - **Allow camera access**
            - **Position face clearly**
            - **Ensure good lighting**
            """)
            
    elif detection_method == "📷 Camera Capture":
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Camera input
            camera_input = st.camera_input("Take a photo for detection")
            
            if camera_input is not None:
                # Convert to PIL Image
                image = Image.open(camera_input)
                
                # Display the image
                st.image(image, caption="Captured Image", use_column_width=True)
                
                # Predict
                with st.spinner("🔄 Analyzing..."):
                    prediction, confidence = predict_liveness(model, image)
                
                # Display results in col2
                with col2:
                    st.markdown("### 📊 Results")
                    
                    if prediction == "LIVE":
                        st.markdown(f'<p class="status-live">✅ STATUS: {prediction}</p>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="status-spoof">❌ STATUS: {prediction}</p>', 
                                  unsafe_allow_html=True)
                    
                    st.markdown(f'<p class="confidence-score">🎯 Confidence: {confidence:.2%}</p>', 
                              unsafe_allow_html=True)
                    
                    # Confidence bar
                    st.progress(confidence)
                    
        with col2:
            if camera_input is None:
                st.info("📸 Take a photo using the camera above to start detection")
    
    else:  # Upload Image
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose an image for liveness detection",
                type=['jpg', 'jpeg', 'png', 'bmp'],
                help="Upload a clear image of a face for analysis"
            )
            
            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Predict
                with st.spinner("🔄 Analyzing uploaded image..."):
                    prediction, confidence = predict_liveness(model, image)
                
                # Display results in col2
                with col2:
                    st.markdown("### 📊 Results")
                    
                    if prediction == "LIVE":
                        st.markdown(f'<p class="status-live">✅ STATUS: {prediction}</p>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<p class="status-spoof">❌ STATUS: {prediction}</p>', 
                                  unsafe_allow_html=True)
                    
                    st.markdown(f'<p class="confidence-score">🎯 Confidence: {confidence:.2%}</p>', 
                              unsafe_allow_html=True)
                    
                    # Confidence bar
                    st.progress(confidence)
                    
        with col2:
            if uploaded_file is None:
                st.info("📁 Upload an image to start detection")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
    🛡️ Anti-Spoofing Detection System | Built with Streamlit, PyTorch & WebRTC | 
    <a href='https://streamlit.io' target='_blank'>Streamlit Cloud Ready</a>
    </div>
    """, 
    unsafe_allow_html=True
) 