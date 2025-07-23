import streamlit as st
import cv2
import numpy as np
import time
import json
from datetime import datetime
from pathlib import Path
import torch
from PIL import Image
import io
import base64
import pickle
import pandas as pd
import threading
from collections import deque
import queue


# Load queue prediction model
@st.cache_resource
def load_queue_model():
    """Load the trained airport check-in queue wait prediction model"""
    try:
        with open('airport_checkin_queue_predictor.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load queue prediction model: {e}")
        st.info("💡 Please ensure 'airport_checkin_queue_predictor.pkl' is in your project directory")
        return None

def create_prediction_features(queue_size, hour_of_day):
    """Create features for the simplified airport check-in model"""
    data = pd.DataFrame({
        'queue_size': [queue_size],
        'hour_of_day': [hour_of_day]
    })
    
    # Same feature engineering as the new model
    data['hour_sin'] = np.sin(2 * np.pi * data['hour_of_day'] / 24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour_of_day'] / 24)
    data['queue_size_squared'] = data['queue_size'] ** 2
    data['queue_size_log'] = np.log1p(data['queue_size'])
    data['queue_hour_interaction'] = data['queue_size'] * data['hour_of_day']
    data['is_peak_morning'] = ((data['hour_of_day'] >= 7) & (data['hour_of_day'] <= 11)).astype(int)
    data['is_peak_evening'] = ((data['hour_of_day'] >= 15) & (data['hour_of_day'] <= 19)).astype(int)
    data['is_off_peak'] = ((data['hour_of_day'] <= 7) | (data['hour_of_day'] >= 20)).astype(int)
    data['is_small_queue'] = (data['queue_size'] <= 20).astype(int)
    data['is_medium_queue'] = ((data['queue_size'] > 20) & (data['queue_size'] <= 50)).astype(int)
    data['is_large_queue'] = ((data['queue_size'] > 50) & (data['queue_size'] <= 100)).astype(int)
    data['is_huge_queue'] = (data['queue_size'] > 100).astype(int)
    
    # Select the same features as training
    feature_cols = [
        'queue_size', 'hour_of_day', 'hour_sin', 'hour_cos',
        'queue_size_squared', 'queue_size_log', 'queue_hour_interaction',
        'is_peak_morning', 'is_peak_evening', 'is_off_peak',
        'is_small_queue', 'is_medium_queue', 'is_large_queue', 'is_huge_queue'
    ]
    
    return data[feature_cols]

def predict_checkin_wait_time(queue_size, hour_of_day, model_data):
    """Predict airport check-in wait time using the simplified model"""
    if model_data is None:
        return None
    
    try:
        # Create features
        X_input = create_prediction_features(queue_size, hour_of_day)
        
        # Make prediction
        prediction = model_data['model'].predict(X_input)[0]
        return max(2.0, prediction)  # Minimum 2 minutes wait
        
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Import model libraries
try:
    from ultralytics import YOLO, RTDETR
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import transformers
    from transformers import DetrImageProcessor, DetrForObjectDetection
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="Airport Queue Detection System",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-card {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
    }
    .warning-card {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
    }
    .error-card {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

class ModelManager:
    def __init__(self):
        self.models = {}
        self.model_info = {
            'YOLOv8m': {
                'file': 'yolov8m.pt',
                'type': 'yolo',
                'description': 'Medium YOLO model - Good balance of speed and accuracy',
                'requirements': ['ultralytics']
            },
            'YOLOv8x': {
                'file': 'yolov8x.pt',
                'type': 'yolo',
                'description': 'Extra large YOLO model - High accuracy',
                'requirements': ['ultralytics']
            },
            'YOLOv9e': {
                'file': 'yolov9e.pt',
                'type': 'yolo',
                'description': 'YOLO v9 Efficient - Latest improvements',
                'requirements': ['ultralytics']
            },
            'YOLOv10x': {
                'file': 'yolov10x.pt',
                'type': 'yolo',
                'description': 'YOLO v10 Extra large - Most advanced YOLO',
                'requirements': ['ultralytics']
            },
            'RT-DETR-X': {
                'file': 'rtdetr-x.pt',
                'type': 'rtdetr',
                'description': 'Real-time Detection Transformer - Superior crowd detection',
                'requirements': ['ultralytics']
            },
            'DETR': {
                'file': 'facebook/detr-resnet-50',
                'type': 'detr',
                'description': 'Detection Transformer - End-to-end detection',
                'requirements': ['transformers']
            }
        }
    
    def get_available_models(self):
        """Get list of available models based on installed packages"""
        available = []
        
        if ULTRALYTICS_AVAILABLE:
            available.extend(['YOLOv8m', 'YOLOv8x', 'YOLOv9e', 'YOLOv10x', 'RT-DETR-X'])
        
        if TRANSFORMERS_AVAILABLE:
            available.append('DETR')
        
        return available
    
    def load_model(self, model_name, progress_callback=None):
        """Load a specific model"""
        if model_name in self.models:
            return self.models[model_name]
        
        model_config = self.model_info[model_name]
        
        try:
            if progress_callback:
                progress_callback(f"Loading {model_name}...")
            
            if model_config['type'] == 'yolo':
                model = YOLO(model_config['file'])
            elif model_config['type'] == 'rtdetr':
                model = RTDETR(model_config['file'])
            elif model_config['type'] == 'detr':
                model = self._load_detr_model()
            else:
                raise ValueError(f"Unknown model type: {model_config['type']}")
            
            self.models[model_name] = model
            return model
            
        except Exception as e:
            st.error(f"Failed to load {model_name}: {str(e)}")
            return None
    
    def _load_detr_model(self):
        """Load DETR model"""
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        return {'processor': processor, 'model': model, 'device': device}

class QueueDetector:
    def __init__(self, model_manager):
        self.model_manager = model_manager
        self.coco_classes = [
            'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
            'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
            'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def detect_people(self, image, model_name, confidence_threshold=0.5):
        """Detect people in image using specified model"""
        
        # Load model
        model = self.model_manager.load_model(model_name)
        if model is None:
            return None
        
        # Convert PIL to CV2
        if isinstance(image, Image.Image):
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            image_cv = image
        
        start_time = time.time()
        
        # Run detection based on model type
        model_config = self.model_manager.model_info[model_name]
        
        if model_config['type'] in ['yolo', 'rtdetr']:
            return self._detect_ultralytics(image_cv, image, model, model_name, confidence_threshold)
        elif model_config['type'] == 'detr':
            return self._detect_detr(image_cv, image, model, model_name, confidence_threshold)
    
    def _detect_ultralytics(self, image_cv, image_pil, model, model_name, confidence_threshold):
        """Detect using YOLO/RT-DETR models"""
        start_time = time.time()
        
        # Run inference
        results = model(image_cv)
        inference_time = time.time() - start_time
        
        # Process results
        people_detections = []
        annotated_image = image_cv.copy()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    if int(box.cls[0]) == 0 and float(box.conf[0]) >= confidence_threshold:  # person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0])
                        
                        people_detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence
                        })
                        
                        # Draw bounding box
                        cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        
                        # Draw label
                        label = f'Person: {confidence:.2f}'
                        cv2.putText(annotated_image, label, (int(x1), int(y1) - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add title
        people_count = len(people_detections)
        title = f'{model_name}: {people_count} People Detected'
        cv2.putText(annotated_image, title, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return {
            'people_count': people_count,
            'detections': people_detections,
            'annotated_image': annotated_image,
            'inference_time': inference_time,
            'model_name': model_name
        }
    
    def _detect_detr(self, image_cv, image_pil, model, model_name, confidence_threshold):
        """Detect using DETR model"""
        start_time = time.time()
        
        # Preprocess image
        inputs = model['processor'](images=image_pil, return_tensors="pt")
        inputs = {k: v.to(model['device']) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = model['model'](**inputs)
        
        # Process results
        results = model['processor'].post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([image_pil.size[::-1]]).to(model['device']),
            threshold=confidence_threshold
        )
        
        inference_time = time.time() - start_time
        
        # Extract people detections
        people_detections = []
        annotated_image = image_cv.copy()
        
        for result in results:
            boxes = result['boxes'].cpu().numpy()
            labels = result['labels'].cpu().numpy()
            scores = result['scores'].cpu().numpy()
            
            for box, label, score in zip(boxes, labels, scores):
                if label == 1:  # person class in COCO
                    x1, y1, x2, y2 = box
                    people_detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(score)
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw label
                    label_text = f'Person: {score:.2f}'
                    cv2.putText(annotated_image, label_text, (int(x1), int(y1) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add title
        people_count = len(people_detections)
        title = f'{model_name}: {people_count} People Detected'
        cv2.putText(annotated_image, title, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return {
            'people_count': people_count,
            'detections': people_detections,
            'annotated_image': annotated_image,
            'inference_time': inference_time,
            'model_name': model_name
        }

class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.frame_queue = deque(maxlen=10)  # Store up to 10 frames
        self.is_running = False
        self.thread = None
        self.lock = threading.Lock()
        
        # Set camera properties for better performance
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.thread.start()

    def stop(self):
        self.is_running = False
        if self.cap.isOpened():
            self.cap.release()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)

    def _capture_frames(self):
        while self.is_running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.frame_queue.append(frame)
                else:
                    break
            else:
                break
            time.sleep(0.033)  # ~30 FPS capture rate
        
        if self.cap.isOpened():
            self.cap.release()

    def get_frame(self):
        with self.lock:
            if self.frame_queue:
                return self.frame_queue[-1].copy()  # Return a copy of the latest frame
        return None
    
    def is_active(self):
        return self.is_running and self.cap.isOpened()

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">✈️ Airport Check-in Queue Detection System</h1>', unsafe_allow_html=True)
    st.markdown("**Designed for initial check-in queues outside the airport**")
    st.markdown("---")
    
    # Initialize model manager and detector
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    if 'detector' not in st.session_state:
        st.session_state.detector = QueueDetector(st.session_state.model_manager)
    
    # Load queue prediction model
    queue_model_data = load_queue_model()
    
    # Sidebar
    st.sidebar.header("🔧 Configuration")
    
    # Video streaming info
    with st.sidebar.expander("ℹ️ Video Streaming Info"):
        st.markdown("""
        **🎥 Video Stream Features:**
        - Uses your MacBook's webcam as video source
        - Auto-detects people every 8 seconds (configurable)
        - Real-time video display with continuous updates
        - Automatic detection history tracking
        - Live queue monitoring
        
        **💡 Tips:**
        - Position camera to capture queue area
        - Ensure good lighting for better detection
        - Adjust confidence threshold as needed
        - Stream auto-refreshes every 2 seconds
        """)
    
    # Check available models
    available_models = st.session_state.model_manager.get_available_models()
    
    if not available_models:
        st.error("❌ No detection models available. Please install required packages:")
        st.code("""
        pip install ultralytics  # For YOLO models
        pip install transformers torch  # For DETR
        """)
        return
    
    # Model selection
    selected_model = st.sidebar.selectbox(
        "🤖 Select Detection Model",
        available_models,
        help="Choose which AI model to use for people detection"
    )
    
    # Show model info
    if selected_model:
        model_info = st.session_state.model_manager.model_info[selected_model]
        st.sidebar.markdown(f"**Description:** {model_info['description']}")
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "🎯 Confidence Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.1,
        help="Minimum confidence score for detections"
    )
    
    # Advanced options
    with st.sidebar.expander("⚙️ Settings"):
        show_details = st.checkbox("Show detection details", value=True)
        estimate_wait_time = st.checkbox("Estimate wait time", value=True)
        
        # Video stream settings
        st.markdown("**🎥 Video Stream Settings**")
        frame_interval = st.slider(
            "Frame sampling interval (seconds)",
            min_value=1,
            max_value=30,
            value=8,
            help="How often to sample frames from the video stream for detection"
        )
        
        # Update frame interval in session state
        if 'frame_interval' in st.session_state:
            st.session_state.frame_interval = frame_interval
        
        # ML Queue prediction settings
        st.markdown("**🤖 Airport Check-in Queue Prediction**")
        use_ml_prediction = st.checkbox("Use ML prediction model", value=True, 
                                       help="Uses queue size and current time to predict wait time")
        
        if use_ml_prediction and queue_model_data:
            st.success("✅ ML model loaded successfully")
            current_hour = st.slider("Current hour (24h format)", 
                                    min_value=0, max_value=23, 
                                    value=datetime.now().hour,
                                    help="Time affects service speed (peak vs off-peak hours)")
            
            # Show time-based info
            if 7 <= current_hour <= 11:
                st.warning("⏰ **Peak Morning Hours (7-11 AM)** - Expect slower service")
            elif 15 <= current_hour <= 19:
                st.warning("⏰ **Peak Evening Hours (3-7 PM)** - Busy period")
            elif current_hour <= 7 or current_hour >= 20:
                st.info("⏰ **Off-Peak Hours** - Faster service expected")
            else:
                st.info("⏰ **Normal Hours** - Standard service speed")
        elif use_ml_prediction:
            st.error("❌ ML model not available")
            current_hour = datetime.now().hour
        else:
            current_hour = datetime.now().hour
            st.info("📊 Using simple calculation only")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📹 Video Stream & Image Upload")
        
        # Video streaming section
        st.subheader("🎥 Live Video Stream")
        
        # Initialize video stream in session state
        if 'video_stream' not in st.session_state:
            st.session_state.video_stream = None
            st.session_state.stream_active = False
            st.session_state.last_frame_time = 0
            st.session_state.frame_interval = 8  # seconds
            st.session_state.frame_history = []  # Store recent frames with timestamps
            st.session_state.detection_history = []  # Store detection results
        
        # Video stream controls
        col_video1, col_video2, col_video3 = st.columns(3)
        
        with col_video1:
            if not st.session_state.stream_active:
                if st.button("🎬 Start Video Stream", type="primary"):
                    try:
                        st.session_state.video_stream = VideoStream(src=0)  # Use MacBook webcam
                        st.session_state.video_stream.start()
                        st.session_state.stream_active = True
                        st.session_state.last_frame_time = 0  # Force immediate first detection
                        st.success("✅ Video stream started! First detection will happen immediately.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Failed to start video stream: {e}")
        
        with col_video2:
            if st.session_state.stream_active:
                if st.button("⏹️ Stop Video Stream"):
                    if st.session_state.video_stream:
                        st.session_state.video_stream.stop()
                    st.session_state.stream_active = False
                    st.session_state.video_stream = None
                    st.success("✅ Video stream stopped!")
                    st.rerun()
        
        with col_video3:
            # Stream status indicator
            if st.session_state.stream_active:
                if st.session_state.video_stream and st.session_state.video_stream.is_active():
                    st.markdown("🟢 **Stream Active**")
                else:
                    st.markdown("🔴 **Stream Error**")
            else:
                st.markdown("⚪ **Stream Inactive**")
        
        # Display video stream with auto-refresh
        if st.session_state.stream_active and st.session_state.video_stream:
            # Create placeholders for dynamic content
            video_placeholder = st.empty()
            status_placeholder = st.empty()
            detection_placeholder = st.empty()
            
            # Get current frame
            current_frame = st.session_state.video_stream.get_frame()
            current_time = time.time()
            
            if current_frame is not None:
                # Convert frame to RGB for display
                frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
                
                # Always display the current frame
                video_placeholder.image(frame_rgb, caption=f"Live Video Stream (Auto-detecting every {st.session_state.frame_interval} seconds)", use_column_width=True)
                
                # Check if it's time to run detection (every 8 seconds)
                # Initialize last_frame_time if not set
                if 'last_frame_time' not in st.session_state:
                    st.session_state.last_frame_time = 0
                
                # Debug: Show current timing info
                time_since_last_detection = current_time - st.session_state.last_frame_time
                st.markdown(f"**Debug:** Time since last detection: {time_since_last_detection:.1f}s / {st.session_state.frame_interval}s")
                
                # Check if enough time has passed since last detection
                if time_since_last_detection >= st.session_state.frame_interval:
                    st.session_state.last_frame_time = current_time
                    
                    # Run detection
                    with status_placeholder.container():
                        with st.spinner("🔍 Detecting people in current frame..."):
                            results = st.session_state.detector.detect_people(
                                Image.fromarray(frame_rgb), selected_model, confidence_threshold
                            )
                            
                            if results:
                                st.session_state.detection_results = results
                                st.session_state.last_detection_time = current_time
                                
                                # Store in history
                                detection_record = {
                                    'timestamp': current_time,
                                    'people_count': results['people_count'],
                                    'model_used': selected_model,
                                    'confidence_threshold': confidence_threshold
                                }
                                st.session_state.detection_history.append(detection_record)
                                
                                # Keep only last 10 detections
                                if len(st.session_state.detection_history) > 10:
                                    st.session_state.detection_history.pop(0)
                                
                                # Show detection results with more details
                                col_result1, col_result2 = st.columns(2)
                                with col_result1:
                                    st.success(f"✅ **{results['people_count']} people detected**")
                                with col_result2:
                                    st.info(f"⚡ {results['inference_time']:.2f}s | 🎯 {selected_model}")
                            else:
                                st.error("❌ Detection failed")
                else:
                    # Show countdown and last detection info
                    time_until_next = st.session_state.frame_interval - (current_time - st.session_state.last_frame_time)
                    
                    with status_placeholder.container():
                        # Show countdown with progress bar
                        st.markdown(f"⏱️ **Next detection in {time_until_next:.1f} seconds**")
                        
                        # Progress bar for countdown
                        progress = 1 - (time_until_next / st.session_state.frame_interval)
                        st.progress(progress)
                        
                        # Show last detection info if available
                        if 'detection_results' in st.session_state and 'last_detection_time' in st.session_state:
                            time_since_detection = current_time - st.session_state.last_detection_time
                            st.info(f"📊 **Last detection:** {st.session_state.detection_results['people_count']} people ({time_since_detection:.1f}s ago)")
                
            else:
                st.warning("⚠️ No video frame available")
        
        # Add auto-refresh for video stream
        if st.session_state.stream_active:
            # Track refresh time in session state
            if 'last_refresh_time' not in st.session_state:
                st.session_state.last_refresh_time = time.time()
            
            current_time = time.time()
            time_since_refresh = current_time - st.session_state.last_refresh_time
            
            # Show refresh status
            st.markdown(f"🔄 Auto-refreshing... (Last refresh: {time_since_refresh:.1f}s ago)")
            
            # Manual refresh button for testing
            if st.button("🔄 Manual Refresh", key="manual_refresh"):
                st.session_state.last_refresh_time = current_time
                st.rerun()
            
            # Force refresh every 3 seconds using JavaScript
            if time_since_refresh >= 3:
                st.session_state.last_refresh_time = current_time
                st.markdown("""
                <script>
                    setTimeout(function(){
                        window.location.reload();
                    }, 100);
                </script>
                """, unsafe_allow_html=True)
        
        # Video stream statistics
        if st.session_state.stream_active and st.session_state.video_stream:
            st.subheader("📊 Video Stream Statistics")
            
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            with col_stats1:
                st.metric("Frame Interval", f"{st.session_state.frame_interval}s")
            
            with col_stats2:
                if 'last_detection_time' in st.session_state:
                    time_since_last = time.time() - st.session_state.last_detection_time
                    st.metric("Last Detection", f"{time_since_last:.1f}s ago")
                else:
                    st.metric("Last Detection", "None")
            
            with col_stats3:
                if st.session_state.video_stream.is_active():
                    st.metric("Stream Status", "🟢 Active")
                else:
                    st.metric("Stream Status", "🔴 Error")
            
            # Debug information
            if st.session_state.stream_active:
                st.subheader("🐛 Debug Information")
                col_debug1, col_debug2 = st.columns(2)
                
                with col_debug1:
                    st.markdown(f"**Current Time:** {datetime.now().strftime('%H:%M:%S')}")
                    st.markdown(f"**Last Frame Time:** {st.session_state.last_frame_time:.1f}")
                    st.markdown(f"**Frame Interval:** {st.session_state.frame_interval}s")
                
                with col_debug2:
                    if 'detection_results' in st.session_state:
                        st.markdown(f"**Last Detection:** {st.session_state.detection_results['people_count']} people")
                        st.markdown(f"**Detection Time:** {st.session_state.last_detection_time:.1f}")
                    else:
                        st.markdown("**Last Detection:** None")
                    
                    st.markdown(f"**Detection History Count:** {len(st.session_state.detection_history)}")
            
            # Detection history
            if st.session_state.detection_history:
                st.subheader("📈 Recent Detections")
                
                # Create a simple chart of recent detections
                if len(st.session_state.detection_history) > 1:
                    import plotly.express as px
                    import plotly.graph_objects as go
                    
                    # Prepare data for plotting
                    times = [datetime.fromtimestamp(d['timestamp']).strftime('%H:%M:%S') for d in st.session_state.detection_history]
                    counts = [d['people_count'] for d in st.session_state.detection_history]
                    
                    # Create line chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=times,
                        y=counts,
                        mode='lines+markers',
                        name='People Count',
                        line=dict(color='#1f77b4', width=3),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title="People Count Over Time",
                        xaxis_title="Time",
                        yaxis_title="Number of People",
                        height=300,
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show recent detection table
                st.markdown("**Recent Detection Results:**")
                detection_data = []
                for i, detection in enumerate(reversed(st.session_state.detection_history[-5:])):  # Show last 5
                    time_str = datetime.fromtimestamp(detection['timestamp']).strftime('%H:%M:%S')
                    detection_data.append({
                        'Time': time_str,
                        'People Count': detection['people_count'],
                        'Model': detection['model_used'],
                        'Confidence': detection['confidence_threshold']
                    })
                
                st.dataframe(detection_data, use_container_width=True)
                
                # Clear history button
                if st.button("🗑️ Clear Detection History", key="clear_history"):
                    st.session_state.detection_history = []
                    st.success("✅ Detection history cleared!")
                    st.rerun()
        
        # Image upload section
        st.subheader("📁 Upload Queue Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image of people waiting in the airport check-in queue"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Detection button
            if st.button("🔍 Detect People in Queue", type="primary"):
                with st.spinner(f"Running {selected_model} detection..."):
                    # Run detection
                    results = st.session_state.detector.detect_people(
                        image, selected_model, confidence_threshold
                    )
                    
                    if results:
                        # Store results in session state
                        st.session_state.detection_results = results
                        st.success(f"✅ Detection completed! Found {results['people_count']} people")
                    else:
                        st.error("❌ Detection failed")
    
    with col2:
        st.header("📊 Detection Results")
        
        if 'detection_results' in st.session_state:
            results = st.session_state.detection_results
            
            # Display annotated image
            annotated_image_rgb = cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB)
            st.image(annotated_image_rgb, caption=f"Detection Results - {results['model_name']}", use_column_width=True)
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("👥 People Count", results['people_count'])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_b:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("⚡ Inference Time", f"{results['inference_time']:.2f}s")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col_c:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🚀 FPS", f"{1/results['inference_time']:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Wait time estimation
            if estimate_wait_time and results['people_count'] > 0:
                # ML prediction
                ml_wait_time = None
                if use_ml_prediction and queue_model_data:
                    ml_wait_time = predict_checkin_wait_time(
                        queue_size=results['people_count'],
                        hour_of_day=current_hour,
                        model_data=queue_model_data
                    )
                
                # Simple fallback calculation (3 minutes per person, 3 counters)
                simple_wait_time = (results['people_count'] * 3) / 3  # minutes
                
                # Determine which wait time to display
                display_wait_time = float(ml_wait_time) if ml_wait_time is not None else simple_wait_time
                
                # Color coding based on wait time
                if display_wait_time < 15:
                    card_class = "success-card"
                    status = "✅ Short wait"
                elif display_wait_time < 45:
                    card_class = "warning-card"
                    status = "⚠️ Moderate wait"
                else:
                    card_class = "error-card"
                    status = "🚨 Long wait"
                
                st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                st.markdown(f"**{status}**")
                
                if ml_wait_time is not None:
                    # Display ML prediction
                    ml_wait_time_float = float(ml_wait_time)
                    st.markdown(f"**🤖 Predicted wait time:** {ml_wait_time_float:.0f} minutes")
                    st.markdown(f"**📍 Queue size:** {results['people_count']} people")
                    st.markdown(f"**🕐 Current time:** {current_hour}:00")
                    
                    # Show insights based on queue size
                    if results['people_count'] <= 20:
                        st.markdown("💡 **Small queue** - Relatively quick processing")
                    elif results['people_count'] <= 50:
                        st.markdown("💡 **Medium queue** - Moderate wait expected")
                    elif results['people_count'] <= 100:
                        st.markdown("💡 **Large queue** - Significant wait time")
                    else:
                        st.markdown("💡 **Very large queue** - Consider alternative timing")
                    
                    # Show time-based insights
                    if 7 <= current_hour <= 11:
                        st.markdown("⏰ **Morning rush factor included** - Peak travel time")
                    elif 15 <= current_hour <= 19:
                        st.markdown("⏰ **Evening rush factor included** - Busy period")
                    elif current_hour <= 7 or current_hour >= 20:
                        st.markdown("⏰ **Off-peak hours** - Faster service")
                
                else:
                    # Fallback to simple calculation
                    st.markdown(f"**📊 Estimated wait time:** {simple_wait_time:.1f} minutes")
                    st.markdown("**⚠️ Note:** ML model not available, using simple calculation")
                    st.markdown("**📍 Assumption:** 3 minutes per person, 3 service counters")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detection details
            if show_details and results['detections']:
                st.subheader("🔍 Detection Details")
                
                detection_data = []
                for i, detection in enumerate(results['detections']):
                    detection_data.append({
                        'Person #': i + 1,
                        'Confidence': f"{detection['confidence']:.3f}",
                        'Bounding Box': f"({detection['bbox'][0]}, {detection['bbox'][1]}) to ({detection['bbox'][2]}, {detection['bbox'][3]})"
                    })
                
                st.dataframe(detection_data, use_container_width=True)
                
                # Average confidence
                avg_confidence = np.mean([d['confidence'] for d in results['detections']])
                st.info(f"📊 Average detection confidence: {avg_confidence:.3f}")
            
            # Download results
            st.subheader("💾 Download Results")
            
            # Create download data
            download_data = {
                'timestamp': datetime.now().isoformat(),
                'model_used': results['model_name'],
                'people_count': results['people_count'],
                'confidence_threshold': confidence_threshold,
                'inference_time': results['inference_time'],
                'detections': results['detections'],
                'queue_prediction': {}
            }
            
            # Add ML prediction data if available
            if estimate_wait_time and results['people_count'] > 0:
                if use_ml_prediction and queue_model_data:
                    ml_wait_time = predict_checkin_wait_time(
                        queue_size=results['people_count'],
                        hour_of_day=current_hour,
                        model_data=queue_model_data
                    )
                    
                    if ml_wait_time is not None:
                        download_data['queue_prediction'] = {
                            'ml_prediction_minutes': float(ml_wait_time),
                            'queue_size': results['people_count'],
                            'hour_of_day': current_hour,
                            'model_type': 'airport_checkin_simplified',
                            'features_used': ['queue_size', 'hour_of_day']
                        }
            
            col_d, col_e = st.columns(2)
            
            with col_d:
                # Download JSON
                json_str = json.dumps(download_data, indent=2)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name=f"detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            
            with col_e:
                # Download annotated image
                _, buffer = cv2.imencode('.jpg', results['annotated_image'])
                st.download_button(
                    label="🖼️ Download Image",
                    data=buffer.tobytes(),
                    file_name=f"annotated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
                    mime="image/jpeg"
                )
        
        else:
            st.info("👆 Upload an image and click 'Detect People in Queue' to see results")
    
    # Footer
    st.markdown("---")
    
    # System Status
    st.subheader("🔧 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.markdown("**📦 Available Detection Models**")
        for model in available_models:
            st.markdown(f"✅ {model}")
    
    with status_col2:
        st.markdown("**⚙️ Dependencies**")
        st.markdown(f"✅ Ultralytics: {ULTRALYTICS_AVAILABLE}")
        st.markdown(f"{'✅' if TRANSFORMERS_AVAILABLE else '❌'} Transformers: {TRANSFORMERS_AVAILABLE}")
        st.markdown(f"🖥️ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    with status_col3:
        st.markdown("**🤖 ML Queue Prediction**")
        if queue_model_data:
            st.markdown("✅ Airport check-in model loaded")
            st.markdown("🎯 Features: Queue size + Hour")
            st.markdown("📊 Realistic wait times (2-240 min)")
            if 'mae' in queue_model_data:
                st.markdown(f"📈 Model accuracy: {queue_model_data['mae']:.1f} min MAE")
        else:
            st.markdown("❌ Queue prediction model not available")
            st.markdown("💡 Train model in Google Colab first")
            st.markdown("📁 Save as 'airport_checkin_queue_predictor.pkl'")
    
    # Add video streaming status
    status_col4, status_col5 = st.columns(2)
    
    with status_col4:
        st.markdown("**🎥 Video Streaming**")
        if 'stream_active' in st.session_state and st.session_state.stream_active:
            if st.session_state.video_stream and st.session_state.video_stream.is_active():
                st.markdown("🟢 Stream Active")
                st.markdown(f"⏱️ Frame interval: {st.session_state.frame_interval}s")
                if 'detection_history' in st.session_state:
                    st.markdown(f"📊 Detections: {len(st.session_state.detection_history)}")
            else:
                st.markdown("🔴 Stream Error")
        else:
            st.markdown("⚪ Stream Inactive")
    
    with status_col5:
        st.markdown("**📈 Detection History**")
        if 'detection_history' in st.session_state and st.session_state.detection_history:
            latest_count = st.session_state.detection_history[-1]['people_count']
            st.markdown(f"👥 Latest: {latest_count} people")
            st.markdown(f"📊 Total: {len(st.session_state.detection_history)} detections")
        else:
            st.markdown("📊 No detections yet")
    
    # Add example predictions section
    if queue_model_data:
        st.markdown("---")
        st.subheader("📈 Example Wait Time Predictions")
        
        # Create example scenarios
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            st.markdown("**🌅 Morning Scenarios**")
            
            morning_examples = [
                (10, 7, "10 people at 7 AM"),
                (25, 8, "25 people at 8 AM"), 
                (40, 9, "40 people at 9 AM"),
                (60, 10, "60 people at 10 AM")
            ]
            
            for queue_size, hour, description in morning_examples:
                wait_time = predict_checkin_wait_time(queue_size, hour, queue_model_data)
                if wait_time:
                    if wait_time < 15:
                        icon = "🟢"
                    elif wait_time < 45:
                        icon = "🟡"
                    else:
                        icon = "🔴"
                    st.markdown(f"{icon} {description}: **{wait_time:.0f} min**")
        
        with example_col2:
            st.markdown("**🌆 Evening Scenarios**")
            
            evening_examples = [
                (15, 17, "15 people at 5 PM"),
                (30, 18, "30 people at 6 PM"),
                (50, 19, "50 people at 7 PM"),
                (20, 22, "20 people at 10 PM")
            ]
            
            for queue_size, hour, description in evening_examples:
                wait_time = predict_checkin_wait_time(queue_size, hour, queue_model_data)
                if wait_time:
                    if wait_time < 15:
                        icon = "🟢"
                    elif wait_time < 45:
                        icon = "🟡"
                    else:
                        icon = "🔴"
                    st.markdown(f"{icon} {description}: **{wait_time:.0f} min**")
        
        # Color legend
        st.markdown("**Legend:** 🟢 Short (<15 min) | 🟡 Moderate (15-45 min) | 🔴 Long (>45 min)")

if __name__ == "__main__":
    main()