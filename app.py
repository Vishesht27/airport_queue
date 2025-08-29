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
import os


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

def save_detection_data(people_count, wait_time, model_name, timestamp=None):
    """Save detection data to JSON file for display page"""
    if timestamp is None:
        timestamp = time.time()
    
    # Convert NumPy types to native Python types for JSON serialization
    people_count = int(people_count) if people_count is not None else 0
    wait_time = float(wait_time) if wait_time is not None else None
    timestamp = float(timestamp)
    
    data = {
        'people_count': people_count,
        'wait_time': wait_time,
        'model': str(model_name),
        'timestamp': timestamp,
        'last_updated': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        # Write to temporary file first, then rename for atomic operation
        temp_file = 'detection_data.json.tmp'
        with open(temp_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Atomic rename to prevent corruption during write
        if os.path.exists(temp_file):
            os.replace(temp_file, 'detection_data.json')
            
    except Exception as e:
        st.warning(f"Could not save detection data: {e}")
        # Clean up temp file if it exists
        try:
            if os.path.exists('detection_data.json.tmp'):
                os.remove('detection_data.json.tmp')
        except:
            pass

def crop_image(image, left_percent, top_percent, right_percent, bottom_percent):
    """Crop image based on percentage coordinates"""
    width, height = image.size
    
    # Convert percentages to pixel coordinates
    left = int(width * left_percent / 100)
    top = int(height * top_percent / 100)
    right = int(width * right_percent / 100)
    bottom = int(height * bottom_percent / 100)
    
    # Ensure coordinates are within bounds
    left = max(0, min(left, width))
    top = max(0, min(top, height))
    right = max(left, min(right, width))
    bottom = max(top, min(bottom, height))
    
    # Crop the image
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image

def show_crop_controls(image, key_prefix=""):
    """Show cropping controls and return cropped image"""
    st.subheader("✂️ Crop Image")
    st.info("💡 Adjust the sliders to crop the image. This can help focus on the queue area.")
    
    # Quick preset options
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    
    with preset_col1:
        if st.button("🎯 Focus Center", key=f"{key_prefix}_center"):
            st.session_state[f"{key_prefix}_left"] = 15
            st.session_state[f"{key_prefix}_top"] = 15
            st.session_state[f"{key_prefix}_right"] = 85
            st.session_state[f"{key_prefix}_bottom"] = 85
    
    with preset_col2:
        if st.button("⬆️ Remove Top 25%", key=f"{key_prefix}_top25"):
            st.session_state[f"{key_prefix}_left"] = 0
            st.session_state[f"{key_prefix}_top"] = 25
            st.session_state[f"{key_prefix}_right"] = 100
            st.session_state[f"{key_prefix}_bottom"] = 100
    
    with preset_col3:
        if st.button("🔄 Reset", key=f"{key_prefix}_reset"):
            st.session_state[f"{key_prefix}_left"] = 0
            st.session_state[f"{key_prefix}_top"] = 0
            st.session_state[f"{key_prefix}_right"] = 100
            st.session_state[f"{key_prefix}_bottom"] = 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        left_percent = st.slider(
            "Left (%)", 
            min_value=0, max_value=90, value=0, step=5,
            key=f"{key_prefix}_left",
            help="Left edge of crop area"
        )
        top_percent = st.slider(
            "Top (%)", 
            min_value=0, max_value=90, value=0, step=5,
            key=f"{key_prefix}_top",
            help="Top edge of crop area"
        )
    
    with col2:
        right_percent = st.slider(
            "Right (%)", 
            min_value=10, max_value=100, value=100, step=5,
            key=f"{key_prefix}_right",
            help="Right edge of crop area"
        )
        bottom_percent = st.slider(
            "Bottom (%)", 
            min_value=10, max_value=100, value=100, step=5,
            key=f"{key_prefix}_bottom",
            help="Bottom edge of crop area"
        )
    
    # Validate crop area
    if left_percent >= right_percent:
        st.error("❌ Left must be less than Right")
        return None
    
    if top_percent >= bottom_percent:
        st.error("❌ Top must be less than Bottom")
        return None
    
    # Crop the image
    cropped_image = crop_image(image, left_percent, top_percent, right_percent, bottom_percent)
    
    # Show preview of cropped area
    col_preview1, col_preview2 = st.columns(2)
    
    with col_preview1:
        st.markdown("**Original Image**")
        st.image(image, use_column_width=True)
    
    with col_preview2:
        st.markdown("**Cropped Preview**")
        st.image(cropped_image, use_column_width=True)
        
        # Show crop info
        orig_width, orig_height = image.size
        crop_width, crop_height = cropped_image.size
        crop_ratio = (crop_width * crop_height) / (orig_width * orig_height)
        
        st.info(f"📏 Original: {orig_width}×{orig_height}")
        st.info(f"✂️ Cropped: {crop_width}×{crop_height}")
        st.info(f"📊 Area: {crop_ratio:.1%} of original")
    
    return cropped_image

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

# Check if running locally or on Streamlit Cloud
def is_local_environment():
    """Check if running locally or on Streamlit Cloud"""
    # Check for Streamlit Cloud environment variables
    return not (
        os.getenv('STREAMLIT_SHARING_MODE') or 
        os.getenv('STREAMLIT_SERVER_HEADLESS') == 'true' or
        'streamlit.app' in os.getenv('STREAMLIT_SERVER_ADDRESS', '') or
        'share.streamlit.io' in st.get_option('server.baseUrlPath') if hasattr(st, 'get_option') else False
    )

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

# Local environment: Use OpenCV camera capture (same as before)
class LiveVideoDetector:
    def __init__(self, model_manager, detector):
        self.model_manager = model_manager
        self.detector = detector
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.capture_thread = None
        
    def start_camera(self, camera_index=0):
        """Start the camera capture"""
        try:
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                raise Exception("Could not open camera")
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            self.is_running = True
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            
            return True
            
        except Exception as e:
            st.error(f"Failed to start camera: {e}")
            return False
    
    def stop_camera(self):
        """Stop the camera capture"""
        self.is_running = False
        
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2)
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear frame queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def _capture_frames(self):
        """Capture frames in background thread"""
        while self.is_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                try:
                    if not self.frame_queue.empty():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, timeout=0.1)
                except (queue.Full, queue.Empty):
                    pass
            else:
                break
            
            time.sleep(0.033)
    
    def get_latest_frame(self):
        """Get the most recent frame"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def is_camera_active(self):
        """Check if camera is active"""
        return self.is_running and self.cap and self.cap.isOpened()

# Browser-based camera for deployed apps
def browser_camera_section(selected_model, confidence_threshold, queue_model_data, current_hour, estimate_wait_time, use_ml_prediction, enable_cropping_globally):
    """Browser-based camera for deployed Streamlit apps"""
    
    st.subheader("📸 Browser Camera Capture")
    
    # Info about browser camera
    st.info("""
    🌐 **Deployed App Mode**: Using browser camera capture
    📱 Click 'Take Photo' to capture from your camera, then run detection
    🔄 You can take multiple photos and run detection on each
    """)
    
    # Initialize session state for browser camera
    if 'browser_detection_history' not in st.session_state:
        st.session_state.browser_detection_history = []
    
    # Camera input widget
    camera_photo = st.camera_input("📷 Take a photo of the queue")
    
    # Show last detection info when no photo is being taken
    if camera_photo is None and 'last_browser_detection' in st.session_state:
        st.markdown("---")
        st.subheader("📊 Last Detection Summary")
        
        last_detection = st.session_state.last_browser_detection
        people_count = last_detection['people_count']
        wait_time = last_detection['wait_time']
        detection_time = datetime.fromtimestamp(last_detection['timestamp']).strftime('%H:%M:%S')
        
        col_summary1, col_summary2, col_summary3 = st.columns(3)
        
        with col_summary1:
            st.metric("👥 Last Count", people_count)
        
        with col_summary2:
            if wait_time is not None:
                st.metric("⏱️ Last Wait Time", f"{wait_time:.0f} min")
            else:
                st.metric("⏱️ Last Wait Time", "N/A")
        
        with col_summary3:
            st.metric("🕐 Detection Time", detection_time)
        
        # Show wait time with color coding
        if wait_time is not None:
            if wait_time < 15:
                st.success(f"✅ **Last result: {people_count} people, {wait_time:.0f} minute wait**")
            elif wait_time < 45:
                st.warning(f"⚠️ **Last result: {people_count} people, {wait_time:.0f} minute wait**")
            else:
                st.error(f"🚨 **Last result: {people_count} people, {wait_time:.0f} minute wait**")
        else:
            st.info(f"📊 **Last result: {people_count} people detected**")
    
    if camera_photo is not None:
        # Convert camera input to PIL Image
        image = Image.open(camera_photo)
        
        # Display the captured photo
        st.image(image, caption="Captured Photo", use_column_width=True)
        
        # Cropping option (only show if globally enabled)
        final_image = image
        if enable_cropping_globally:
            enable_crop = st.checkbox("✂️ Enable Image Cropping", key="browser_crop_enable")
            if enable_crop:
                cropped_image = show_crop_controls(image, key_prefix="browser")
                if cropped_image is not None:
                    final_image = cropped_image
        
        # Detection button
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔍 Detect People in Photo", type="primary"):
                with st.spinner(f"Running {selected_model} detection..."):
                    # Run detection on final image (cropped or original)
                    results = st.session_state.detector.detect_people(
                        final_image, selected_model, confidence_threshold
                    )
                    
                    if results:
                        # Store results in session state
                        st.session_state.browser_detection_results = results
                        
                        # Calculate wait time for saving
                        current_time = time.time()
                        wait_time = None
                        if estimate_wait_time and results['people_count'] > 0:
                            if use_ml_prediction and queue_model_data:
                                wait_time = predict_checkin_wait_time(
                                    queue_size=results['people_count'],
                                    hour_of_day=current_hour,
                                    model_data=queue_model_data
                                )
                            else:
                                wait_time = (results['people_count'] * 3) / 3
                        
                        # Save detection data for display page
                        save_detection_data(
                            people_count=results['people_count'],
                            wait_time=wait_time,
                            model_name=selected_model,
                            timestamp=current_time
                        )
                        
                        # Add to history
                        detection_record = {
                            'timestamp': current_time,
                            'people_count': results['people_count'],
                            'model': selected_model,
                            'confidence': confidence_threshold,
                            'inference_time': results['inference_time']
                        }
                        
                        st.session_state.browser_detection_history.append(detection_record)
                        
                        # Keep only last 20 detections
                        if len(st.session_state.browser_detection_history) > 20:
                            st.session_state.browser_detection_history.pop(0)
                        
                        st.success(f"✅ Detection completed! Found {results['people_count']} people")
                        st.rerun()
                    else:
                        st.error("❌ Detection failed")
        
        with col2:
            if st.button("🗑️ Clear Photo"):
                # Force camera input to reset
                st.session_state.clear()
                st.rerun()
    
    # Show detection results if available
    if 'browser_detection_results' in st.session_state:
        results = st.session_state.browser_detection_results
        
        st.markdown("---")
        st.subheader("🔍 Latest Detection Results")
        
        # Display annotated image
        annotated_image_rgb = cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB)
        st.image(
            annotated_image_rgb, 
            caption=f"Detection Results - {results['model_name']}", 
            use_column_width=True
        )
        
        # Show metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("👥 People Count", results['people_count'])
        
        with metric_col2:
            st.metric("⚡ Inference Time", f"{results['inference_time']:.2f}s")
        
        with metric_col3:
            st.metric("🎯 Model Used", results['model_name'])
        
        # Wait time prediction
        if estimate_wait_time and results['people_count'] > 0:
            # ML prediction
            ml_wait_time = None
            if use_ml_prediction and queue_model_data:
                ml_wait_time = predict_checkin_wait_time(
                    queue_size=results['people_count'],
                    hour_of_day=current_hour,
                    model_data=queue_model_data
                )
            
            # Simple fallback calculation
            simple_wait_time = (results['people_count'] * 3) / 3  # minutes
            
            # Determine which wait time to display
            display_wait_time = float(ml_wait_time) if ml_wait_time is not None else simple_wait_time
            
            # Show wait time with color coding
            if display_wait_time < 15:
                st.success(f"✅ **Short wait: {display_wait_time:.0f} minutes**")
            elif display_wait_time < 45:
                st.warning(f"⚠️ **Moderate wait: {display_wait_time:.0f} minutes**")
            else:
                st.error(f"🚨 **Long wait: {display_wait_time:.0f} minutes**")
            
            # Show additional info
            st.info(f"📊 Queue: {results['people_count']} people | 🕐 Time: {current_hour}:00")
    
    # Detection history for browser camera
    if len(st.session_state.browser_detection_history) > 1:
        st.markdown("---")
        st.subheader("📈 Browser Detection History")
        
        # Create simple line chart
        try:
            import plotly.graph_objects as go
            
            times = [
                datetime.fromtimestamp(d['timestamp']).strftime('%H:%M:%S') 
                for d in st.session_state.browser_detection_history
            ]
            counts = [d['people_count'] for d in st.session_state.browser_detection_history]
            
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
                title=f"Browser Detection History ({len(st.session_state.browser_detection_history)} captures)",
                xaxis_title="Time",
                yaxis_title="Number of People",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info("📊 Install plotly for detection history charts")
        
        # Recent detections table
        if st.expander("📊 Recent Browser Detections", expanded=False):
            recent_detections = []
            for detection in reversed(st.session_state.browser_detection_history[-10:]):
                wait_time_str = f"{detection['wait_time']:.0f} min" if detection['wait_time'] is not None else "N/A"
                recent_detections.append({
                    'Time': datetime.fromtimestamp(detection['timestamp']).strftime('%H:%M:%S'),
                    'People': detection['people_count'],
                    'Wait Time': wait_time_str,
                    'Model': detection['model'],
                    'Confidence': detection['confidence'],
                    'Inference (s)': f"{detection['inference_time']:.2f}"
                })
            
            st.dataframe(recent_detections, use_container_width=True)
            
            # Clear history button
            if st.button("🗑️ Clear Browser History", key="clear_browser_history"):
                st.session_state.browser_detection_history = []
                st.success("✅ Browser detection history cleared!")
                st.rerun()

# Local live video section (same as before but simplified)
def local_live_video_section(selected_model, confidence_threshold, queue_model_data, current_hour, estimate_wait_time, use_ml_prediction, enable_cropping_globally):
    """Local live video detection section"""
    
    st.subheader("🎥 Live Local Camera")
    
    # Initialize video detector in session state
    if 'video_detector' not in st.session_state:
        st.session_state.video_detector = LiveVideoDetector(
            st.session_state.model_manager, 
            st.session_state.detector
        )
        st.session_state.camera_active = False
        st.session_state.last_detection_time = 0
        st.session_state.detection_interval = 10
        st.session_state.current_detection_results = None
        st.session_state.live_detection_history = []
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not st.session_state.camera_active:
            if st.button("🎬 Start Camera", type="primary"):
                if st.session_state.video_detector.start_camera():
                    st.session_state.camera_active = True
                    st.session_state.last_detection_time = 0
                    st.success("✅ Camera started!")
                    st.rerun()
        else:
            if st.button("⏹️ Stop Camera"):
                st.session_state.video_detector.stop_camera()
                st.session_state.camera_active = False
                st.success("✅ Camera stopped!")
                st.rerun()
    
    with col2:
        new_interval = st.selectbox(
            "Detection Interval", 
            [5, 8, 10, 15, 20], 
            index=2,
            key="detection_interval_select"
        )
        if new_interval != st.session_state.detection_interval:
            st.session_state.detection_interval = new_interval
    
    with col3:
        if st.session_state.camera_active:
            if st.session_state.video_detector.is_camera_active():
                st.success("🟢 Camera Active")
            else:
                st.error("🔴 Camera Error")
        else:
            st.info("⚪ Camera Off")
    
    # Cropping controls for live video (only when camera is active and cropping is enabled)
    if enable_cropping_globally and st.session_state.camera_active:
        st.markdown("---")
        enable_live_crop = st.checkbox("✂️ Enable Live Video Cropping", key="live_crop_enable")
        
        if enable_live_crop:
            # Initialize crop settings in session state if not exists or is None
            if ('live_crop_settings' not in st.session_state or 
                st.session_state.live_crop_settings is None):
                st.session_state.live_crop_settings = {
                    'left': 0, 'top': 0, 'right': 100, 'bottom': 100
                }
            
            st.info("💡 Crop settings will be applied to detection. Adjust and see results in real-time!")
            
            # Quick preset buttons
            preset_col1, preset_col2, preset_col3 = st.columns(3)
            
            with preset_col1:
                if st.button("🎯 Focus Center", key="live_center"):
                    st.session_state.live_crop_settings = {'left': 15, 'top': 15, 'right': 85, 'bottom': 85}
                    st.rerun()
            
            with preset_col2:
                if st.button("⬆️ Remove Top 25%", key="live_top25"):
                    st.session_state.live_crop_settings = {'left': 0, 'top': 25, 'right': 100, 'bottom': 100}
                    st.rerun()
            
            with preset_col3:
                if st.button("🔄 Reset", key="live_reset"):
                    st.session_state.live_crop_settings = {'left': 0, 'top': 0, 'right': 100, 'bottom': 100}
                    st.rerun()
            
            # Manual sliders
            crop_col1, crop_col2 = st.columns(2)
            
            with crop_col1:
                new_left = st.slider(
                    "Left (%)", min_value=0, max_value=90, 
                    value=st.session_state.live_crop_settings['left'], step=5, key="live_left"
                )
                new_top = st.slider(
                    "Top (%)", min_value=0, max_value=90, 
                    value=st.session_state.live_crop_settings['top'], step=5, key="live_top"
                )
            
            with crop_col2:
                new_right = st.slider(
                    "Right (%)", min_value=10, max_value=100, 
                    value=st.session_state.live_crop_settings['right'], step=5, key="live_right"
                )
                new_bottom = st.slider(
                    "Bottom (%)", min_value=10, max_value=100, 
                    value=st.session_state.live_crop_settings['bottom'], step=5, key="live_bottom"
                )
            
            # Update crop settings
            st.session_state.live_crop_settings.update({
                'left': new_left,
                'top': new_top,
                'right': new_right,
                'bottom': new_bottom
            })
            
            # Validate crop area
            if new_left >= new_right or new_top >= new_bottom:
                st.error("❌ Invalid crop area! Left must be < Right and Top must be < Bottom")
                st.session_state.live_crop_settings = {'left': 0, 'top': 0, 'right': 100, 'bottom': 100}
            else:
                # Show crop area info
                crop_area = ((new_right - new_left) * (new_bottom - new_top)) / 10000
                st.info(f"📊 Crop area: {crop_area:.1%} of original frame")
        else:
            # Don't set to None immediately, just mark as disabled
            if 'live_crop_settings' in st.session_state:
                st.session_state.live_crop_settings = None
    
    # Main video display and detection (same logic as before)
    if st.session_state.camera_active:
        video_col, info_col = st.columns([2, 1])
        
        with video_col:
            video_placeholder = st.empty()
        
        with info_col:
            status_placeholder = st.empty()
            detection_placeholder = st.empty()
        
        current_frame = st.session_state.video_detector.get_latest_frame()
        current_time = time.time()
        
        if current_frame is not None:
            frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            
            # Show crop overlay if cropping is enabled
            display_frame = frame_rgb.copy()
            caption_text = f"Live Video Feed (Detecting every {st.session_state.detection_interval}s)"
            
            if (enable_cropping_globally and 
                'live_crop_settings' in st.session_state and 
                st.session_state.live_crop_settings is not None):
                
                # Draw crop area overlay on display frame
                h, w = display_frame.shape[:2]
                crop_settings = st.session_state.live_crop_settings
                
                left = int(w * crop_settings['left'] / 100)
                top = int(h * crop_settings['top'] / 100)
                right = int(w * crop_settings['right'] / 100)
                bottom = int(h * crop_settings['bottom'] / 100)
                
                # Create semi-transparent overlay
                overlay = display_frame.copy()
                # Darken areas outside crop
                overlay[:top, :] = overlay[:top, :] * 0.3  # Top
                overlay[bottom:, :] = overlay[bottom:, :] * 0.3  # Bottom
                overlay[:, :left] = overlay[:, :left] * 0.3  # Left
                overlay[:, right:] = overlay[:, right:] * 0.3  # Right
                
                # Draw crop rectangle
                cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), 2)
                
                display_frame = overlay
                caption_text += " ✂️ (Cropped area highlighted)"
            
            with video_placeholder.container():
                st.image(
                    display_frame, 
                    caption=caption_text, 
                    use_column_width=True
                )
            
            time_since_last = current_time - st.session_state.last_detection_time
            
            if time_since_last >= st.session_state.detection_interval:
                st.session_state.last_detection_time = current_time
                
                with status_placeholder.container():
                    with st.spinner("🔍 Detecting people..."):
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # Apply cropping if enabled
                        final_image = pil_image
                        if (enable_cropping_globally and 
                            'live_crop_settings' in st.session_state and 
                            st.session_state.live_crop_settings is not None):
                            
                            crop_settings = st.session_state.live_crop_settings
                            final_image = crop_image(
                                pil_image,
                                crop_settings['left'], crop_settings['top'],
                                crop_settings['right'], crop_settings['bottom']
                            )
                        
                        detection_results = st.session_state.detector.detect_people(
                            final_image, selected_model, confidence_threshold
                        )
                        
                        if detection_results:
                            st.session_state.current_detection_results = detection_results
                            
                            # Calculate wait time for this detection
                            wait_time = None
                            if estimate_wait_time and detection_results['people_count'] > 0:
                                if use_ml_prediction and queue_model_data:
                                    wait_time = predict_checkin_wait_time(
                                        queue_size=detection_results['people_count'],
                                        hour_of_day=current_hour,
                                        model_data=queue_model_data
                                    )
                                else:
                                    wait_time = (detection_results['people_count'] * 3) / 3  # Simple calculation
                            
                            # Store last detection info for display
                            st.session_state.last_live_detection = {
                                'people_count': detection_results['people_count'],
                                'wait_time': wait_time,
                                'timestamp': current_time
                            }
                            
                            # Save detection data for display page
                            save_detection_data(
                                people_count=detection_results['people_count'],
                                wait_time=wait_time,
                                model_name=selected_model,
                                timestamp=current_time
                            )
                            
                            detection_record = {
                                'timestamp': current_time,
                                'people_count': detection_results['people_count'],
                                'model': selected_model,
                                'confidence': confidence_threshold,
                                'inference_time': detection_results['inference_time'],
                                'wait_time': wait_time
                            }
                            
                            st.session_state.live_detection_history.append(detection_record)
                            
                            if len(st.session_state.live_detection_history) > 20:
                                st.session_state.live_detection_history.pop(0)
                            
                            with detection_placeholder.container():
                                crop_indicator = ""
                                if (enable_cropping_globally and 
                                    'live_crop_settings' in st.session_state and 
                                    st.session_state.live_crop_settings is not None):
                                    crop_indicator = " ✂️"
                                
                                st.success(f"✅ **{detection_results['people_count']} people detected{crop_indicator}**")
                                st.info(f"⚡ {detection_results['inference_time']:.2f}s | 🎯 {selected_model}")
                                
                                # Wait time prediction
                                if estimate_wait_time and detection_results['people_count'] > 0:
                                    ml_wait_time = None
                                    if use_ml_prediction and queue_model_data:
                                        ml_wait_time = predict_checkin_wait_time(
                                            queue_size=detection_results['people_count'],
                                            hour_of_day=current_hour,
                                            model_data=queue_model_data
                                        )
                                    
                                    simple_wait_time = (detection_results['people_count'] * 3) / 3
                                    display_wait_time = float(ml_wait_time) if ml_wait_time is not None else simple_wait_time
                                    
                                    if display_wait_time < 15:
                                        st.success(f"⏱️ **Wait time: {display_wait_time:.0f} minutes**")
                                    elif display_wait_time < 45:
                                        st.warning(f"⏱️ **Wait time: {display_wait_time:.0f} minutes**")
                                    else:
                                        st.error(f"⏱️ **Wait time: {display_wait_time:.0f} minutes**")
                        else:
                            with detection_placeholder.container():
                                st.error("❌ Detection failed")
            
            else:
                time_remaining = st.session_state.detection_interval - time_since_last
                
                with status_placeholder.container():
                    st.info(f"⏱️ Next detection in {time_remaining:.1f}s")
                    progress = 1 - (time_remaining / st.session_state.detection_interval)
                    st.progress(progress)
                    
                    # Show last detection info if available
                    if 'last_live_detection' in st.session_state:
                        last_detection = st.session_state.last_live_detection
                        people_count = last_detection['people_count']
                        wait_time = last_detection['wait_time']
                        
                        if wait_time is not None:
                            st.markdown(f"📊 **Last detection:** {people_count} people")
                            if wait_time < 15:
                                st.success(f"⏱️ **Last wait time:** {wait_time:.0f} minutes")
                            elif wait_time < 45:
                                st.warning(f"⏱️ **Last wait time:** {wait_time:.0f} minutes")
                            else:
                                st.error(f"⏱️ **Last wait time:** {wait_time:.0f} minutes")
                        else:
                            st.markdown(f"📊 **Last detection:** {people_count} people")
                    elif st.session_state.current_detection_results:
                        people_count = st.session_state.current_detection_results['people_count']
                        st.markdown(f"📊 **Last detection:** {people_count} people")
        
        else:
            with video_placeholder.container():
                st.warning("⚠️ No video frame available")
        
        time.sleep(0.5)
        st.rerun()

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">✈️ Airport Check-in Queue Detection System</h1>', unsafe_allow_html=True)
    st.markdown("**Designed for initial check-in queues outside the airport**")
    
    # Check if running locally or deployed
    is_local = is_local_environment()
    
    if is_local:
        st.success("🖥️ **Local Mode**: Full live camera support enabled")
    else:
        st.info("🌐 **Deployed Mode**: Using browser camera capture")
    
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
        
        st.markdown("**✂️ Image Cropping**")
        enable_cropping_globally = st.checkbox("Enable image cropping", value=False, help="Show cropping controls for images and live video")
        if enable_cropping_globally:
            st.success("✅ Cropping controls will appear for:")
            st.write("• 📷 Camera captures")
            st.write("• 📁 Uploaded images") 
            st.write("• 🎥 Live video feeds")
        else:
            st.info("Crop images/video to focus on specific queue areas for better detection accuracy")
        
        # ML Queue prediction settings
        st.markdown("**🤖 Airport Check-in Queue Prediction**")
        use_ml_prediction = st.checkbox("Use ML prediction model", value=True)
        
        if use_ml_prediction and queue_model_data:
            st.success("✅ ML model loaded successfully")
            current_hour = st.slider("Current hour (24h format)", 
                                    min_value=0, max_value=23, 
                                    value=datetime.now().hour)
            
            # Show time-based info
            if 7 <= current_hour <= 11:
                st.warning("⏰ **Peak Morning Hours** - Expect slower service")
            elif 15 <= current_hour <= 19:
                st.warning("⏰ **Peak Evening Hours** - Busy period")
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
        st.header("📹 Camera Detection")
        
        # Choose between local live video or browser camera based on environment
        if is_local:
            local_live_video_section(
                selected_model, confidence_threshold, queue_model_data, 
                current_hour, estimate_wait_time, use_ml_prediction, enable_cropping_globally
            )
        else:
            browser_camera_section(
                selected_model, confidence_threshold, queue_model_data, 
                current_hour, estimate_wait_time, use_ml_prediction, enable_cropping_globally
            )
        
        # Image upload section
        st.markdown("---")
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
            
            # Cropping option (only show if globally enabled)
            final_image = image
            if enable_cropping_globally:
                enable_crop = st.checkbox("✂️ Enable Image Cropping", key="upload_crop_enable")
                if enable_crop:
                    cropped_image = show_crop_controls(image, key_prefix="upload")
                    if cropped_image is not None:
                        final_image = cropped_image
            
            # Detection button
            if st.button("🔍 Detect People in Queue", type="primary"):
                with st.spinner(f"Running {selected_model} detection..."):
                    # Run detection on final image (cropped or original)
                    results = st.session_state.detector.detect_people(
                        final_image, selected_model, confidence_threshold
                    )
                    
                    if results:
                        # Store results in session state
                        st.session_state.detection_results = results
                        
                        # Calculate wait time for saving
                        wait_time = None
                        if estimate_wait_time and results['people_count'] > 0:
                            if use_ml_prediction and queue_model_data:
                                wait_time = predict_checkin_wait_time(
                                    queue_size=results['people_count'],
                                    hour_of_day=current_hour,
                                    model_data=queue_model_data
                                )
                            else:
                                wait_time = (results['people_count'] * 3) / 3
                        
                        # Save detection data for display page
                        save_detection_data(
                            people_count=results['people_count'],
                            wait_time=wait_time,
                            model_name=selected_model
                        )
                        
                        st.success(f"✅ Detection completed! Found {results['people_count']} people")
                    else:
                        st.error("❌ Detection failed")
    
    with col2:
        st.header("📊 Detection Results")
        
        # Show uploaded image results if available
        if 'detection_results' in st.session_state:
            results = st.session_state.detection_results
            
            st.subheader("🖼️ Uploaded Image Results")
            
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
            
            # Wait time estimation (same as before)
            if estimate_wait_time and results['people_count'] > 0:
                ml_wait_time = None
                if use_ml_prediction and queue_model_data:
                    ml_wait_time = predict_checkin_wait_time(
                        queue_size=results['people_count'],
                        hour_of_day=current_hour,
                        model_data=queue_model_data
                    )
                
                simple_wait_time = (results['people_count'] * 3) / 3
                display_wait_time = float(ml_wait_time) if ml_wait_time is not None else simple_wait_time
                
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
                    ml_wait_time_float = float(ml_wait_time)
                    st.markdown(f"**🤖 Predicted wait time:** {ml_wait_time_float:.0f} minutes")
                    st.markdown(f"**📍 Queue size:** {results['people_count']} people")
                    st.markdown(f"**🕐 Current time:** {current_hour}:00")
                else:
                    st.markdown(f"**📊 Estimated wait time:** {simple_wait_time:.1f} minutes")
                    st.markdown("**⚠️ Note:** ML model not available, using simple calculation")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detection details (same as before)
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
                
                avg_confidence = np.mean([d['confidence'] for d in results['detections']])
                st.info(f"📊 Average detection confidence: {avg_confidence:.3f}")
            
        else:
            st.info("👆 Use camera or upload an image to see detection results")
    
    # Footer
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.markdown("**📦 Available Detection Models**")
        for model in available_models:
            st.markdown(f"✅ {model}")
    
    with status_col2:
        st.markdown("**⚙️ Environment**")
        st.markdown(f"🖥️ Local: {'✅' if is_local else '❌'}")
        st.markdown(f"🌐 Deployed: {'✅' if not is_local else '❌'}")
        st.markdown(f"🔥 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    with status_col3:
        st.markdown("**🤖 ML Queue Prediction**")
        if queue_model_data:
            st.markdown("✅ Airport check-in model loaded")
            st.markdown("🎯 Features: Queue size + Hour")
            if 'mae' in queue_model_data:
                st.markdown(f"📈 Model accuracy: {queue_model_data['mae']:.1f} min MAE")
        else:
            st.markdown("❌ Queue prediction model not available")

if __name__ == "__main__":
    main()