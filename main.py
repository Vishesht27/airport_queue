import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="IP Camera Queue Detection",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import everything else
import cv2
import numpy as np
import time
import json
from datetime import datetime
import torch
from PIL import Image
import pickle
import pandas as pd
import threading
import queue
import os
import random

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

# Custom CSS
st.markdown("""
<style>
    /* Hide Streamlit UI elements */
    .stDeployButton {display: none !important;}
    footer {visibility: hidden !important;}
    header[data-testid="stHeader"] {display: none !important;}
    .stToolbar {display: none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    div[data-testid="stDecoration"] {display: none !important;}
    div[data-testid="stStatusWidget"] {display: none !important;}
    #MainMenu {visibility: hidden !important;}
    
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

def save_detection_data(people_count, wait_time, model_name, timestamp=None, gate_name="GATE", gate_number="02", 
                       theme_option="Dark (Airport Standard)", font_size_multiplier=1.0, font_weight="Bold",
                       accent_color="#1f77b4", display_text_color="#ffffff", brightness_level=1.0, contrast_level=1.0):
    """Save detection data and display settings to JSON file for display page"""
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
        'last_updated': datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'),
        'gate_name': str(gate_name),
        'gate_number': str(gate_number),
        'display_settings': {
            'theme': str(theme_option),
            'font_size_multiplier': float(font_size_multiplier),
            'font_weight': str(font_weight),
            'accent_color': str(accent_color),
            'display_text_color': str(display_text_color),
            'brightness_level': float(brightness_level),
            'contrast_level': float(contrast_level)
        }
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

def resize_image_for_detection(image, quality_setting):
    """Resize image based on quality setting"""
    if quality_setting == "Smaller (Fast Processing)":
        target_size = (320, 240)
    elif quality_setting == "Medium (Balanced)":
        target_size = (640, 480)
    else:  # Original (Best Quality)
        return image  # No resizing
    
    # Resize while maintaining aspect ratio
    original_size = image.size
    original_ratio = original_size[0] / original_size[1]
    target_ratio = target_size[0] / target_size[1]
    
    if original_ratio > target_ratio:
        # Original is wider, fit to width
        new_width = target_size[0]
        new_height = int(target_size[0] / original_ratio)
    else:
        # Original is taller, fit to height
        new_height = target_size[1]
        new_width = int(target_size[1] * original_ratio)
    
    # Resize image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return resized_image

class IPCameraStream:
    def __init__(self):
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)
        self.capture_thread = None
        self.stream_url = None
        
    def connect_to_stream(self, stream_url, stream_type="rtsp"):
        """Connect to IP camera or NVR stream"""
        try:
            # Stop any existing stream
            self.stop_stream()
            
            self.stream_url = stream_url
            
            print(f"🔗 DEBUG: Attempting to connect to: {stream_url}")
            print(f"🔗 DEBUG: Stream type: {stream_type}")
            
            # Try to connect
            self.cap = cv2.VideoCapture(stream_url)
            
            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Set timeout properties if supported
            try:
                self.cap.set(cv2.CAP_PROP_TIMEOUT, 10000)  # 10 second timeout for HTTP
                print("🔗 DEBUG: Timeout set to 10 seconds")
            except Exception as timeout_err:
                print(f"🔗 DEBUG: Could not set timeout: {timeout_err}")
                pass
            
            print("🔗 DEBUG: Checking if camera opened...")
            if not self.cap.isOpened():
                print("🔗 DEBUG: Camera failed to open")
                raise Exception("Could not connect to stream")
            
            print("🔗 DEBUG: Camera opened successfully, testing frame read...")
            
            # Test read a frame with multiple attempts
            for attempt in range(3):
                print(f"🔗 DEBUG: Frame read attempt {attempt + 1}/3")
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    print(f"🔗 DEBUG: Frame read successful! Frame shape: {frame.shape}")
                    break
                else:
                    print(f"🔗 DEBUG: Frame read failed on attempt {attempt + 1}")
                    time.sleep(1)
            
            if not ret or frame is None:
                print("🔗 DEBUG: All frame read attempts failed")
                raise Exception("Could not read from stream after 3 attempts")
            
            self.is_running = True
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
            
            print("🔗 DEBUG: Connection successful, capture thread started")
            return True, "Connected successfully"
            
        except Exception as e:
            print(f"🔗 DEBUG: Connection error: {str(e)}")
            self.stop_stream()
            return False, f"Connection failed: {str(e)}"
    
    def stop_stream(self):
        """Stop the stream"""
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
                    # Keep only the latest frame
                    if not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    pass
            else:
                # Connection lost, try to reconnect
                time.sleep(1)
                if self.is_running and self.stream_url:
                    try:
                        self.cap = cv2.VideoCapture(self.stream_url)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except:
                        break
            
            time.sleep(0.033)  # ~30 FPS
    
    def get_latest_frame(self):
        """Get the most recent frame"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def is_connected(self):
        """Check if stream is connected"""
        return self.is_running and self.cap and self.cap.isOpened()

def build_stream_url(connection_type, ip_address, port, username, password, channel=1, stream_type="main", camera_brand="Generic"):
    """Build stream URL based on connection type and camera brand"""
    
    if connection_type == "Custom URL":
        # This will be handled separately
        return None
    
    elif connection_type == "NVR":
        # Common NVR RTSP formats
        if stream_type == "main":
            # Main stream (high quality)
            return f"rtsp://{username}:{password}@{ip_address}:{port}/cam/realmonitor?channel={channel}&subtype=0"
        else:
            # Sub stream (lower quality, better for detection)
            return f"rtsp://{username}:{password}@{ip_address}:{port}/cam/realmonitor?channel={channel}&subtype=1"
    
    elif connection_type == "Direct Camera":
        # Brand-specific RTSP URLs
        if camera_brand == "Pelco":
            if stream_type == "main":
                return f"rtsp://{username}:{password}@{ip_address}:{port}/rtsp/defaultPrimary?streamType=u"
            else:
                return f"rtsp://{username}:{password}@{ip_address}:{port}/rtsp/defaultSecondary?streamType=u"
        
        elif camera_brand == "Hikvision":
            if stream_type == "main":
                return f"rtsp://{username}:{password}@{ip_address}:{port}/Streaming/Channels/101/httppreview"
            else:
                return f"rtsp://{username}:{password}@{ip_address}:{port}/Streaming/Channels/102/httppreview"
        
        elif camera_brand == "Dahua":
            if stream_type == "main":
                return f"rtsp://{username}:{password}@{ip_address}:{port}/cam/realmonitor?channel=1&subtype=0"
            else:
                return f"rtsp://{username}:{password}@{ip_address}:{port}/cam/realmonitor?channel=1&subtype=1"
        
        elif camera_brand == "Axis":
            if stream_type == "main":
                return f"rtsp://{username}:{password}@{ip_address}:{port}/axis-media/media.amp?videocodec=h264"
            else:
                return f"rtsp://{username}:{password}@{ip_address}:{port}/axis-media/media.amp?resolution=320x240"
        
        else:  # Generic
            if stream_type == "main":
                return f"rtsp://{username}:{password}@{ip_address}:{port}/stream1"
            else:
                return f"rtsp://{username}:{password}@{ip_address}:{port}/stream2"
    
    elif connection_type == "HTTP Stream":
        # Brand-specific HTTP/MJPEG URLs
        if camera_brand == "Pelco":
            return f"http://{username}:{password}@{ip_address}:{port}/media/cam0/still.jpg"
        
        elif camera_brand == "Hikvision":
            return f"http://{username}:{password}@{ip_address}:{port}/ISAPI/Streaming/channels/101/picture"
        
        elif camera_brand == "Dahua":
            return f"http://{username}:{password}@{ip_address}:{port}/cgi-bin/mjpg/video.cgi"
        
        elif camera_brand == "Axis":
            return f"http://{username}:{password}@{ip_address}:{port}/axis-cgi/mjpg/video.cgi"
        
        else:  # Generic
            return f"http://{username}:{password}@{ip_address}:{port}/video.cgi"
    
    return None

def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">📹 IP Camera Queue Detection System</h1>', unsafe_allow_html=True)
    st.markdown("**Live detection from NVR or Direct IP Camera**")
    st.markdown("---")
    
    # Initialize components
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    if 'detector' not in st.session_state:
        st.session_state.detector = QueueDetector(st.session_state.model_manager)
    
    if 'ip_stream' not in st.session_state:
        st.session_state.ip_stream = IPCameraStream()
        st.session_state.stream_connected = False
        st.session_state.last_detection_time = 0
        st.session_state.detection_interval = 10
        st.session_state.detection_history = []
    
    # Load queue prediction model
    queue_model_data = load_queue_model()
    
    # Sidebar Configuration
    st.sidebar.header("📹 Camera Configuration")
    
    # Connection Type
    connection_type = st.sidebar.selectbox(
        "Connection Type",
        ["NVR", "Direct Camera", "HTTP Stream", "Custom URL"],
        help="Choose how to connect to your camera"
    )
    
    # Connection Details
    st.sidebar.subheader("Connection Details")
    
    if connection_type == "Custom URL":
        # Custom URL input
        custom_url = st.sidebar.text_area(
            "Stream URL",
            value="rtsp://admin:password@192.168.1.125:554/rtsp/defaultPrimary?streamType=u",
            help="Enter the complete stream URL (RTSP or HTTP)"
        )
        
        # Extract credentials for display (optional)
        st.sidebar.info("💡 Enter the complete URL with credentials")
        st.sidebar.markdown("**Examples:**")
        st.sidebar.code("rtsp://user:pass@ip:554/path")
        st.sidebar.code("http://user:pass@ip:80/path")
        
        # Set dummy values for compatibility
        ip_address = "custom"
        port = 554
        username = "custom"
        password = "custom"
        channel = 1
        stream_quality = "main"
        
    else:
        # Standard connection details
        # Pre-fill with your NVR IP
        default_ip = "192.168.1.165" if connection_type == "NVR" else "192.168.1.125"
        ip_address = st.sidebar.text_input("IP Address", value=default_ip)
        
        # Port selection based on connection type
        if connection_type == "NVR":
            default_port = 554
            port_help = "RTSP port (usually 554)"
        elif connection_type == "Direct Camera":
            default_port = 554
            port_help = "RTSP port (usually 554)"
        else:  # HTTP Stream
            default_port = 80
            port_help = "HTTP port (usually 80)"
        
        port = st.sidebar.number_input("Port", value=default_port, min_value=1, max_value=65535, help=port_help)
        
        # Authentication
        username = st.sidebar.text_input("Username", value="admin")
        password = st.sidebar.text_input("Password", type="password", value="")
        
        # Camera brand selection for smart URL building
        camera_brand = st.sidebar.selectbox(
            "Camera Brand",
            ["Generic", "Pelco", "Hikvision", "Dahua", "Axis"],
            help="Select your camera brand for optimized URL format"
        )
        
        # Additional options for NVR
        if connection_type == "NVR":
            channel = st.sidebar.number_input("Camera Channel", value=1, min_value=1, max_value=32, help="NVR camera channel number")
            stream_quality = st.sidebar.selectbox(
                "Stream Quality",
                ["sub", "main"],
                help="Sub-stream: Lower quality, better for detection. Main-stream: Higher quality"
            )
        else:
            channel = 1
            stream_quality = "main"
    
    # Model Configuration
    st.sidebar.header("🤖 Detection Settings")
    
    available_models = st.session_state.model_manager.get_available_models()
    if not available_models:
        st.sidebar.error("❌ No detection models available")
        return
    
    selected_model = st.sidebar.selectbox("🎯 Detection Model", available_models)
    confidence_threshold = st.sidebar.slider("🎚️ Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
    
    # Image quality selection
    image_quality = st.sidebar.radio(
        "📏 Image Quality",
        ["Smaller (Fast Processing)", "Medium (Balanced)", "Original (Best Quality)"],
        index=1,
        help="Choose processing speed vs accuracy trade-off"
    )
    
    # Show current settings
    quality_info = {
        "Smaller (Fast Processing)": "320×240 pixels - Very fast, good for real-time",
        "Medium (Balanced)": "640×480 pixels - Recommended balance",
        "Original (Best Quality)": "Full resolution - Highest accuracy, slower"
    }
    st.sidebar.info(f"📊 {quality_info[image_quality]}")
    
    # Show model info
    if selected_model:
        model_info = st.session_state.model_manager.model_info[selected_model]
        st.sidebar.markdown(f"**Description:** {model_info['description']}")
    
    # Detection Settings
    detection_interval = st.sidebar.selectbox(
        "Detection Interval (seconds)",
        [1, 3, 5, 8, 10, 15, 20],
        index=2,
        help="How often to run people detection (shorter = more real-time, more CPU usage)"
    )
    st.session_state.detection_interval = detection_interval
    
    # Wait time settings
    st.sidebar.header("⏱️ Wait Time Calculation")
    
    # Simple wait time method selection
    wait_time_method = st.sidebar.radio(
        "📊 How to calculate wait time?",
        ["Smart AI Prediction (Recommended)", "Custom AI"],
        index=0,
        help="Choose how to estimate queue wait times"
    )
    
    # Always estimate wait time now
    estimate_wait_time = True
    
    if wait_time_method == "Smart AI Prediction (Recommended)":
        use_ml_prediction = True
        
        # Time of day setting
        time_setting = st.sidebar.radio(
            "🕐 Time of day",
            ["Auto-detect current time", "Set manually"],
            index=0
        )
        
        if time_setting == "Set manually":
            current_hour = st.sidebar.slider("Current hour (24h format)", 0, 23, datetime.now().hour)
        else:
            current_hour = datetime.now().hour
            
        # Show time-based info
        if 7 <= current_hour <= 11:
            st.sidebar.warning("⏰ Peak Morning - Slower service expected")
        elif 15 <= current_hour <= 19:
            st.sidebar.warning("⏰ Peak Evening - Busy period")
        elif current_hour <= 7 or current_hour >= 20:
            st.sidebar.success("⏰ Off-Peak - Faster service")
        else:
            st.sidebar.info("⏰ Normal Hours - Standard service")
            
    else:  # Custom AI
        use_ml_prediction = False
        current_hour = datetime.now().hour
        
    # User can set seconds per person (always available)
    if not use_ml_prediction:
        st.sidebar.info("🎲 Custom AI generates random processing time between 40-50 seconds per person")
        seconds_per_person = None  # Will be generated randomly
    else:
        seconds_per_person = 40  # Default value when using ML
    
    # Display Settings
    st.sidebar.header("📺 Display Settings")
    gate_name = st.sidebar.text_input("Gate Name", value="GATE", help="Name to display (e.g., GATE, TERMINAL, CHECKPOINT)")
    gate_number = st.sidebar.text_input("Gate Number", value="02", help="Gate number to display")
    
    # People Count Adjustment
    st.sidebar.subheader("👥 People Count Adjustment")
    enable_people_adjustment = st.sidebar.checkbox("Enable people count adjustment", value=False, help="Add/subtract people from detected count")
    
    if enable_people_adjustment:
        people_adjustment = st.sidebar.slider(
            "Adjust detected count",
            min_value=-20,
            max_value=20,
            value=0,
            step=1,
            help="Add (+) or subtract (-) people from detected count before calculations"
        )
        
        if people_adjustment > 0:
            st.sidebar.success(f"✅ Adding {people_adjustment} people to detected count")
        elif people_adjustment < 0:
            st.sidebar.warning(f"⚠️ Subtracting {abs(people_adjustment)} people from detected count")
        else:
            st.sidebar.info("📊 Using exact detected count")
    else:
        people_adjustment = 0
    
    # Theme Settings
    st.sidebar.subheader("🎨 Display Theme")
    theme_option = st.sidebar.selectbox(
        "Theme",
        ["Dark (Airport Standard)", "Light (Bright Areas)", "High Contrast (Accessibility)"],
        index=0,
        help="Select display theme for different lighting conditions"
    )
    
    # Font Settings
    st.sidebar.subheader("🔤 Font Settings")
    font_size_multiplier = st.sidebar.slider(
        "Font Size",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Adjust font size (1.0 = normal, 2.0 = double size)"
    )
    
    font_weight = st.sidebar.selectbox(
        "Font Weight",
        ["Normal", "Bold", "Extra Bold"],
        index=1,
        help="Text thickness"
    )
    
    # Color Settings
    st.sidebar.subheader("🌈 Color Settings")
    accent_color = st.sidebar.color_picker(
        "Gate Name Color",
        value="#1f77b4",
        help="Color for gate name display"
    )
    
    display_text_color = st.sidebar.color_picker(
        "Text & Numbers Color",
        value="#ffffff",
        help="Color for all text and numbers (PEOPLE IN QUEUE, WAIT TIME, and their values)"
    )
    
    # Brightness Settings
    st.sidebar.subheader("💡 Brightness Settings")
    brightness_level = st.sidebar.slider(
        "Screen Brightness",
        min_value=0.3,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="Adjust overall display brightness"
    )
    
    contrast_level = st.sidebar.slider(
        "Contrast Level",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Adjust contrast for better visibility"
    )
    
    # Cropping settings
    st.sidebar.header("✂️ Video Cropping")
    enable_cropping = st.sidebar.checkbox("Enable live video cropping", value=False, help="Crop video feed to focus on specific queue area")
    
    if enable_cropping:
        st.sidebar.subheader("Crop Area")
        
        # Quick preset buttons
        preset_col1, preset_col2 = st.sidebar.columns(2)
        
        with preset_col1:
            if st.button("🎯 Center", key="crop_center"):
                st.session_state.crop_left = 15
                st.session_state.crop_top = 15
                st.session_state.crop_right = 85
                st.session_state.crop_bottom = 85
        
        with preset_col2:
            if st.button("🔄 Reset", key="crop_reset"):
                st.session_state.crop_left = 0
                st.session_state.crop_top = 0
                st.session_state.crop_right = 100
                st.session_state.crop_bottom = 100
        
        # Crop sliders
        crop_left = st.sidebar.slider(
            "Left (%)", 0, 90, 0, 5, key="crop_left",
            help="Left edge of crop area"
        )
        crop_top = st.sidebar.slider(
            "Top (%)", 0, 90, 0, 5, key="crop_top",
            help="Top edge of crop area"
        )
        crop_right = st.sidebar.slider(
            "Right (%)", 10, 100, 100, 5, key="crop_right",
            help="Right edge of crop area"
        )
        crop_bottom = st.sidebar.slider(
            "Bottom (%)", 10, 100, 100, 5, key="crop_bottom",
            help="Bottom edge of crop area"
        )
        
        # Validate crop area
        if crop_left >= crop_right or crop_top >= crop_bottom:
            st.sidebar.error("❌ Invalid crop area!")
            enable_cropping = False
        else:
            crop_area = ((crop_right - crop_left) * (crop_bottom - crop_top)) / 10000
            st.sidebar.info(f"📊 Crop area: {crop_area:.1%} of frame")
    else:
        crop_left = crop_top = 0
        crop_right = crop_bottom = 100
    
    # Connection Controls
    st.sidebar.header("🔌 Connection Control")
    
    # Build stream URL
    if connection_type == "Custom URL":
        stream_url = custom_url
        st.sidebar.code(stream_url, language="text")
    else:
        stream_url = build_stream_url(connection_type, ip_address, port, username, password, channel, stream_quality, camera_brand)
        
        if stream_url:
            st.sidebar.code(stream_url, language="text")
            
            # Show brand-specific info
            if camera_brand != "Generic":
                st.sidebar.success(f"✅ Using {camera_brand} optimized URL")
    
    # Connect/Disconnect buttons
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if not st.session_state.stream_connected:
            if st.button("🔗 Connect", type="primary"):
                if stream_url and username and password:
                    with st.spinner("Connecting to stream..."):
                        success, message = st.session_state.ip_stream.connect_to_stream(stream_url)
                        if success:
                            st.session_state.stream_connected = True
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                else:
                    st.error("Please fill in all connection details")
    
    with col2:
        if st.session_state.stream_connected:
            if st.button("🔌 Disconnect"):
                st.session_state.ip_stream.stop_stream()
                st.session_state.stream_connected = False
                st.success("Disconnected")
                st.rerun()
    
    # Main Content
    if st.session_state.stream_connected:
        # Stream Status
        if st.session_state.ip_stream.is_connected():
            st.success(f"✅ Connected to {connection_type} - {ip_address}:{port}")
        else:
            st.error("❌ Connection lost - attempting to reconnect...")
        
        # Video Display and Detection
        video_col, info_col = st.columns([2, 1])
        
        with video_col:
            video_placeholder = st.empty()
        
        with info_col:
            status_placeholder = st.empty()
            detection_placeholder = st.empty()
        
        # Get latest frame
        current_frame = st.session_state.ip_stream.get_latest_frame()
        current_time = time.time()
        
        if current_frame is not None:
            # Display frame
            frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            
            # Show crop overlay if cropping is enabled
            display_frame = frame_rgb.copy()
            caption_text = f"Live Stream - {connection_type} (Channel {channel})"
            
            if enable_cropping:
                # Draw crop area overlay on display frame
                h, w = display_frame.shape[:2]
                
                left = int(w * crop_left / 100)
                top = int(h * crop_top / 100)
                right = int(w * crop_right / 100)
                bottom = int(h * crop_bottom / 100)
                
                # Create semi-transparent overlay
                overlay = display_frame.copy()
                # Darken areas outside crop
                overlay[:top, :] = overlay[:top, :] * 0.3  # Top
                overlay[bottom:, :] = overlay[bottom:, :] * 0.3  # Bottom
                overlay[:, :left] = overlay[:, :left] * 0.3  # Left
                overlay[:, right:] = overlay[:, right:] * 0.3  # Right
                
                # Draw crop rectangle
                cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 0), 3)
                
                display_frame = overlay
                caption_text += " ✂️ (Cropped area highlighted)"
            
            with video_placeholder.container():
                st.image(
                    display_frame,
                    caption=caption_text,
                    use_column_width=True
                )
            
            # Detection logic
            time_since_last = current_time - st.session_state.last_detection_time
            
            if time_since_last >= st.session_state.detection_interval:
                st.session_state.last_detection_time = current_time
                
                with status_placeholder.container():
                    with st.spinner("🔍 Detecting people..."):
                        # Convert frame to PIL Image
                        pil_image = Image.fromarray(frame_rgb)
                        
                        # Apply cropping if enabled
                        final_image = pil_image
                        if enable_cropping:
                            final_image = crop_image(
                                pil_image,
                                crop_left, crop_top, crop_right, crop_bottom
                            )
                        
                        # Resize image based on quality setting
                        final_image = resize_image_for_detection(final_image, image_quality)
                        
                        # Run detection on final image (cropped and/or resized)
                        detection_results = st.session_state.detector.detect_people(
                            final_image, selected_model, confidence_threshold
                        )
                        
                        if detection_results:
                            # Apply people count adjustment
                            raw_people_count = detection_results['people_count']
                            people_count = max(0, raw_people_count + people_adjustment)  # Ensure non-negative
                            
                            # Calculate wait time
                            wait_time = None
                            if estimate_wait_time and people_count > 0:
                                if use_ml_prediction and queue_model_data:
                                    wait_time = predict_checkin_wait_time(
                                        queue_size=people_count,
                                        hour_of_day=current_hour,
                                        model_data=queue_model_data
                                    )
                                else:
                                    # Custom AI: Generate random seconds between 40-50
                                    random_seconds = random.randint(40, 50)
                                    wait_time = (people_count * random_seconds) / 60  # Random seconds per person in minutes
                            
                            # Save detection data for display page
                            save_detection_data(
                                people_count=people_count,
                                wait_time=wait_time,
                                model_name=selected_model,
                                timestamp=current_time,
                                gate_name=gate_name,
                                gate_number=gate_number,
                                theme_option=theme_option,
                                font_size_multiplier=font_size_multiplier,
                                font_weight=font_weight,
                                accent_color=accent_color,
                                display_text_color=display_text_color,
                                brightness_level=brightness_level,
                                contrast_level=contrast_level
                            )
                            
                            # Add to history
                            detection_record = {
                                'timestamp': current_time,
                                'people_count': people_count,
                                'model': selected_model,
                                'confidence': confidence_threshold,
                                'inference_time': detection_results['inference_time'],
                                'wait_time': wait_time
                            }
                            
                            st.session_state.detection_history.append(detection_record)
                            
                            # Keep only last 20 detections
                            if len(st.session_state.detection_history) > 20:
                                st.session_state.detection_history.pop(0)
                            
                            # Display results
                            with detection_placeholder.container():
                                crop_indicator = " ✂️" if enable_cropping else ""
                                quality_indicator = {
                                    "Smaller (Fast Processing)": " 🚀",
                                    "Medium (Balanced)": " ⚖️",
                                    "Original (Best Quality)": " 🔍"
                                }[image_quality]
                                
                                # Show adjustment info
                                adjustment_indicator = ""
                                if people_adjustment != 0:
                                    adjustment_indicator = f" (Raw: {raw_people_count}, Adjusted: {people_count})"
                                
                                st.success(f"✅ **{people_count} people detected{crop_indicator}{quality_indicator}**")
                                st.info(f"⚡ {detection_results['inference_time']:.2f}s | 🎯 {selected_model}")
                                
                                # Show processing info
                                if image_quality != "Original (Best Quality)":
                                    st.caption(f"📏 Processed at {quality_info[image_quality].split(' - ')[0]}")
                                
                                if adjustment_indicator:
                                    st.caption(f"👥 {adjustment_indicator}")
                                
                                # Wait time display
                                if wait_time is not None:
                                    if wait_time < 15:
                                        st.success(f"⏱️ **Wait time: {wait_time:.0f} minutes**")
                                    elif wait_time < 45:
                                        st.warning(f"⏱️ **Wait time: {wait_time:.0f} minutes**")
                                    else:
                                        st.error(f"⏱️ **Wait time: {wait_time:.0f} minutes**")
                        else:
                            with detection_placeholder.container():
                                st.error("❌ Detection failed")
            
            else:
                # Show countdown to next detection
                time_remaining = st.session_state.detection_interval - time_since_last
                
                with status_placeholder.container():
                    st.info(f"⏱️ Next detection in {time_remaining:.1f}s")
                    progress = 1 - (time_remaining / st.session_state.detection_interval)
                    st.progress(progress)
                    
                    # Show last detection if available
                    if st.session_state.detection_history:
                        last_detection = st.session_state.detection_history[-1]
                        people_count = last_detection['people_count']
                        wait_time = last_detection['wait_time']
                        
                        st.markdown(f"📊 **Last detection:** {people_count} people")
                        if wait_time is not None:
                            if wait_time < 15:
                                st.success(f"⏱️ **Last wait time:** {wait_time:.0f} minutes")
                            elif wait_time < 45:
                                st.warning(f"⏱️ **Last wait time:** {wait_time:.0f} minutes")
                            else:
                                st.error(f"⏱️ **Last wait time:** {wait_time:.0f} minutes")
        
        else:
            with video_placeholder.container():
                st.warning("⚠️ No video frame available")
        
        # Auto refresh
        time.sleep(0.5)
        st.rerun()
        
        # Detection History
        if len(st.session_state.detection_history) > 1:
            st.markdown("---")
            st.subheader("📈 Detection History")
            
            # Simple chart
            try:
                import plotly.graph_objects as go
                
                times = [
                    datetime.fromtimestamp(d['timestamp']).strftime('%H:%M:%S') 
                    for d in st.session_state.detection_history
                ]
                counts = [d['people_count'] for d in st.session_state.detection_history]
                
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
                    title=f"Live Detection History ({len(st.session_state.detection_history)} detections)",
                    xaxis_title="Time",
                    yaxis_title="Number of People",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                st.info("📊 Install plotly for detection history charts")
    
    else:
        # Not connected
        st.info("🔌 Please configure and connect to your camera stream using the sidebar")
        
        # Show connection examples
        st.subheader("📋 Connection Examples")
        
        example_col1, example_col2 = st.columns(2)
        
        with example_col1:
            st.markdown("**NVR Connection:**")
            st.code(f"""
IP: {default_ip}
Port: 554
Username: admin
Password: [your_password]
Channel: 1
Quality: sub (recommended for detection)
            """)
        
        with example_col2:
            st.markdown("**Direct Camera:**")
            st.code(f"""
IP: [camera_ip]
Port: 554
Username: admin
Password: [your_password]
Quality: main
            """)
        
        st.markdown("**Common RTSP URLs:**")
        st.code(f"""
NVR Main Stream: rtsp://admin:password@{default_ip}:554/cam/realmonitor?channel=1&subtype=0
NVR Sub Stream:  rtsp://admin:password@{default_ip}:554/cam/realmonitor?channel=1&subtype=1
Direct Camera:   rtsp://admin:password@camera_ip:554/stream1
        """)
    
    # Footer
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.markdown("**📹 Stream Status**")
        if st.session_state.stream_connected:
            st.markdown("✅ Connected")
            st.markdown(f"📡 {connection_type}")
            st.markdown(f"🏠 {ip_address}:{port}")
        else:
            st.markdown("❌ Disconnected")
    
    with status_col2:
        st.markdown("**🤖 Detection Models**")
        for model in available_models[:3]:  # Show first 3
            st.markdown(f"✅ {model}")
    
    with status_col3:
        st.markdown("**⚙️ Environment**")
        st.markdown(f"🔥 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        if queue_model_data:
            st.markdown("✅ ML queue prediction")
        else:
            st.markdown("❌ Simple calculation only")

if __name__ == "__main__":
    main()
