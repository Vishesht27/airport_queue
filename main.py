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
import tempfile

# WebRTC imports
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

# File-based shared storage for cross-thread communication - LOCAL PROJECT DIRECTORY
try:
    # Try current directory first (local development)
    WEBRTC_RESULTS_FILE = os.path.join(os.getcwd(), "webrtc_results.json")
    # Test write permissions
    with open(WEBRTC_RESULTS_FILE, 'w') as test_file:
        test_file.write('{"test": true}')
    os.remove(WEBRTC_RESULTS_FILE)  # Clean up test file
    print(f"[INIT] Using project directory for results: {WEBRTC_RESULTS_FILE}")
except:
    # Fallback to temp directory (deployment environments)
    WEBRTC_RESULTS_FILE = os.path.join(tempfile.gettempdir(), "webrtc_results.json")
    print(f"[INIT] Fallback to temp directory: {WEBRTC_RESULTS_FILE}")

def write_webrtc_results(detection_data):
    """Write WebRTC results to shared file - JSON-safe version"""
    try:
        # Create JSON-safe data by excluding numpy arrays and other non-serializable objects
        safe_data = {}
        
        # Copy basic data that's JSON serializable
        for key, value in detection_data.items():
            if key == 'annotated_image':
                # Skip the numpy array image
                continue
            elif key == 'detections':
                # Make detections JSON-safe by converting numpy values
                safe_detections = []
                for det in value:
                    safe_det = {
                        'bbox': [int(x) for x in det['bbox']],  # Convert numpy to int
                        'confidence': float(det['confidence'])  # Convert numpy to float
                    }
                    safe_detections.append(safe_det)
                safe_data[key] = safe_detections
            elif isinstance(value, (str, int, float, bool, list, dict)):
                # Only include basic JSON-serializable types
                safe_data[key] = value
            elif hasattr(value, 'item'):
                # Convert numpy scalars to Python types
                safe_data[key] = value.item()
            else:
                # Skip any other complex objects
                print(f"[WebRTC] Skipping non-serializable field: {key} (type: {type(value)})")
        
        # Create the file data structure
        file_data = {
            'latest_result': safe_data,
            'detection_count': safe_data.get('detection_count', 0),
            'last_update': time.time(),
            'timestamp': safe_data.get('timestamp', 0),
            'people_count': safe_data.get('people_count', 0),
            'model_name': safe_data.get('model_name', 'unknown'),
            'inference_time': safe_data.get('inference_time', 0),
            'wait_time': safe_data.get('wait_time', 0)
        }
        
        with open(WEBRTC_RESULTS_FILE, 'w') as f:
            json.dump(file_data, f, indent=2)
        
        print(f"[WebRTC] FILE WRITE SUCCESS: detection_count = {file_data['detection_count']}")
        print(f"[WebRTC] FILE WRITE SUCCESS: people_count = {file_data['people_count']}")
        return True
        
    except Exception as e:
        print(f"[WebRTC] FILE WRITE ERROR: {e}")
        return False

def read_webrtc_results():
    """Read WebRTC results from shared file - Improved version"""
    try:
        if not os.path.exists(WEBRTC_RESULTS_FILE):
            print(f"[UI] Results file doesn't exist yet: {WEBRTC_RESULTS_FILE}")
            return None
        
        # Check file size
        file_size = os.path.getsize(WEBRTC_RESULTS_FILE)
        if file_size == 0:
            print(f"[UI] Results file is empty")
            return None
            
        with open(WEBRTC_RESULTS_FILE, 'r') as f:
            data = json.load(f)
        
        # Validate the data structure
        if not isinstance(data, dict):
            print(f"[UI] Invalid file format - not a dictionary")
            return None
            
        detection_count = data.get('detection_count', 0)
        people_count = data.get('people_count', 0)
        
        print(f"[UI] FILE READ SUCCESS:")
        print(f"[UI] - detection_count: {detection_count}")
        print(f"[UI] - people_count: {people_count}")
        print(f"[UI] - file_size: {file_size} bytes")
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"[UI] FILE READ JSON ERROR: {e}")
        print(f"[UI] File may be corrupted, attempting to read raw content...")
        
        # Try to read raw content for debugging
        try:
            with open(WEBRTC_RESULTS_FILE, 'r') as f:
                raw_content = f.read()
                print(f"[UI] Raw file content (first 200 chars): {raw_content[:200]}")
        except Exception as raw_e:
            print(f"[UI] Couldn't even read raw content: {raw_e}")
        
        return None
        
    except Exception as e:
        print(f"[UI] FILE READ ERROR: {e}")
        return None

# GLOBAL THREAD-SAFE STORAGE FOR WEBRTC RESULTS - FIXED VERSION
import threading
webrtc_results_lock = threading.Lock()
webrtc_results_container = {
    "latest_result": None,
    "detection_history": [],
    "last_update": 0,
    "detection_count": 0,
    "last_sync_time": 0
}

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

# Check if running locally or on Streamlit Cloud
def is_local_environment():
    """Check if running locally or on Streamlit Cloud"""
    # Multiple checks for Streamlit Cloud deployment
    cloud_indicators = [
        os.getenv('STREAMLIT_SERVER_HEADLESS') == 'true',
        os.getenv('HOME') == '/home/appuser',
        'streamlit.app' in os.getenv('STREAMLIT_SERVER_ADDRESS', ''),
        'share.streamlit.io' in os.getenv('STREAMLIT_SERVER_ADDRESS', ''),
        os.path.exists('/.dockerenv'),
    ]
    
    # If any indicator suggests cloud deployment, return False (not local)
    is_deployed = any(cloud_indicators)
    
    # Additional check: try camera access
    if not is_deployed:
        try:
            test_cap = cv2.VideoCapture(0)
            has_camera = test_cap.isOpened()
            test_cap.release()
            if not has_camera:
                is_deployed = True
        except:
            is_deployed = True
    
    return not is_deployed

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

# FIXED WebRTC Video Processor with better synchronization
class QueueVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_detection_time = 0
        self.detection_interval = 5  # Reduced to 5 seconds for more frequent updates
        self.detector = None
        self.model_name = "YOLOv8m"
        self.confidence = 0.5
        self.queue_model_data = None
        self.current_hour = datetime.now().hour
        self.estimate_wait_time = True
        self.use_ml_prediction = True
        self.detection_count = 0
        self.last_successful_detection = 0
        
    def recv(self, frame):
        """Process video frame with people detection - FIXED VERSION WITH DEBUGGING"""
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        
        # Basic status overlay
        cv2.putText(img, f"WebRTC Queue Detection System", (20, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img, f"Detector: {'Ready' if self.detector else 'Loading...'}", (20, 60), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.detector else (0, 255, 255), 2)
        
        # Enhanced debugging overlay
        cv2.putText(img, f"Frame time: {current_time:.1f}", (20, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"Last detection: {self.last_detection_time:.1f}", (20, 110), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Detection timing logic
        should_detect = current_time - self.last_detection_time > self.detection_interval
        
        # Show timing info on overlay
        time_until_next = self.detection_interval - (current_time - self.last_detection_time)
        cv2.putText(img, f"Next detection in: {max(0, time_until_next):.1f}s", (20, 130), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(img, f"Should detect: {'YES' if should_detect else 'NO'}", (20, 150), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if should_detect else (255, 255, 0), 1)
        
        # Force first detection for testing (bypass timing for first detection)
        if self.detection_count == 0 and self.detector:
            should_detect = True
            print(f"[WebRTC] FORCING FIRST DETECTION for testing")
        
        if should_detect and self.detector:
            try:
                self.last_detection_time = current_time
                self.detection_count += 1
                
                print(f"[WebRTC] STARTING detection #{self.detection_count} at time {current_time}")
                
                # Add processing indicator
                cv2.putText(img, f"🔍 PROCESSING Detection #{self.detection_count}...", (20, 170), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # Convert to PIL for detector
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                
                print(f"[WebRTC] Image converted, calling detect_people with model: {self.model_name}, confidence: {self.confidence}")
                
                # Run detection
                try:
                    results = self.detector.detect_people(
                        pil_image, self.model_name, self.confidence
                    )
                    print(f"[WebRTC] Detection call completed, results: {results is not None}")
                    if results:
                        print(f"[WebRTC] People count: {results.get('people_count', 'unknown')}")
                except Exception as detect_error:
                    print(f"[WebRTC] DETECTION ERROR: {detect_error}")
                    results = None
                
                if results and results['annotated_image'] is not None:
                    # Use annotated image from detector
                    img = results['annotated_image']
                    self.last_successful_detection = current_time
                    
                    # Calculate wait time
                    wait_time = None
                    if self.estimate_wait_time and results['people_count'] > 0:
                        if self.use_ml_prediction and self.queue_model_data:
                            wait_time = predict_checkin_wait_time(
                                queue_size=results['people_count'],
                                hour_of_day=self.current_hour,
                                model_data=self.queue_model_data
                            )
                        else:
                            wait_time = (results['people_count'] * 3) / 3
                    
                    # Store results with better synchronization
                    detection_data = {
                        **results,
                        'wait_time': wait_time,
                        'timestamp': current_time,
                        'detection_count': self.detection_count,
                        'processing_time': time.time() - current_time,
                        'sync_id': f"webrtc_{self.detection_count}_{int(current_time)}"
                    }
                    
                    # Thread-safe storage with immediate sync flag
                    with webrtc_results_lock:
                        webrtc_results_container["latest_result"] = detection_data
                        webrtc_results_container["last_update"] = current_time
                        webrtc_results_container["detection_count"] = self.detection_count
                        
                        # Debug: Confirm the storage worked
                        stored_count = webrtc_results_container.get("detection_count", "ERROR")
                        stored_result_exists = webrtc_results_container.get("latest_result") is not None
                        print(f"[WebRTC] STORAGE VERIFICATION:")
                        print(f"[WebRTC] - Stored detection_count: {stored_count}")
                        print(f"[WebRTC] - Stored latest_result exists: {stored_result_exists}")
                        
                        # Detection history record
                        detection_record = {
                            'timestamp': current_time,
                            'people_count': results['people_count'],
                            'model': self.model_name,
                            'confidence': self.confidence,
                            'inference_time': results['inference_time'],
                            'wait_time': wait_time,
                            'sync_id': detection_data['sync_id']
                        }
                        
                        webrtc_results_container["detection_history"].append(detection_record)
                        
                        # Keep history manageable
                        if len(webrtc_results_container["detection_history"]) > 50:
                            webrtc_results_container["detection_history"].pop(0)
                    
                    # *** NEW: WRITE TO SHARED FILE FOR CROSS-THREAD ACCESS ***
                    file_write_success = write_webrtc_results(detection_data)
                    print(f"[WebRTC] Shared file write: {'SUCCESS' if file_write_success else 'FAILED'}")
                    
                    # Add success overlay to video
                    cv2.putText(img, f"✓ DETECTED: {results['people_count']} people", (20, img.shape[0] - 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(img, f"Wait Time: {wait_time:.0f} min" if wait_time else "Wait Time: Calculating...", 
                                (20, img.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    print(f"[WebRTC] Detection #{self.detection_count}: {results['people_count']} people, wait: {wait_time}")
                    print(f"[WebRTC] Stored in container with sync_id: {detection_data['sync_id']}")
                    print(f"[WebRTC] Container now has {len(webrtc_results_container['detection_history'])} history items")
                    
                else:
                    # Detection failed
                    cv2.putText(img, "⚠ Detection failed - trying again...", (20, 220), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        
            except Exception as e:
                error_msg = str(e)[:50]
                cv2.putText(img, f"✗ Error: {error_msg}", (20, 220), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                print(f"[WebRTC] Detection error: {e}")
        
        else:
            # Show status when not detecting - with debugging
            if self.detector:
                if not should_detect:
                    time_until_next = self.detection_interval - (current_time - self.last_detection_time)
                    cv2.putText(img, f"⏳ Waiting {time_until_next:.1f}s for next detection", (20, 190), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(img, "⚠️ Detection should happen but didn't", (20, 190), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                # Show last successful detection time
                if self.last_successful_detection > 0:
                    time_since_last = current_time - self.last_successful_detection
                    cv2.putText(img, f"Last success: {time_since_last:.0f}s ago", (20, 210), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            else:
                cv2.putText(img, "❌ NO DETECTOR - Waiting for initialization...", (20, 190), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                print(f"[WebRTC] Frame processed but NO DETECTOR available")
        
        # Always show current stats in corner
        cv2.putText(img, f"Total detections: {self.detection_count}", (20, img.shape[0] - 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# FIXED sync function with better error handling
def sync_webrtc_results():
    """FIXED: Sync thread-safe WebRTC results to session state with better error handling"""
    try:
        with webrtc_results_lock:
            # Get current data
            latest_result = webrtc_results_container.get("latest_result")
            detection_history = webrtc_results_container.get("detection_history", [])
            last_update = webrtc_results_container.get("last_update", 0)
            detection_count = webrtc_results_container.get("detection_count", 0)
            
            # Sync latest result if available
            if latest_result:
                st.session_state.webrtc_detection_results = latest_result.copy()
                st.session_state.webrtc_last_sync = time.time()
            
            # Sync detection history
            if detection_history:
                st.session_state.webrtc_detection_history = detection_history.copy()
            
            # Store sync metadata
            st.session_state.webrtc_sync_info = {
                'last_update': last_update,
                'detection_count': detection_count,
                'sync_time': time.time(),
                'has_results': latest_result is not None,
                'history_length': len(detection_history)
            }
            
            return {
                'success': True,
                'last_update': last_update,
                'detection_count': detection_count,
                'has_results': latest_result is not None
            }
            
    except Exception as e:
        print(f"[SYNC ERROR] {e}")
        return {'success': False, 'error': str(e)}

# SIMPLE results display that forces sync every time
def show_webrtc_results_simple():
    """SIMPLE: Force sync and show results immediately - FILE-BASED VERSION"""
    
    # Try to get data from shared file
    try:
        print(f"[UI] Reading from shared file: {WEBRTC_RESULTS_FILE}")
        
        # Read from shared file
        file_data = read_webrtc_results()
        
        if file_data:
            latest_result = file_data.get('latest_result')
            detection_count = file_data.get('detection_count', 0)
            last_update = file_data.get('last_update', 0)
            
            print(f"[UI] File data found:")
            print(f"[UI] - detection_count: {detection_count}")
            print(f"[UI] - latest_result exists: {latest_result is not None}")
            print(f"[UI] - last_update: {last_update}")
            
            # Debug: Show file-based data
            st.write("🔧 **DEBUG: File-Based Data**")
            st.write(f"Detection count: {detection_count}")
            st.write(f"Latest result exists: {latest_result is not None}")
            st.write(f"File exists: {os.path.exists(WEBRTC_RESULTS_FILE)}")
            
            # If we have results, show them immediately
            if latest_result and detection_count > 0:
                st.success(f"🎉 **DETECTION FOUND!** #{detection_count}")
                
                # Show the basic info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("👥 People Count", latest_result['people_count'])
                with col2:
                    st.metric("⚡ Processing Time", f"{latest_result['inference_time']:.2f}s")
                with col3:
                    st.metric("🎯 Model", latest_result['model_name'])
                
                # Show wait time if available
                if latest_result.get('wait_time'):
                    wait_time = latest_result['wait_time']
                    if wait_time < 15:
                        st.success(f"✅ **Short wait: {wait_time:.0f} minutes**")
                    elif wait_time < 45:
                        st.warning(f"⚠️ **Moderate wait: {wait_time:.0f} minutes**")
                    else:
                        st.error(f"🚨 **Long wait: {wait_time:.0f} minutes**")
                
                # Show timestamp
                result_time = datetime.fromtimestamp(latest_result['timestamp']).strftime('%H:%M:%S')
                st.info(f"🕐 **Detection time:** {result_time}")
                
                return True
                
            else:
                st.info(f"📊 **File status:** {detection_count} detections, but no latest result")
                return False
        
        else:
            # No file data, fall back to old container method
            st.warning("📁 **No file data found, checking container...**")
            
            # Original container-based approach as fallback
            with webrtc_results_lock:
                container_keys = list(webrtc_results_container.keys())
                latest_result = webrtc_results_container.get("latest_result")
                detection_count = webrtc_results_container.get("detection_count", 0)
                
                st.write("🔧 **DEBUG: Container Fallback Data**")
                st.write(f"Detection count: {detection_count}")
                st.write(f"Latest result exists: {latest_result is not None}")
                st.write(f"Container keys: {container_keys}")
            
            st.info(f"📊 **Container status:** {detection_count} detections, but no latest result")
            return False
            
    except Exception as e:
        st.error(f"❌ **Error getting results:** {e}")
        print(f"[UI ERROR] {e}")
        return False

# FIXED WebRTC live video section with improved synchronization
def webrtc_live_video_section(selected_model, confidence_threshold, queue_model_data, current_hour, estimate_wait_time, use_ml_prediction):
    """FIXED WebRTC live video section with better result synchronization"""
    
    st.subheader("🎥 Live Video Detection (WebRTC)")
    st.success("🌐 **Works on deployed apps!** Real-time video processing with automatic detection")
    
    # SIMPLIFIED STATUS CHECK - FIXED THREAD SYNC
    try:
        with webrtc_results_lock:
            container_data = dict(webrtc_results_container)  # Make a copy
        
        container_detection_count = container_data.get("detection_count", 0)
        container_has_results = container_data.get("latest_result") is not None
        last_update_time = container_data.get("last_update", 0)
        
        # FORCE SYNC if we have data but UI doesn't
        if container_has_results and 'webrtc_detection_results' not in st.session_state:
            st.session_state.webrtc_detection_results = container_data["latest_result"].copy()
            st.warning("🔄 **FORCED SYNC** - Found results and synced to UI!")
            
    except Exception as e:
        st.error(f"Status check error: {e}")
        container_detection_count = 0
        container_has_results = False
        last_update_time = 0
    
    session_has_results = 'webrtc_detection_results' in st.session_state
    
    # Real-time status bar
    status1, status2, status3, status4 = st.columns(4)
    with status1:
        st.metric("🔄 Container Detections", container_detection_count)
    with status2:
        st.metric("📦 Container Has Results", "✅" if container_has_results else "❌")
    with status3:
        st.metric("💾 Session Has Results", "✅" if session_has_results else "❌")
    with status4:
        if last_update_time > 0:
            last_update_str = datetime.fromtimestamp(last_update_time).strftime('%H:%M:%S')
            st.metric("🕐 Last Update", last_update_str)
        else:
            st.metric("🕐 Last Update", "Never")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        detection_interval = st.selectbox(
            "🕐 Detection Interval", 
            [3, 5, 8, 10, 15], 
            index=1,  # Default to 5 seconds
            help="How often to run detection (seconds)"
        )
    
    with col2:
        auto_refresh = st.checkbox("🔄 Auto-refresh Results", value=True)
        
    with col3:
        show_debug = st.checkbox("🔧 Show Debug Info", value=False)
    
    # WebRTC Streamer
    webrtc_ctx = webrtc_streamer(
        key="queue-detection-live-fixed",  # Changed key to reset
        video_processor_factory=QueueVideoProcessor,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={
            "video": {"width": {"min": 640, "ideal": 1280}, "height": {"min": 480, "ideal": 720}},
            "audio": False,
        },
        async_processing=True,
    )
    
    # Configure the processor when active - IMPROVED with debugging
    if webrtc_ctx.video_processor:
        processor = webrtc_ctx.video_processor
        processor.detection_interval = detection_interval
        processor.detector = st.session_state.detector
        processor.model_name = selected_model
        processor.confidence = confidence_threshold
        processor.queue_model_data = queue_model_data
        processor.current_hour = current_hour
        processor.estimate_wait_time = estimate_wait_time
        processor.use_ml_prediction = use_ml_prediction
        
        # Debug: Verify detector is properly set
        detector_status = "Ready" if processor.detector is not None else "Missing"
        st.sidebar.write(f"🔧 **Processor Detector:** {detector_status}")
        if processor.detector:
            st.sidebar.write(f"🎯 **Model:** {processor.model_name}")
            st.sidebar.write(f"🎚️ **Confidence:** {processor.confidence}")
        else:
            st.sidebar.error("❌ Detector not loaded in processor!")
    
    # Status and controls
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        if webrtc_ctx.state.playing:
            st.success("🟢 **Live streaming active**")
        else:
            st.info("⚪ **Click START to begin**")
    
    with status_col2:
        if st.button("🔄 Force Refresh Results"):
            # Multiple sync attempts for reliability
            for i in range(3):
                sync_result = sync_webrtc_results()
                if sync_result.get('has_results', False):
                    break
                time.sleep(0.1)  # Brief pause between attempts
            
            # Show what we found
            if sync_result['success']:
                with webrtc_results_lock:
                    container_count = webrtc_results_container.get("detection_count", 0)
                    has_latest = webrtc_results_container.get("latest_result") is not None
                st.success(f"✅ Sync success! Container detections: {container_count}, Has results: {has_latest}")
                if 'webrtc_detection_results' in st.session_state:
                    results = st.session_state.webrtc_detection_results
                    st.info(f"📊 Found {results.get('people_count', 0)} people at {datetime.fromtimestamp(results.get('timestamp', 0)).strftime('%H:%M:%S')}")
                st.rerun()
            else:
                st.error(f"❌ Sync failed: {sync_result.get('error', 'Unknown error')}")
    
    with status_col3:
        # Manual detection trigger for debugging
        if st.button("🔍 Force Detection Now"):
            if webrtc_ctx.video_processor and webrtc_ctx.video_processor.detector:
                # Reset detection time to force immediate detection
                webrtc_ctx.video_processor.last_detection_time = 0
                st.success("✅ Detection forced! Check video overlay.")
            else:
                st.error("❌ No active processor/detector")
        
        # Manual container check button
        if st.button("🔧 Check Container Now"):
            try:
                with webrtc_results_lock:
                    container_data = dict(webrtc_results_container)
                
                st.write("**Direct Container Check:**")
                st.json(container_data)
                
                detection_count = container_data.get("detection_count", 0)
                has_result = container_data.get("latest_result") is not None
                st.write(f"Count: {detection_count}, Has Result: {has_result}")
                
            except Exception as e:
                st.error(f"Container check failed: {e}")
        
        # Manual file check button
        if st.button("📁 Check File Now"):
            try:
                st.write(f"**File path:** {WEBRTC_RESULTS_FILE}")
                st.write(f"**File exists:** {os.path.exists(WEBRTC_RESULTS_FILE)}")
                
                if os.path.exists(WEBRTC_RESULTS_FILE):
                    file_size = os.path.getsize(WEBRTC_RESULTS_FILE)
                    st.write(f"**File size:** {file_size} bytes")
                    
                    file_data = read_webrtc_results()
                    if file_data:
                        st.write("**File contents:**")
                        st.json(file_data)
                        
                        detection_count = file_data.get("detection_count", 0)
                        people_count = file_data.get("people_count", 0)
                        has_result = file_data.get("latest_result") is not None
                        st.write(f"File Count: {detection_count}, People: {people_count}, Has Result: {has_result}")
                    else:
                        st.warning("File exists but couldn't read data")
                        
                        # Show raw file content for debugging
                        try:
                            with open(WEBRTC_RESULTS_FILE, 'r') as f:
                                raw_content = f.read()
                            st.text(f"Raw content (first 500 chars):\n{raw_content[:500]}")
                        except Exception as e:
                            st.error(f"Couldn't read raw content: {e}")
                else:
                    st.info("Results file doesn't exist yet - no detections have occurred")
                
            except Exception as e:
                st.error(f"File check failed: {e}")
        
        # Delete corrupted file button
        if st.button("🗑️ Delete Corrupted File"):
            try:
                if os.path.exists(WEBRTC_RESULTS_FILE):
                    os.remove(WEBRTC_RESULTS_FILE)
                    st.success("✅ Corrupted file deleted! Next detection will create a fresh file.")
                else:
                    st.info("No file to delete")
            except Exception as e:
                st.error(f"Couldn't delete file: {e}")
        
        if st.button("🗑️ Clear All Data"):
            # Clear both session state and global container
            with webrtc_results_lock:
                webrtc_results_container["latest_result"] = None
                webrtc_results_container["detection_history"] = []
                webrtc_results_container["last_update"] = 0
                webrtc_results_container["detection_count"] = 0
            
            # Clear file
            try:
                if os.path.exists(WEBRTC_RESULTS_FILE):
                    os.remove(WEBRTC_RESULTS_FILE)
                    st.info("🗑️ Results file deleted")
            except Exception as e:
                st.warning(f"Couldn't delete file: {e}")
            
            # Clear session state
            for key in ['webrtc_detection_results', 'webrtc_detection_history', 'webrtc_sync_info']:
                if key in st.session_state:
                    del st.session_state[key]
            
            st.success("✅ All data cleared!")
            st.rerun()
    
    # Auto-sync results when streaming is active - IMPROVED
    if webrtc_ctx.state.playing:
        # Force sync multiple times to ensure we get results
        sync_result = sync_webrtc_results()
        
        # Add immediate debug info to see what's happening
        st.sidebar.write("🔄 **Live Sync Status**")
        st.sidebar.write(f"Success: {sync_result['success']}")
        st.sidebar.write(f"Detection Count: {sync_result.get('detection_count', 0)}")
        st.sidebar.write(f"Has Results: {sync_result.get('has_results', False)}")
        
        # Show what's in session state
        has_webrtc_results = 'webrtc_detection_results' in st.session_state
        st.sidebar.write(f"Session State Has Results: {has_webrtc_results}")
        
        if has_webrtc_results:
            st.sidebar.success("✅ Results found!")
        else:
            st.sidebar.warning("⚠️ No results in session state")
            
        # Show sync status
        if show_debug:
            st.markdown("---")
            st.subheader("🔧 Debug Information")
            
            debug_col1, debug_col2 = st.columns(2)
            
            with debug_col1:
                st.write("**WebRTC Status:**")
                st.write(f"- State: {webrtc_ctx.state.playing}")
                st.write(f"- Processor: {'Active' if webrtc_ctx.video_processor else 'None'}")
                st.write(f"- Detector: {'Ready' if st.session_state.detector else 'None'}")
            
            with debug_col2:
                st.write("**Sync Status:**")
                st.write(f"- Success: {sync_result['success']}")
                st.write(f"- Detection Count: {sync_result.get('detection_count', 0)}")
                st.write(f"- Has Results: {sync_result.get('has_results', False)}")
                
                if 'webrtc_sync_info' in st.session_state:
                    sync_info = st.session_state.webrtc_sync_info
                    last_sync = datetime.fromtimestamp(sync_info['sync_time']).strftime('%H:%M:%S')
                    st.write(f"- Last Sync: {last_sync}")
    
    # SIMPLE RESULTS DISPLAY - Just show what we have!
    if webrtc_ctx.state.playing:
        st.markdown("---")
        st.subheader("🔍 Live Detection Results")
        
        # Use the simple results function
        has_results = show_webrtc_results_simple()
        
        if not has_results:
            st.info("⏳ **Waiting for detection...** Check the video overlay for status")
            st.markdown("**Expected flow:** Video → Detection → Results appear here")
    else:
        st.info("⚪ **WebRTC not active** - Click START above to begin live detection")
    
    # Simple auto-refresh for live updates
    if webrtc_ctx.state.playing and auto_refresh:
        time.sleep(1)  # Short delay
        st.rerun()

# Local live video section (placeholder - your existing code)
def local_live_video_section(selected_model, confidence_threshold, queue_model_data, current_hour, estimate_wait_time, use_ml_prediction):
    """Local live video detection section"""
    st.subheader("🎥 Live Local Camera")
    st.info("🖥️ **Local camera streaming** - For local development only")
    st.warning("⚠️ This feature requires local camera access and is not available on deployed apps")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">✈️ Airport Check-in Queue Detection System</h1>', unsafe_allow_html=True)
    st.markdown("**AI-powered queue analysis with live video streaming**")
    
    # Environment detection
    is_local = is_local_environment()
    
    # Show environment status
    env_col1, env_col2, env_col3 = st.columns(3)
    
    with env_col1:
        if is_local:
            st.success("🖥️ **Local Environment**")
        else:
            st.info("🌐 **Deployed Environment**")
    
    with env_col2:
        if WEBRTC_AVAILABLE:
            st.success("📡 **WebRTC Available**")
            st.markdown("Live streaming: ✅")
        else:
            st.warning("📡 **WebRTC Not Available**")
            st.markdown("Install: `pip install streamlit-webrtc`")
    
    with env_col3:
        device_info = "GPU" if torch.cuda.is_available() else "CPU"
        st.info(f"⚙️ **Processing: {device_info}**")
    
    st.markdown("---")
    
    # Initialize components
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = ModelManager()
    
    if 'detector' not in st.session_state:
        st.session_state.detector = QueueDetector(st.session_state.model_manager)
    
    # Debug: Check detector status
    detector_ready = st.session_state.detector is not None
    st.sidebar.write("🔧 **Main Detector Status**")
    st.sidebar.write(f"Detector exists: {'✅' if detector_ready else '❌'}")
    if detector_ready:
        st.sidebar.write(f"Model manager: {'✅' if st.session_state.detector.model_manager else '❌'}")
    
    print(f"[MAIN] Detector ready: {detector_ready}")
    
    # Load queue prediction model
    queue_model_data = load_queue_model()
    
    # Sidebar configuration
    st.sidebar.header("🔧 Configuration")
    
    available_models = st.session_state.model_manager.get_available_models()
    
    if not available_models:
        st.error("❌ No detection models available. Please install required packages:")
        st.code("pip install ultralytics transformers torch")
        return
    
    selected_model = st.sidebar.selectbox("🤖 Select Detection Model", available_models)
    confidence_threshold = st.sidebar.slider("🎯 Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
    
    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings"):
        show_details = st.checkbox("Show detection details", value=True)
        estimate_wait_time = st.checkbox("Estimate wait time", value=True)
        
        st.markdown("**🤖 ML Queue Prediction**")
        use_ml_prediction = st.checkbox("Use ML prediction model", value=True)
        
        if use_ml_prediction and queue_model_data:
            st.success("✅ ML model loaded")
            current_hour = st.slider("Current hour", 0, 23, datetime.now().hour)
        else:
            current_hour = datetime.now().hour
            if not queue_model_data:
                st.info("ℹ️ ML model not available - using simple formula")
    
    # Main content layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📹 Camera Detection")
        
        # Camera mode selection
        camera_mode = st.radio(
            "📷 Select Camera Mode",
            ["🎥 Live Video (WebRTC)", "📸 Photo Capture", "🖥️ Local Camera (Local Only)"],
            help="Choose your preferred detection method"
        )
        
        if camera_mode == "🎥 Live Video (WebRTC)":
            if WEBRTC_AVAILABLE:
                webrtc_live_video_section(
                    selected_model, confidence_threshold, queue_model_data,
                    current_hour, estimate_wait_time, use_ml_prediction
                )
            else:
                st.error("❌ WebRTC not available. Install with: `pip install streamlit-webrtc`")
                st.info("💡 Using photo capture mode instead:")
                browser_camera_section(
                    selected_model, confidence_threshold, queue_model_data,
                    current_hour, estimate_wait_time, use_ml_prediction
                )
        
        elif camera_mode == "📸 Photo Capture":
            browser_camera_section(
                selected_model, confidence_threshold, queue_model_data,
                current_hour, estimate_wait_time, use_ml_prediction
            )
        
        elif camera_mode == "🖥️ Local Camera (Local Only)":
            if is_local:
                local_live_video_section(
                    selected_model, confidence_threshold, queue_model_data,
                    current_hour, estimate_wait_time, use_ml_prediction
                )
            else:
                st.warning("⚠️ Local camera not available on deployed apps")
                st.info("💡 Use WebRTC or Photo Capture mode instead")
        
        # Image upload section
        st.markdown("---")
        st.subheader("📁 Upload Queue Image")
        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png', 'bmp'])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            if st.button("🔍 Detect People in Queue", type="primary"):
                with st.spinner(f"Running {selected_model} detection..."):
                    results = st.session_state.detector.detect_people(
                        image, selected_model, confidence_threshold
                    )
                    
                    if results:
                        st.session_state.detection_results = results
                        st.success(f"✅ Detection completed! Found {results['people_count']} people")
                    else:
                        st.error("❌ Detection failed")
    
    with col2:
        st.header("📊 Detection Results")
        
        # Show upload results or live results
        results_to_show = None
        result_source = None
        
        if 'detection_results' in st.session_state:
            results_to_show = st.session_state.detection_results
            result_source = "🖼️ Uploaded Image Results"
        elif 'webrtc_detection_results' in st.session_state:
            results_to_show = st.session_state.webrtc_detection_results
            result_source = "🎥 Live Video Results"
        elif 'browser_detection_results' in st.session_state:
            results_to_show = st.session_state.browser_detection_results
            result_source = "📸 Camera Results"
        
        if results_to_show:
            st.subheader(result_source)
            
            # Display annotated image
            annotated_image_rgb = cv2.cvtColor(results_to_show['annotated_image'], cv2.COLOR_BGR2RGB)
            st.image(annotated_image_rgb, caption=f"Detection Results - {results_to_show['model_name']}", use_column_width=True)
            
            # Metrics
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric("👥 People Count", results_to_show['people_count'])
            
            with col_b:
                st.metric("⚡ Inference Time", f"{results_to_show['inference_time']:.2f}s")
            
            with col_c:
                st.metric("🚀 FPS", f"{1/results_to_show['inference_time']:.1f}")
            
            # Wait time estimation
            if estimate_wait_time and results_to_show['people_count'] > 0:
                wait_time = results_to_show.get('wait_time')
                
                if wait_time is None:
                    # Calculate wait time if not already calculated
                    if use_ml_prediction and queue_model_data:
                        wait_time = predict_checkin_wait_time(
                            queue_size=results_to_show['people_count'],
                            hour_of_day=current_hour,
                            model_data=queue_model_data
                        )
                    else:
                        wait_time = (results_to_show['people_count'] * 3) / 3
                
                if wait_time:
                    display_wait_time = float(wait_time)
                    
                    if display_wait_time < 15:
                        st.success(f"✅ **Short wait: {display_wait_time:.0f} minutes**")
                    elif display_wait_time < 45:
                        st.warning(f"⚠️ **Moderate wait: {display_wait_time:.0f} minutes**")
                    else:
                        st.error(f"🚨 **Long wait: {display_wait_time:.0f} minutes**")
                    
                    st.info(f"📊 Queue: {results_to_show['people_count']} people | 🕐 Time: {current_hour}:00")
            
            # Detection details
            if show_details and results_to_show.get('detections'):
                st.subheader("🔍 Detection Details")
                
                detection_data = []
                for i, detection in enumerate(results_to_show['detections']):
                    detection_data.append({
                        'Person #': i + 1,
                        'Confidence': f"{detection['confidence']:.3f}",
                        'Bounding Box': f"({detection['bbox'][0]}, {detection['bbox'][1]}) to ({detection['bbox'][2]}, {detection['bbox'][3]})"
                    })
                
                st.dataframe(detection_data, use_container_width=True)
                
                avg_confidence = np.mean([d['confidence'] for d in results_to_show['detections']])
                st.info(f"📊 Average confidence: {avg_confidence:.3f}")
        
        else:
            st.info("👆 Use camera or upload an image to see detection results")
    
    # Footer with system status
    st.markdown("---")
    st.subheader("🔧 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.markdown("**📦 Available Models**")
        for model in available_models:
            st.markdown(f"✅ {model}")
    
    with status_col2:
        st.markdown("**⚙️ Environment**")
        st.markdown(f"🖥️ Local: {'✅' if is_local else '❌'}")
        st.markdown(f"🌐 Deployed: {'✅' if not is_local else '❌'}")
        st.markdown(f"📡 WebRTC: {'✅' if WEBRTC_AVAILABLE else '❌'}")
    
    with status_col3:
        st.markdown("**🤖 ML Queue Prediction**")
        if queue_model_data:
            st.markdown("✅ Model loaded")
            if 'mae' in queue_model_data:
                st.markdown(f"📈 Accuracy: {queue_model_data['mae']:.1f} min MAE")
        else:
            st.markdown("❌ Model not available")

if __name__ == "__main__":
    main()


# Browser camera section (same as before)
def browser_camera_section(selected_model, confidence_threshold, queue_model_data, current_hour, estimate_wait_time, use_ml_prediction):
    """Browser-based camera for fallback"""
    
    st.subheader("📸 Browser Camera Capture")
    st.info("📱 Manual photo capture mode - Click 'Take Photo' for each detection")
    
    # Initialize session state
    if 'browser_detection_history' not in st.session_state:
        st.session_state.browser_detection_history = []
    
    # Camera input
    camera_photo = st.camera_input("📷 Take a photo of the queue")
    
    if camera_photo is not None:
        st.image(camera_photo, caption="Captured Photo", use_column_width=True)
        
        if st.button("🔍 Detect People in Photo", type="primary"):
            with st.spinner(f"Running {selected_model} detection..."):
                image = Image.open(camera_photo)
                
                results = st.session_state.detector.detect_people(
                    image, selected_model, confidence_threshold
                )
                
                if results:
                    st.session_state.browser_detection_results = results
                    
                    # Calculate wait time
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
                    
                    # Store results
                    detection_record = {
                        'timestamp': time.time(),
                        'people_count': results['people_count'],
                        'model': selected_model,
                        'confidence': confidence_threshold,
                        'inference_time': results['inference_time'],
                        'wait_time': wait_time
                    }
                    
                    st.session_state.browser_detection_history.append(detection_record)
                    
                    if len(st.session_state.browser_detection_history) > 20:
                        st.session_state.browser_detection_history.pop(0)
                    
                    st.success(f"✅ Detection completed! Found {results['people_count']} people")
                    st.rerun()
                else:
                    st.error("❌ Detection failed")