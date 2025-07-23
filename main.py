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

class LiveVideoDetector:
    def __init__(self, model_manager, detector):
        self.model_manager = model_manager
        self.detector = detector
        self.cap = None
        self.is_running = False
        self.frame_queue = queue.Queue(maxsize=2)  # Small buffer
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
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer lag
            
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
                # Only keep the latest frame to avoid lag
                try:
                    if not self.frame_queue.empty():
                        self.frame_queue.get_nowait()  # Remove old frame
                    self.frame_queue.put(frame, timeout=0.1)
                except (queue.Full, queue.Empty):
                    pass
            else:
                break
            
            time.sleep(0.033)  # ~30 FPS
    
    def get_latest_frame(self):
        """Get the most recent frame"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    def is_camera_active(self):
        """Check if camera is active"""
        return self.is_running and self.cap and self.cap.isOpened()

def enhanced_live_video_section(selected_model, confidence_threshold, queue_model_data, current_hour, estimate_wait_time, use_ml_prediction):
    """Enhanced live video detection section"""
    
    st.subheader("🎥 Live Webcam Detection")
    
    # Initialize video detector in session state
    if 'video_detector' not in st.session_state:
        st.session_state.video_detector = LiveVideoDetector(
            st.session_state.model_manager, 
            st.session_state.detector
        )
        st.session_state.camera_active = False
        st.session_state.last_detection_time = 0
        st.session_state.detection_interval = 10  # seconds
        st.session_state.current_detection_results = None
        st.session_state.live_detection_history = []
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not st.session_state.camera_active:
            if st.button("🎬 Start Camera", type="primary"):
                if st.session_state.video_detector.start_camera():
                    st.session_state.camera_active = True
                    st.session_state.last_detection_time = 0  # Trigger immediate detection
                    st.success("✅ Camera started!")
                    st.rerun()
        else:
            if st.button("⏹️ Stop Camera"):
                st.session_state.video_detector.stop_camera()
                st.session_state.camera_active = False
                st.success("✅ Camera stopped!")
                st.rerun()
    
    with col2:
        # Detection interval control
        new_interval = st.selectbox(
            "Detection Interval", 
            [5, 8, 10, 15, 20], 
            index=2,  # Default to 10 seconds
            key="detection_interval_select"
        )
        if new_interval != st.session_state.detection_interval:
            st.session_state.detection_interval = new_interval
    
    with col3:
        # Camera status
        if st.session_state.camera_active:
            if st.session_state.video_detector.is_camera_active():
                st.success("🟢 Camera Active")
            else:
                st.error("🔴 Camera Error")
        else:
            st.info("⚪ Camera Off")
    
    # Main video display and detection
    if st.session_state.camera_active:
        
        # Create placeholders for dynamic content
        video_col, info_col = st.columns([2, 1])
        
        with video_col:
            video_placeholder = st.empty()
        
        with info_col:
            status_placeholder = st.empty()
            detection_placeholder = st.empty()
        
        # Get current frame
        current_frame = st.session_state.video_detector.get_latest_frame()
        current_time = time.time()
        
        if current_frame is not None:
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
            
            # Always show current frame
            with video_placeholder.container():
                st.image(
                    frame_rgb, 
                    caption=f"Live Video Feed (Detecting every {st.session_state.detection_interval}s)", 
                    use_column_width=True
                )
            
            # Check if it's time for detection
            time_since_last = current_time - st.session_state.last_detection_time
            
            if time_since_last >= st.session_state.detection_interval:
                # Time for detection!
                st.session_state.last_detection_time = current_time
                
                with status_placeholder.container():
                    with st.spinner("🔍 Detecting people..."):
                        # Run detection on current frame
                        pil_image = Image.fromarray(frame_rgb)
                        
                        detection_results = st.session_state.detector.detect_people(
                            pil_image, selected_model, confidence_threshold
                        )
                        
                        if detection_results:
                            st.session_state.current_detection_results = detection_results
                            
                            # Add to history
                            detection_record = {
                                'timestamp': current_time,
                                'people_count': detection_results['people_count'],
                                'model': selected_model,
                                'confidence': confidence_threshold,
                                'inference_time': detection_results['inference_time']
                            }
                            
                            st.session_state.live_detection_history.append(detection_record)
                            
                            # Keep only last 20 detections
                            if len(st.session_state.live_detection_history) > 20:
                                st.session_state.live_detection_history.pop(0)
                            
                            # Show results
                            with detection_placeholder.container():
                                st.success(f"✅ **{detection_results['people_count']} people detected**")
                                st.info(f"⚡ {detection_results['inference_time']:.2f}s | 🎯 {selected_model}")
                                
                                # Show wait time prediction for live detection
                                if estimate_wait_time and detection_results['people_count'] > 0:
                                    # ML prediction
                                    ml_wait_time = None
                                    if use_ml_prediction and queue_model_data:
                                        # Use real-time hour if in automatic mode
                                        prediction_hour = datetime.now().hour if st.session_state.get('time_mode', '🤖 Automatic (Real-time)') == '🤖 Automatic (Real-time)' else current_hour
                                        
                                        ml_wait_time = predict_checkin_wait_time(
                                            queue_size=detection_results['people_count'],
                                            hour_of_day=prediction_hour,
                                            model_data=queue_model_data
                                        )
                                    
                                    # Simple fallback calculation
                                    simple_wait_time = (detection_results['people_count'] * 3) / 3  # minutes
                                    
                                    # Determine which wait time to display
                                    display_wait_time = float(ml_wait_time) if ml_wait_time is not None else simple_wait_time
                                    
                                    # Show wait time with color coding
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
                # Show countdown to next detection
                time_remaining = st.session_state.detection_interval - time_since_last
                
                with status_placeholder.container():
                    st.info(f"⏱️ Next detection in {time_remaining:.1f}s")
                    
                    # Progress bar
                    progress = 1 - (time_remaining / st.session_state.detection_interval)
                    st.progress(progress)
                    
                    # Show last detection if available
                    if st.session_state.current_detection_results:
                        people_count = st.session_state.current_detection_results['people_count']
                        st.markdown(f"📊 Last detection: **{people_count} people**")
        
        else:
            with video_placeholder.container():
                st.warning("⚠️ No video frame available")
        
        # Auto-refresh for continuous updates
        time.sleep(0.5)  # Small delay to prevent too frequent updates
        st.rerun()
    
    # Show detection results if available
    if st.session_state.current_detection_results:
        st.markdown("---")
        st.subheader("🔍 Latest Live Detection Results")
        
        results = st.session_state.current_detection_results
        
        # Display annotated image
        annotated_image_rgb = cv2.cvtColor(results['annotated_image'], cv2.COLOR_BGR2RGB)
        st.image(
            annotated_image_rgb, 
            caption=f"Latest Detection - {results['model_name']}", 
            use_column_width=True
        )
        
        # Metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            st.metric("👥 People Count", results['people_count'])
        
        with metric_col2:
            st.metric("⚡ Inference Time", f"{results['inference_time']:.2f}s")
        
        with metric_col3:
            detection_time = datetime.fromtimestamp(st.session_state.last_detection_time)
            st.metric("🕐 Detection Time", detection_time.strftime("%H:%M:%S"))
    
    # Detection history chart
    if len(st.session_state.live_detection_history) > 1:
        st.markdown("---")
        st.subheader("📈 Live Detection History")
        
        # Create simple line chart
        try:
            import plotly.graph_objects as go
            
            times = [
                datetime.fromtimestamp(d['timestamp']).strftime('%H:%M:%S') 
                for d in st.session_state.live_detection_history
            ]
            counts = [d['people_count'] for d in st.session_state.live_detection_history]
            
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
                title=f"Live People Count Over Time (Last {len(st.session_state.live_detection_history)} detections)",
                xaxis_title="Time",
                yaxis_title="Number of People",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            # Fallback if plotly not available
            st.info("📊 Install plotly for detection history charts: `pip install plotly`")
        
        # Recent detections table
        if st.expander("📊 Recent Live Detection Details", expanded=False):
            recent_detections = []
            for detection in reversed(st.session_state.live_detection_history[-10:]):  # Last 10
                recent_detections.append({
                    'Time': datetime.fromtimestamp(detection['timestamp']).strftime('%H:%M:%S'),
                    'People': detection['people_count'],
                    'Model': detection['model'],
                    'Confidence': detection['confidence'],
                    'Inference (s)': f"{detection['inference_time']:.2f}"
                })
            
            st.dataframe(recent_detections, use_container_width=True)
            
            # Clear history button
            if st.button("🗑️ Clear Live History", key="clear_live_history"):
                st.session_state.live_detection_history = []
                st.success("✅ Live detection history cleared!")
                st.rerun()

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
        
        # ML Queue prediction settings
        st.markdown("**🤖 Airport Check-in Queue Prediction**")
        use_ml_prediction = st.checkbox("Use ML prediction model", value=True, 
                                       help="Uses queue size and current time to predict wait time")
        
        if use_ml_prediction and queue_model_data:
            st.success("✅ ML model loaded successfully")
            
            # Time detection mode
            time_mode = st.radio(
                "⏰ Time Detection Mode",
                ["🤖 Automatic (Real-time)", "⚙️ Manual Override"],
                index=0,
                help="Choose automatic real-time detection or manual time setting"
            )
            
            if time_mode == "🤖 Automatic (Real-time)":
                current_hour = datetime.now().hour
                st.info(f"🕐 **Current time: {current_hour}:00** (Auto-detected and updating)")
            else:
                current_hour = st.slider("Current hour (24h format)", 
                                        min_value=0, max_value=23, 
                                        value=datetime.now().hour,
                                        help="Time affects service speed (peak vs off-peak hours)")
                st.info(f"🕐 **Using manual time: {current_hour}:00**")
            
            # Show time-based info
            if 7 <= current_hour <= 11:
                st.warning("⏰ **Peak Morning Hours (7-11 AM)** - Expect slower service")
            elif 15 <= current_hour <= 19:
                st.warning("⏰ **Peak Evening Hours (3-7 PM)** - Busy period")
            elif current_hour <= 7 or current_hour >= 20:
                st.info("⏰ **Off-Peak Hours** - Faster service expected")
            else:
                st.info("⏰ **Normal Hours** - Standard service speed")
                
            # Store time mode in session state for live detection
            st.session_state.time_mode = time_mode
                
        elif use_ml_prediction:
            st.error("❌ ML model not available")
            current_hour = datetime.now().hour
            st.session_state.time_mode = "🤖 Automatic (Real-time)"
        else:
            current_hour = datetime.now().hour
            st.session_state.time_mode = "🤖 Automatic (Real-time)"
            st.info("📊 Using simple calculation only")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📹 Video & Image Detection")
        
        # Enhanced live video section
        enhanced_live_video_section(
            selected_model, confidence_threshold, queue_model_data, 
            current_hour, estimate_wait_time, use_ml_prediction
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
            
            # Wait time estimation
            if estimate_wait_time and results['people_count'] > 0:
                # ML prediction
                ml_wait_time = None
                if use_ml_prediction and queue_model_data:
                    # Use real-time hour if in automatic mode
                    prediction_hour = datetime.now().hour if st.session_state.get('time_mode', '🤖 Automatic (Real-time)') == '🤖 Automatic (Real-time)' else current_hour
                    
                    ml_wait_time = predict_checkin_wait_time(
                        queue_size=results['people_count'],
                        hour_of_day=prediction_hour,
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
                    # Show real-time or manual time based on mode
                    display_hour = datetime.now().hour if st.session_state.get('time_mode', '🤖 Automatic (Real-time)') == '🤖 Automatic (Real-time)' else current_hour
                    time_mode_text = "Auto" if st.session_state.get('time_mode', '🤖 Automatic (Real-time)') == '🤖 Automatic (Real-time)' else "Manual"
                    st.markdown(f"**🕐 Current time:** {display_hour}:00 ({time_mode_text})")
                else:
                    # Fallback to simple calculation
                    st.markdown(f"**📊 Estimated wait time:** {simple_wait_time:.1f} minutes")
                    st.markdown("**⚠️ Note:** ML model not available, using simple calculation")
                
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
                    # Use real-time hour if in automatic mode
                    prediction_hour = datetime.now().hour if st.session_state.get('time_mode', '🤖 Automatic (Real-time)') == '🤖 Automatic (Real-time)' else current_hour
                    
                    ml_wait_time = predict_checkin_wait_time(
                        queue_size=results['people_count'],
                        hour_of_day=prediction_hour,
                        model_data=queue_model_data
                    )
                    
                    if ml_wait_time is not None:
                        download_data['queue_prediction'] = {
                            'ml_prediction_minutes': float(ml_wait_time),
                            'queue_size': results['people_count'],
                            'hour_of_day': prediction_hour,
                            'model_type': 'airport_checkin_simplified',
                            'features_used': ['queue_size', 'hour_of_day'],
                            'time_mode': st.session_state.get('time_mode', '🤖 Automatic (Real-time)')
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
            st.info("👆 Start live video or upload an image to see detection results")
    
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
    
    # Add live video streaming status
    status_col4, status_col5, status_col6 = st.columns(3)
    
    with status_col4:
        st.markdown("**🎥 Live Video Streaming**")
        if 'camera_active' in st.session_state and st.session_state.camera_active:
            if 'video_detector' in st.session_state and st.session_state.video_detector.is_camera_active():
                st.markdown("🟢 Camera Active")
                st.markdown(f"⏱️ Detection interval: {st.session_state.detection_interval}s")
                if 'live_detection_history' in st.session_state:
                    st.markdown(f"📊 Live detections: {len(st.session_state.live_detection_history)}")
            else:
                st.markdown("🔴 Camera Error")
        else:
            st.markdown("⚪ Camera Inactive")
    
    with status_col5:
        st.markdown("**📈 Live Detection History**")
        if 'live_detection_history' in st.session_state and st.session_state.live_detection_history:
            latest_count = st.session_state.live_detection_history[-1]['people_count']
            st.markdown(f"👥 Latest: {latest_count} people")
            st.markdown(f"📊 Total: {len(st.session_state.live_detection_history)} live detections")
        else:
            st.markdown("📊 No live detections yet")

if __name__ == "__main__":
    main()