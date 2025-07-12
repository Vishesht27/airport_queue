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


# Load queue prediction model
@st.cache_resource
def load_queue_model():
    """Load the trained queue wait prediction model"""
    try:
        with open('queue_wait_predictor.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load queue prediction model: {e}")
        return None

def predict_queue_wait(queue_size, model_data, num_counters=3, service_time_per_person=50, 
                      is_rush_hour=0, is_weekend=0, efficiency_score=2.0, hour=12, queue_type=1):
    """Predict queue wait time using the trained model"""
    if model_data is None:
        return None
    
    # Calculate derived features
    queue_density = queue_size / num_counters if num_counters > 0 else queue_size
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    
    # Create one-hot encoding for queue type
    queue_type_0 = 1 if queue_type == 0 else 0
    queue_type_1 = 1 if queue_type == 1 else 0
    queue_type_2 = 1 if queue_type == 2 else 0
    queue_type_3 = 1 if queue_type == 3 else 0
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'queue_size': [queue_size],
        'num_counters': [num_counters],
        'service_time_per_person': [service_time_per_person],
        'is_rush_hour': [is_rush_hour],
        'is_weekend': [is_weekend],
        'queue_density': [queue_density],
        'efficiency_score': [efficiency_score],
        'hour_sin': [hour_sin],
        'hour_cos': [hour_cos],
        'queue_type_0': [queue_type_0],
        'queue_type_1': [queue_type_1],
        'queue_type_2': [queue_type_2],
        'queue_type_3': [queue_type_3]
    })
    
    try:
        prediction = model_data['model'].predict(input_data)[0]
        return max(0, prediction)  # Ensure non-negative wait time
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
    import detectron2
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data import MetadataCatalog
    from detectron2.model_zoo import model_zoo
    DETECTRON2_AVAILABLE = True
except ImportError:
    DETECTRON2_AVAILABLE = False

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
            'Detectron2': {
                'file': 'mask_rcnn_R_50_FPN_3x',
                'type': 'detectron2',
                'description': 'Facebook Research - Professional grade with segmentation',
                'requirements': ['detectron2']
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
        
        if DETECTRON2_AVAILABLE:
            available.append('Detectron2')
        
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
            elif model_config['type'] == 'detectron2':
                model = self._load_detectron2_model()
            elif model_config['type'] == 'detr':
                model = self._load_detr_model()
            else:
                raise ValueError(f"Unknown model type: {model_config['type']}")
            
            self.models[model_name] = model
            return model
            
        except Exception as e:
            st.error(f"Failed to load {model_name}: {str(e)}")
            return None
    
    def _load_detectron2_model(self):
        """Load Detectron2 model"""
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        predictor = DefaultPredictor(cfg)
        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
        
        return {'predictor': predictor, 'metadata': metadata, 'cfg': cfg}
    
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
        elif model_config['type'] == 'detectron2':
            return self._detect_detectron2(image_cv, image, model, model_name, confidence_threshold)
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
    
    def _detect_detectron2(self, image_cv, image_pil, model, model_name, confidence_threshold):
        """Detect using Detectron2 model"""
        start_time = time.time()
        
        # Convert BGR to RGB for Detectron2
        rgb_image = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        
        # Run inference
        outputs = model['predictor'](rgb_image)
        inference_time = time.time() - start_time
        
        # Process results
        instances = outputs["instances"].to("cpu")
        people_detections = []
        
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.numpy()
            classes = instances.pred_classes.numpy()
            scores = instances.scores.numpy()
            
            for i in range(len(boxes)):
                if classes[i] == 0 and scores[i] >= confidence_threshold:  # person class
                    x1, y1, x2, y2 = boxes[i]
                    people_detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(scores[i])
                    })
        
        # Create visualization
        v = Visualizer(rgb_image, model['metadata'], scale=1.0)
        vis_output = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        annotated_image = cv2.cvtColor(vis_output.get_image(), cv2.COLOR_RGB2BGR)
        
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

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">✈️ Airport Queue Detection System</h1>', unsafe_allow_html=True)
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
        pip install 'git+https://github.com/facebookresearch/detectron2.git'  # For Detectron2
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
    with st.sidebar.expander("⚙️ Advanced Options"):
        show_details = st.checkbox("Show detection details", value=True)
        estimate_wait_time = st.checkbox("Estimate wait time", value=True)
        service_time = st.number_input("Service time per person (seconds)", min_value=10, max_value=300, value=45)
        
        # Queue prediction settings
        st.markdown("**🤖 ML Queue Prediction Settings**")
        use_ml_prediction = st.checkbox("Use ML prediction model", value=True, help="Use trained ML model for more accurate wait time prediction")
        
        if use_ml_prediction and queue_model_data:
            num_counters = st.number_input("Number of service counters", min_value=1, max_value=10, value=3)
            is_rush_hour = st.checkbox("Rush hour", value=False)
            is_weekend = st.checkbox("Weekend", value=False)
            efficiency_score = st.slider("Staff efficiency score", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
            current_hour = st.slider("Current hour (24h)", min_value=0, max_value=23, value=12)
            
            queue_types = ["Security Check", "Check-in", "Immigration", "Customs"]
            queue_type = st.selectbox("Queue type", options=queue_types, index=1)
            queue_type_index = queue_types.index(queue_type)
        else:
            num_counters = 3
            is_rush_hour = False
            is_weekend = False
            efficiency_score = 2.0
            current_hour = 12
            queue_type_index = 1
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image of people in a queue"
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
                # Simple calculation
                simple_wait_time = (results['people_count'] * service_time) / 60
                
                # ML prediction
                ml_wait_time = None
                if use_ml_prediction and queue_model_data:
                    ml_wait_time = predict_queue_wait(
                        queue_size=results['people_count'],
                        model_data=queue_model_data,
                        num_counters=num_counters,
                        service_time_per_person=service_time,
                        is_rush_hour=1 if is_rush_hour else 0,
                        is_weekend=1 if is_weekend else 0,
                        efficiency_score=efficiency_score,
                        hour=current_hour,
                        queue_type=queue_type_index
                    )
                
                # Determine which wait time to use for status
                display_wait_time = float(ml_wait_time) if ml_wait_time is not None else simple_wait_time
                
                if display_wait_time < 5:
                    card_class = "success-card"
                    status = "✅ Short wait"
                elif display_wait_time < 15:
                    card_class = "warning-card"
                    status = "⚠️ Moderate wait"
                else:
                    card_class = "error-card"
                    status = "🚨 Long wait"
                
                st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                st.markdown(f"**{status}**")
                
                if ml_wait_time is not None:
                    # Convert numpy float to Python float for display
                    ml_wait_time_float = float(ml_wait_time)
                    st.markdown(f"**🤖 ML Predicted wait time:** {ml_wait_time_float:.1f} minutes")
                    st.markdown(f"**📊 Simple estimate:** {simple_wait_time:.1f} minutes")
                    st.markdown(f"**📈 Model confidence:** High (trained on historical data)")
                else:
                    st.markdown(f"**📊 Estimated wait time:** {simple_wait_time:.1f} minutes")
                    st.markdown(f"**Based on:** {service_time}s per person")
                
                # Show additional ML insights
                if ml_wait_time is not None:
                    ml_wait_time_float = float(ml_wait_time)
                    st.markdown(f"**⚙️ Settings:** {num_counters} counters, {queue_type}, {'Rush hour' if is_rush_hour else 'Normal hours'}")
                    if ml_wait_time_float > simple_wait_time:
                        st.markdown("⚠️ **Note:** ML model predicts longer wait due to current conditions")
                    elif ml_wait_time_float < simple_wait_time:
                        st.markdown("✅ **Note:** ML model predicts shorter wait due to efficient operations")
                
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
                'detections': results['detections']
            }
            
            # Add ML prediction data if available
            if estimate_wait_time and results['people_count'] > 0:
                simple_wait_time = (results['people_count'] * service_time) / 60
                download_data['wait_time_estimation'] = {
                    'simple_estimate_minutes': simple_wait_time,
                    'service_time_per_person_seconds': service_time,
                    'ml_prediction_available': use_ml_prediction and queue_model_data is not None
                }
                
                if use_ml_prediction and queue_model_data:
                    ml_wait_time = predict_queue_wait(
                        queue_size=results['people_count'],
                        model_data=queue_model_data,
                        num_counters=num_counters,
                        service_time_per_person=service_time,
                        is_rush_hour=1 if is_rush_hour else 0,
                        is_weekend=1 if is_weekend else 0,
                        efficiency_score=efficiency_score,
                        hour=current_hour,
                        queue_type=queue_type_index
                    )
                    
                    if ml_wait_time is not None:
                        # Convert numpy types to Python types for JSON serialization
                        download_data['wait_time_estimation']['ml_prediction_minutes'] = float(ml_wait_time)
                        download_data['wait_time_estimation']['ml_settings'] = {
                            'num_counters': int(num_counters),
                            'is_rush_hour': bool(is_rush_hour),
                            'is_weekend': bool(is_weekend),
                            'efficiency_score': float(efficiency_score),
                            'current_hour': int(current_hour),
                            'queue_type': str(queue_type),
                            'queue_type_index': int(queue_type_index)
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
    
    # Model status
    st.subheader("🔧 System Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.markdown("**📦 Available Models**")
        for model in available_models:
            st.markdown(f"✅ {model}")
    
    with status_col2:
        st.markdown("**⚙️ Dependencies**")
        st.markdown(f"✅ Ultralytics: {ULTRALYTICS_AVAILABLE}")
        st.markdown(f"{'✅' if DETECTRON2_AVAILABLE else '❌'} Detectron2: {DETECTRON2_AVAILABLE}")
        st.markdown(f"{'✅' if TRANSFORMERS_AVAILABLE else '❌'} Transformers: {TRANSFORMERS_AVAILABLE}")
    
    with status_col3:
        st.markdown("**🎮 Hardware**")
        st.markdown(f"🖥️ Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        if torch.cuda.is_available():
            st.markdown(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    
    # Add ML model status
    st.markdown("---")
    status_col4, status_col5 = st.columns(2)
    
    with status_col4:
        st.markdown("**🤖 ML Queue Prediction**")
        if queue_model_data:
            st.markdown("✅ Queue prediction model loaded")
            st.markdown(f"📊 Model type: {type(queue_model_data.get('model', 'Unknown')).__name__}")
        else:
            st.markdown("❌ Queue prediction model not available")
            st.markdown("💡 Place `queue_wait_predictor.pkl` in the project directory")
    
    with status_col5:
        st.markdown("**📈 Prediction Features**")
        if queue_model_data:
            st.markdown("✅ Queue size, counters, time factors")
            st.markdown("✅ Rush hour, weekend, efficiency")
            st.markdown("✅ Queue type classification")
        else:
            st.markdown("❌ No ML features available")

if __name__ == "__main__":
    main()