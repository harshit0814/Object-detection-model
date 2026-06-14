import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import os
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="YOLOv8 Object Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 18px;
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.markdown("## ⚙️ Configuration")

# Model selection
model_name = st.sidebar.selectbox(
    "Select YOLOv8 Model",
    ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
    help="n=nano, s=small, m=medium, l=large, x=extra-large"
)

# Confidence threshold
confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Lower values detect more objects but may include false positives"
)

# IOU threshold
iou = st.sidebar.slider(
    "IOU Threshold",
    min_value=0.1,
    max_value=1.0,
    value=0.45,
    step=0.05,
    help="Intersection over Union threshold for NMS"
)

# Load model
@st.cache_resource
def load_model(model_path):
    """Load YOLO model with caching"""
    return YOLO(f"{model_path}.pt")

# Load the selected model
try:
    model = load_model(model_name)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Main content
st.title("🎯 YOLOv8 Object Detection")
st.markdown("---")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📹 Webcam Detection", "🎬 Video Detection", "📊 About"])

# ========== TAB 1: WEBCAM DETECTION ==========
with tab1:
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Options")
        frame_skip = st.number_input(
            "Frame Skip",
            min_value=1,
            max_value=10,
            value=1,
            help="Process every Nth frame"
        )
        max_frames = st.number_input(
            "Max Frames",
            min_value=10,
            max_value=1000,
            value=100,
            help="Maximum frames to capture"
        )
    
    with col1:
        st.subheader("Live Webcam Feed")
        
        start_webcam = st.button("🎥 Start Webcam Detection", key="start_webcam")
        
        if start_webcam:
            stframe = st.empty()
            stats_placeholder = st.empty()
            
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                st.error("❌ Cannot access webcam. Please check camera permissions.")
            else:
                frame_count = 0
                total_detections = 0
                
                try:
                    while frame_count < max_frames:
                        ret, frame = cap.read()
                        
                        if not ret:
                            st.warning("Failed to read frame")
                            break
                        
                        # Resize frame for faster processing
                        frame = cv2.resize(frame, (640, 480))
                        
                        if frame_count % frame_skip == 0:
                            # Run inference
                            results = model.predict(
                                frame,
                                conf=confidence,
                                iou=iou,
                                verbose=False
                            )
                            
                            # Draw annotations
                            annotated_frame = results[0].plot()
                            
                            # Count detections
                            num_detections = len(results[0].boxes)
                            total_detections += num_detections
                            
                            # Display frame
                            stframe.image(
                                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                                use_column_width=True
                            )
                            
                            # Display stats
                            with stats_placeholder.container():
                                col_a, col_b, col_c = st.columns(3)
                                with col_a:
                                    st.metric("Frames Processed", frame_count)
                                with col_b:
                                    st.metric("Objects in Frame", num_detections)
                                with col_c:
                                    st.metric("Avg Objects/Frame", 
                                             round(total_detections / (frame_count + 1), 2))
                        
                        frame_count += 1
                
                finally:
                    cap.release()
                    st.success("✅ Webcam detection completed!")

# ========== TAB 2: VIDEO DETECTION ==========
with tab2:
    st.subheader("Upload and Process Video")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        video_file = st.file_uploader(
            "Choose a video file",
            type=["mp4", "avi", "mov", "mkv", "flv"],
            help="Supported formats: MP4, AVI, MOV, MKV, FLV"
        )
    
    with col2:
        st.subheader("Video Options")
        max_video_frames = st.number_input(
            "Max Frames to Process",
            min_value=1,
            max_value=10000,
            value=300,
            help="Limit processing for faster results"
        )
    
    if video_file is not None:
        # Save uploaded file to temp directory
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            temp_video_path = tmp_file.name
        
        try:
            # Process video
            st.info("Processing video... This may take a moment.")
            progress_bar = st.progress(0)
            
            cap = cv2.VideoCapture(temp_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Create output video writer
            output_path = "output_video.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = None
            
            frame_idx = 0
            detection_summary = {
                "total_detections": 0,
                "frames_with_detections": 0,
                "class_counts": {}
            }
            
            while frame_idx < max_video_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Initialize video writer with frame dimensions
                if out is None:
                    h, w = frame.shape[:2]
                    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                
                # Run inference
                results = model.predict(
                    frame,
                    conf=confidence,
                    iou=iou,
                    verbose=False
                )
                
                # Draw annotations
                annotated_frame = results[0].plot()
                out.write(annotated_frame)
                
                # Update statistics
                num_detections = len(results[0].boxes)
                if num_detections > 0:
                    detection_summary["frames_with_detections"] += 1
                    detection_summary["total_detections"] += num_detections
                    
                    # Count by class
                    if hasattr(results[0], 'names'):
                        for cls_id in results[0].boxes.cls:
                            class_name = results[0].names[int(cls_id)]
                            detection_summary["class_counts"][class_name] = \
                                detection_summary["class_counts"].get(class_name, 0) + 1
                
                # Update progress
                progress = (frame_idx + 1) / min(total_frames, max_video_frames)
                progress_bar.progress(progress)
                
                frame_idx += 1
            
            cap.release()
            if out:
                out.release()
            
            # Display results
            st.success("✅ Video processing completed!")
            
            # Show statistics
            st.subheader("📊 Detection Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Detections", detection_summary["total_detections"])
            with col2:
                st.metric("Frames with Detections", detection_summary["frames_with_detections"])
            with col3:
                st.metric("Total Frames Processed", frame_idx)
            
            # Show class distribution
            if detection_summary["class_counts"]:
                st.subheader("Object Class Distribution")
                class_data = detection_summary["class_counts"]
                st.bar_chart(class_data)
            
            # Download processed video
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    st.download_button(
                        label="📥 Download Processed Video",
                        data=f.read(),
                        file_name="detected_video.mp4",
                        mime="video/mp4"
                    )
        
        finally:
            # Cleanup
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)

# ========== TAB 3: ABOUT ==========
with tab3:
    st.subheader("About YOLOv8 Object Detection")
    
    about_text = """
    ### What is YOLO?
    YOLO (You Only Look Once) is a state-of-the-art real-time object detection algorithm.
    It detects objects in images and videos with high accuracy and speed.
    
    ### Features of this Application:
    - **Real-time Webcam Detection**: Live object detection from your camera
    - **Video Processing**: Batch process video files for object detection
    - **Multiple Model Sizes**: Choose from nano to extra-large models based on your needs
    - **Configurable Parameters**: Adjust confidence and IOU thresholds
    - **Detection Statistics**: View detailed metrics about detected objects
    
    ### Model Information:
    """
    
    st.markdown(about_text)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"**Current Model**: {model_name}")
        st.info(f"**Confidence Threshold**: {confidence}")
        st.info(f"**IOU Threshold**: {iou}")
    
    with col2:
        st.markdown("""
        **Model Sizes:**
        - **YOLOv8n** - Nano (fastest, lowest accuracy)
        - **YOLOv8s** - Small
        - **YOLOv8m** - Medium (recommended)
        - **YOLOv8l** - Large
        - **YOLOv8x** - Extra Large (slowest, highest accuracy)
        """)
    
    st.markdown("---")
    st.markdown("""
    ### Tips for Best Results:
    1. **Good Lighting**: Ensure adequate lighting for better detection
    2. **Clear Objects**: Objects should be clearly visible in the frame
    3. **Right Threshold**: Lower confidence = more detections (may include false positives)
    4. **Model Selection**: Use larger models for better accuracy, smaller models for speed
    
    ### Supported Object Classes:
    YOLOv8 can detect 80+ object classes including:
    person, car, dog, cat, bicycle, truck, bus, motorbike, and many more!
    """)
