# YOLOv8 Object Detection UI

A professional, user-friendly web interface for YOLOv8 object detection.

## Features

✨ **Live Webcam Detection** - Real-time object detection from your camera
📹 **Video Processing** - Process and analyze video files with detected bounding boxes
⚙️ **Configurable Parameters** - Adjust confidence thresholds, IOU levels, and model selection
📊 **Detection Statistics** - View detailed metrics and class distribution
🎯 **Multiple Models** - Choose from nano to extra-large YOLOv8 models

## Installation

### 1. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

### 📹 Webcam Detection
1. Go to the "Webcam Detection" tab
2. Configure options (frame skip, max frames)
3. Click "Start Webcam Detection"
4. Press 'q' or wait for max frames to stop

### 🎬 Video Detection
1. Go to the "Video Detection" tab
2. Upload a video file (MP4, AVI, MOV, MKV, FLV)
3. Set max frames to process
4. View detection statistics and class distribution
5. Download the annotated video

### ⚙️ Configuration (Sidebar)
- **Model Selection**: Choose YOLOv8 variant (nano to extra-large)
- **Confidence Threshold**: Adjust detection sensitivity (0.1-1.0)
- **IOU Threshold**: Fine-tune non-maximum suppression (0.1-1.0)

## Model Information

| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| YOLOv8n | ⚡⚡⚡ Fastest | ⭐⭐ Lower | Nano |
| YOLOv8s | ⚡⚡ Fast | ⭐⭐⭐ Medium | Small |
| YOLOv8m | ⚡ Moderate | ⭐⭐⭐⭐ Good | Medium |
| YOLOv8l | 🐢 Slow | ⭐⭐⭐⭐⭐ Excellent | Large |
| YOLOv8x | 🐢🐢 Slowest | ⭐⭐⭐⭐⭐ Best | Extra-Large |

**Recommended**: YOLOv8m for balanced performance

## Tips for Best Results

✅ Ensure good lighting in your environment
✅ Keep objects clearly visible in the frame
✅ Adjust confidence threshold based on your needs
✅ Use larger models for higher accuracy
✅ Use smaller models for faster processing

## Supported Object Classes

The model can detect 80+ object classes including:
- **Persons**: person
- **Vehicles**: car, truck, bus, bicycle, motorbike, train, airplane
- **Animals**: dog, cat, bird, horse, cow, sheep, elephant, zebra, giraffe
- **Sports**: sports ball, baseball bat, baseball glove, skateboard, tennis racket
- **Household**: bottle, wine glass, cup, fork, knife, spoon, bowl, chair, sofa, bed
- **And many more!**

## Troubleshooting

**Issue**: Webcam not working
- Check camera permissions in your OS settings
- Try a different camera application first
- Restart the app

**Issue**: Slow performance
- Use a smaller model (nano or small)
- Reduce frame resolution
- Increase frame skip value
- Reduce max frames

**Issue**: Missing detections
- Increase max frames for longer processing
- Lower confidence threshold
- Ensure good lighting
- Use a larger model

## System Requirements

- Python 3.8+
- 4GB RAM minimum (8GB+ recommended)
- GPU recommended for real-time processing (NVIDIA CUDA)
- Webcam for live detection (optional)

## License

MIT License

---

**Made with ❤️ for Object Detection**
