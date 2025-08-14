# AthleteRise – AI-Powered Cricket Analytics

Real-time cover drive analysis system that processes full cricket videos to provide biomechanical insights and performance evaluation using a modular, maintainable architecture.

## 🏏 Features

- **Full Video Processing**: Analyzes entire cricket videos frame-by-frame in real-time
- **Pose Estimation**: Uses MediaPipe for accurate body keypoint detection
- **Biomechanical Metrics**: Computes elbow angle, spine lean, head alignment, and foot direction
- **Live Overlays**: Real-time feedback with pose skeleton and metric displays
- **Performance Scoring**: Comprehensive evaluation across 5 categories (1-10 scale)
- **Interactive Web App**: Streamlit interface for easy video analysis
- **Modular Architecture**: Clean separation of concerns for maintainability

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenCV-compatible system
- Internet connection (for video download)

### Installation

1. **Clone/Download the project**
```bash
cd /path/to/AthleteRise
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Usage

#### Option 1: Original Monolithic Script
```bash
python cover_drive_analysis_realtime.py
```

#### Option 2: Refactored Modular Script (Recommended)
```bash
python cover_drive_analysis_realtime_refactored.py
```

#### Option 3: Interactive Web App
```bash
streamlit run app.py
```

## 🏗️ Project Structure

The project has been refactored into a modular architecture for better maintainability:

```
AthleteRise/
├── modules/                                    # Core analysis modules
│   ├── __init__.py                            # Module exports
│   ├── pose_estimator.py                      # MediaPipe pose estimation
│   ├── metrics_calculator.py                 # Biomechanical metrics computation
│   ├── video_processor.py                    # Video I/O and processing
│   ├── evaluator.py                          # Performance evaluation & scoring
│   └── utils.py                              # Utility functions
├── cover_drive_analysis_realtime.py          # Original monolithic script
├── cover_drive_analysis_realtime_refactored.py # Refactored modular script ⭐
├── app.py                                     # Streamlit web interface ⭐
├── requirements.txt                           # Python dependencies
├── README.md                                  # This documentation
└── output/                                    # Generated analysis results
    ├── annotated_video.mp4                   # Video with overlays
    ├── evaluation.json                       # Detailed results (JSON)
    └── evaluation.txt                        # Human-readable report
```

### Module Responsibilities

- **`pose_estimator.py`**: MediaPipe integration, keypoint extraction, skeleton drawing
- **`metrics_calculator.py`**: Biomechanical calculations, angle computations, feedback generation
- **`video_processor.py`**: Video download, frame processing, overlay rendering
- **`evaluator.py`**: Performance scoring, evaluation reports, improvement recommendations
- **`utils.py`**: Common utilities, formatting, validation, system info

## 📊 Analysis Metrics

### Biomechanical Measurements
1. **Front Elbow Angle**: Shoulder-elbow-wrist angle (optimal: 90-130°)
2. **Spine Lean**: Hip-shoulder line vs vertical (optimal: 10-25°)
3. **Head-over-Knee Alignment**: Vertical distance (optimal: <50px)
4. **Front Foot Direction**: Foot angle vs crease (optimal: ±15°)

### Performance Categories
- **Footwork** (1-10): Based on front foot positioning
- **Head Position** (1-10): Head stability and alignment
- **Swing Control** (1-10): Elbow positioning and swing path
- **Balance** (1-10): Spine angle and posture
- **Follow-through** (1-10): Technique consistency

## 📁 Output Files

After analysis, the following files are generated in the `output/` directory:

- `annotated_video.mp4`: Original video with pose overlays and real-time metrics
- `evaluation.json`: Detailed analysis results in JSON format
- `evaluation.txt`: Human-readable performance report

## ⚙️ Configuration

### Threshold Customization
Edit the `thresholds` dictionary in `CricketAnalyzer.__init__()`:

```python
self.thresholds = {
    'elbow_angle_good': (90, 130),      # Elbow angle range
    'spine_lean_good': (10, 25),        # Spine lean range
    'head_alignment_good': 50,          # Max head-knee distance (px)
    'foot_angle_good': (-15, 15)        # Foot angle range (degrees)
}
```

## 🎯 Technical Details

### Architecture
- **Pose Estimation**: MediaPipe Pose (lightweight model for speed)
- **Video Processing**: OpenCV with real-time frame processing
- **Metrics Computation**: Custom biomechanical analysis algorithms
- **Visualization**: Real-time overlay rendering with feedback

### Performance Optimization
- Lightweight MediaPipe model (complexity=1)
- Efficient frame processing pipeline
- Target: ≥10 FPS on CPU
- Memory-efficient keypoint storage

### Error Handling
- Graceful handling of missing pose detections
- Robust angle calculation with edge case protection
- Fallback scoring for incomplete data

## 🔧 Troubleshooting

### Common Issues

1. **Video Download Fails**
   - Check internet connection
   - Verify YouTube URL accessibility
   - Try different video quality settings

2. **Low Processing Speed**
   - Reduce video resolution in download settings
   - Close other applications to free up CPU
   - Consider using GPU acceleration (modify MediaPipe config)

3. **Pose Detection Issues**
   - Ensure good lighting in video
   - Check if player is clearly visible
   - Adjust MediaPipe confidence thresholds

### Dependencies Issues
```bash
# If MediaPipe installation fails
pip install --upgrade pip
pip install mediapipe --no-cache-dir

# If OpenCV issues occur
pip uninstall opencv-python
pip install opencv-python-headless
```

## 📈 Advanced Features (Bonus)

### Phase Segmentation
- Automatic detection of cricket shot phases
- Stance → Stride → Downswing → Impact → Follow-through

### Contact Detection
- Identifies likely bat-ball contact moments
- Uses motion peaks and velocity analysis

### Comparative Analysis
- Benchmark against ideal cover drive technique
- Deviation analysis and improvement suggestions

## 🎨 Streamlit Web Interface

The included Streamlit app provides:
- Video upload interface
- Real-time processing progress
- Interactive results visualization
- Downloadable analysis reports

## 📝 Assumptions & Limitations

### Assumptions
- Right-handed batsman (can be modified for left-handed)
- Standard cricket video orientation
- Player clearly visible throughout the shot
- Reasonable video quality and lighting

### Limitations
- Bat tracking not implemented in base version
- Single-player analysis only
- Requires clear pose visibility
- Performance depends on video quality

### Future Enhancements
- Multi-player analysis
- Advanced bat tracking
- 3D pose estimation
- Machine learning-based technique classification
- Integration with wearable sensors

## 📄 License

This project is developed for AthleteRise cricket analytics assignment.

## 🤝 Contributing

For improvements and bug fixes:
1. Fork the repository
2. Create feature branch
3. Submit pull request with detailed description

## 📞 Support

For technical issues or questions, please refer to the troubleshooting section or create an issue in the project repository.

---

**AthleteRise Team** | *Revolutionizing Cricket Analytics with AI*
