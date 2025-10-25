# 🚗 Vehicle Detection System Using YOLO v8

A comprehensive machine learning-based vehicle detection system that uses YOLO v8 for real-time detection of cars, motorcycles, buses, and trucks. The system includes both a web dashboard and real-time video processing capabilities.

## 🌟 Features

- **Real-time Vehicle Detection**: Detect cars, motorcycles, buses, and trucks in real-time
- **Web Dashboard**: Interactive web interface for image and video upload
- **Video Processing**: Process video files with detection overlays
- **Live Statistics**: Real-time detection statistics and analytics
- **Multiple Input Sources**: Support for webcam, image files, and video files
- **Download Results**: Download processed videos with detection annotations

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam (for real-time detection)
- Modern web browser

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/vehicle-detection-yolo.git
   cd vehicle-detection-yolo
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web application**
   ```bash
   python app.py
   ```

4. **Access the dashboard**
   Open your browser and go to: http://localhost:5000

## 📖 Usage

### Web Dashboard

1. **Start the application**: Run `python app.py`
2. **Open browser**: Navigate to http://localhost:5000
3. **Upload files**: Drag and drop images or videos
4. **View results**: See detection results with bounding boxes
5. **Download**: Download processed videos

### Real-time Detection

1. **Webcam detection**:
   ```bash
   python realtime_detection.py
   # Select option 1 for webcam
   ```

2. **Video file detection**:
   ```bash
   python realtime_detection.py
   # Select option 2 and provide video path
   ```

### Quick Start Scripts

- **Start web app**: `start_web_app.bat` (Windows)
- **Start real-time**: `start_realtime.bat` (Windows)
- **Webcam only**: `python start_webcam.py`

## 🛠️ Technical Details

### Architecture

- **Backend**: Flask web framework
- **AI Model**: YOLO v8 (Ultralytics)
- **Computer Vision**: OpenCV
- **Frontend**: HTML5, Bootstrap, JavaScript, Plotly.js
- **Video Processing**: XVID codec for maximum compatibility

### Supported File Formats

- **Images**: JPG, JPEG, PNG, BMP, GIF
- **Videos**: MP4, AVI, MOV, MKV, WMV

### Detection Classes

- **Car** (Class ID: 2)
- **Motorcycle** (Class ID: 3)
- **Bus** (Class ID: 5)
- **Truck** (Class ID: 7)

### Performance Metrics

- **Processing Speed**: 70-150ms per frame
- **Detection Accuracy**: 20-26 vehicles per frame
- **Confidence Threshold**: 0.5 (50%)
- **Input Resolution**: 384x640 pixels

## 📁 Project Structure

```
vehicle-detection-yolo/
├── app.py                      # Main Flask application
├── realtime_detection.py       # Real-time detection module
├── start_webcam.py            # Webcam detection script
├── requirements.txt           # Python dependencies
├── templates/
│   └── index.html             # Web dashboard template
├── static/                    # CSS, JS, and assets
├── test_installation.py       # Installation verification
├── create_test_video.py       # Test video generator
├── play_video.py              # Video player utility
└── README.md                  # This file
```

## 🔧 Configuration

### Model Configuration

The system uses YOLO v8 Nano model by default. To use a different model:

```python
# In app.py or realtime_detection.py
model = YOLO('yolov8s.pt')  # Small model
model = YOLO('yolov8m.pt')  # Medium model
model = YOLO('yolov8l.pt')  # Large model
model = YOLO('yolov8x.pt')  # Extra large model
```

### Confidence Threshold

Adjust detection sensitivity:

```python
# In detect_vehicles function
if class_id in vehicle_classes and confidence > 0.3:  # Lower threshold
```

## 📊 API Endpoints

- `GET /` - Main dashboard
- `POST /upload` - Upload and process files
- `GET /download/<filename>` - Download processed videos
- `GET /stats` - Get detection statistics
- `GET /history` - Get detection history

## 🧪 Testing

### Test Installation
```bash
python test_installation.py
```

### Create Test Video
```bash
python create_test_video.py
```

### Play Videos
```bash
python play_video.py
```

## 🚨 Troubleshooting

### Common Issues

1. **"Model not loaded" error**
   - Ensure internet connection for first-time model download
   - Check available disk space

2. **Video playback issues**
   - Use VLC Media Player for best compatibility
   - Convert videos to AVI format if needed

3. **Webcam not working**
   - Check camera permissions
   - Try different camera index (0, 1, 2)

4. **Slow performance**
   - Use YOLO v8 Nano model
   - Reduce input resolution
   - Close other applications

## 📈 Performance Optimization

### For Better Speed
- Use YOLO v8 Nano model
- Reduce input image size
- Use GPU acceleration (CUDA)

### For Better Accuracy
- Use YOLO v8 Large or Extra Large model
- Increase confidence threshold
- Use higher resolution input

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO v8
- [OpenCV](https://opencv.org/) for computer vision
- [Flask](https://flask.palletsprojects.com/) for web framework
- [COCO Dataset](https://cocodataset.org/) for training data

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/vehicle-detection-yolo/issues) page
2. Create a new issue with detailed description
3. Include system information and error logs

## 🔄 Updates

### Version 1.0.0
- Initial release
- YOLO v8 integration
- Web dashboard
- Real-time detection
- Video processing
- Statistics tracking

---

**Made with ❤️ for the computer vision community**