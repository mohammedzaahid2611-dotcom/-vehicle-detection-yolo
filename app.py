from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import io
from PIL import Image
import json
import os
import time
import threading
from collections import defaultdict
import pandas as pd

app = Flask(__name__)
CORS(app)

# Global variables for detection statistics
detection_stats = {
    'total_detections': 0,
    'vehicle_counts': defaultdict(int),
    'detection_history': [],
    'last_detection_time': None
}

# Initialize YOLO model
model = None

def initialize_model():
    global model
    try:
        model = YOLO('yolov8n.pt')  # Using nano version for faster inference
        print("YOLO model loaded successfully!")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")

def detect_vehicles(image):
    """Detect vehicles in the given image using YOLO v8"""
    global detection_stats
    
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Run detection
        results = model(image)
        
        # Vehicle classes in COCO dataset (YOLO v8 uses COCO classes)
        vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        vehicle_names = ['car', 'motorcycle', 'bus', 'truck']
        
        detections = []
        vehicle_count = defaultdict(int)
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get class and confidence
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    # Check if it's a vehicle
                    if class_id in vehicle_classes and confidence > 0.5:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        vehicle_name = vehicle_names[vehicle_classes.index(class_id)]
                        vehicle_count[vehicle_name] += 1
                        
                        detections.append({
                            'class': vehicle_name,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
        
        # Update statistics
        detection_stats['total_detections'] += len(detections)
        detection_stats['last_detection_time'] = time.time()
        
        for vehicle, count in vehicle_count.items():
            detection_stats['vehicle_counts'][vehicle] += count
        
        # Add to history
        detection_stats['detection_history'].append({
            'timestamp': time.time(),
            'detections': len(detections),
            'vehicles': dict(vehicle_count)
        })
        
        # Keep only last 100 detections in history
        if len(detection_stats['detection_history']) > 100:
            detection_stats['detection_history'] = detection_stats['detection_history'][-100:]
        
        return detections, None
        
    except Exception as e:
        return None, str(e)

def draw_detections(image, detections):
    """Draw bounding boxes and labels on the image"""
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        class_name = detection['class']
        confidence = detection['confidence']
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{class_name}: {confidence:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Check file type
        file_extension = file.filename.lower().split('.')[-1]
        
        if file_extension in ['jpg', 'jpeg', 'png', 'bmp', 'gif']:
            # Process as image
            return process_image(file)
        elif file_extension in ['mp4', 'avi', 'mov', 'mkv', 'wmv']:
            # Process as video
            return process_video(file)
        else:
            return jsonify({'error': 'Unsupported file type. Please upload an image or video file.'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_image(file):
    """Process uploaded image file"""
    try:
        # Read image
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Detect vehicles
        detections, error = detect_vehicles(image)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Draw detections
        result_image = draw_detections(image.copy(), detections)
        
        # Convert result to base64
        _, buffer = cv2.imencode('.jpg', result_image)
        result_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'success': True,
            'type': 'image',
            'detections': detections,
            'image': result_base64,
            'stats': {
                'total_vehicles': len(detections),
                'vehicle_breakdown': {det['class']: sum(1 for d in detections if d['class'] == det['class']) for det in detections}
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video(file):
    """Process uploaded video file"""
    try:
        # Save uploaded video temporarily
        video_filename = f"temp_video_{int(time.time())}.{file.filename.split('.')[-1]}"
        video_path = os.path.join(os.getcwd(), video_filename)
        file.save(video_path)
        
        # Process video and create output
        output_filename = f"output_video_{int(time.time())}.avi"
        output_path = os.path.join(os.getcwd(), output_filename)
        
        # Process video
        result = process_video_file(video_path, output_path)
        
        # Clean up input file
        if os.path.exists(video_path):
            os.remove(video_path)
        
        if result['success']:
            return jsonify({
                'success': True,
                'type': 'video',
                'message': 'Video processed successfully!',
                'output_file': output_filename,
                'stats': result['stats'],
                'total_frames': result['total_frames'],
                'detection_summary': result['detection_summary']
            })
        else:
            return jsonify({'error': result['error']}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video_file(input_path, output_path):
    """Process video file and create output with detections"""
    try:
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            return {'success': False, 'error': 'Could not open video file'}
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define codec and create VideoWriter - use more compatible codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # Change extension to .avi for better compatibility
        output_path = output_path.replace('.mp4', '.avi')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_detections = 0
        detection_summary = defaultdict(int)
        
        print(f"Processing video: {total_frames} frames at {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect vehicles in frame
            detections, error = detect_vehicles(frame)
            
            if not error and detections:
                # Draw detections
                frame = draw_detections(frame, detections)
                
                # Update statistics
                total_detections += len(detections)
                for det in detections:
                    detection_summary[det['class']] += 1
            
            # Write frame to output video
            out.write(frame)
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:  # Every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
        
        # Release everything
        cap.release()
        out.release()
        
        return {
            'success': True,
            'total_frames': frame_count,
            'stats': {
                'total_detections': total_detections,
                'detection_summary': dict(detection_summary)
            },
            'detection_summary': dict(detection_summary)
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

@app.route('/download/<filename>')
def download_file(filename):
    """Download processed video file"""
    try:
        file_path = os.path.join(os.getcwd(), filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    """Get detection statistics for dashboard"""
    return jsonify(detection_stats)

@app.route('/history')
def get_history():
    """Get detection history for charts"""
    history = detection_stats['detection_history']
    
    # Convert to DataFrame for easier processing
    df = pd.DataFrame(history)
    
    if df.empty:
        return jsonify({'timestamps': [], 'detections': [], 'vehicles': {}})
    
    # Prepare data for charts
    timestamps = [time.strftime('%H:%M:%S', time.localtime(ts)) for ts in df['timestamp']]
    detection_counts = df['detections'].tolist()
    
    # Vehicle counts over time
    vehicle_data = {}
    for vehicle in ['car', 'motorcycle', 'bus', 'truck']:
        vehicle_data[vehicle] = [entry.get(vehicle, 0) for entry in df['vehicles']]
    
    return jsonify({
        'timestamps': timestamps,
        'detections': detection_counts,
        'vehicles': vehicle_data
    })

if __name__ == '__main__':
    # Initialize model in a separate thread
    threading.Thread(target=initialize_model).start()
    app.run(debug=True, host='0.0.0.0', port=5000)
