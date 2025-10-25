#!/usr/bin/env python3
"""
Simple script to start webcam detection automatically
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict

def main():
    print("Starting Vehicle Detection - Webcam Mode")
    print("Press 'q' to quit, 'r' to reset statistics")
    
    # Initialize YOLO model
    print("Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    
    # Vehicle classes
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    vehicle_names = ['car', 'motorcycle', 'bus', 'truck']
    detection_stats = defaultdict(int)
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("Camera opened successfully!")
    print("Starting detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            break
        
        # Detect vehicles
        results = model(frame)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id in vehicle_classes and confidence > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        vehicle_name = vehicle_names[vehicle_classes.index(class_id)]
                        
                        detections.append({
                            'class': vehicle_name,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
                        
                        detection_stats[vehicle_name] += 1
        
        # Draw detections
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class']
            confidence = detection['confidence']
            
            # Choose color based on vehicle type
            colors = {
                'car': (0, 255, 0),      # Green
                'motorcycle': (0, 255, 255),  # Yellow
                'bus': (255, 0, 0),      # Red
                'truck': (255, 0, 255)   # Magenta
            }
            color = colors.get(class_name, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Draw statistics
        y_offset = 30
        cv2.putText(frame, "Vehicle Detection Statistics:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i, (vehicle, count) in enumerate(detection_stats.items()):
            y_offset += 30
            text = f"{vehicle.capitalize()}: {count}"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow('Vehicle Detection - Live Camera', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detection_stats.clear()
            print("Statistics reset!")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Detection stopped.")

if __name__ == "__main__":
    main()
