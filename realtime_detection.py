import cv2
import numpy as np
from ultralytics import YOLO
import time
from collections import defaultdict

class RealTimeVehicleDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """Initialize the real-time vehicle detector"""
        self.model = YOLO(model_path)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.vehicle_names = ['car', 'motorcycle', 'bus', 'truck']
        self.detection_stats = defaultdict(int)
        
    def detect_vehicles(self, frame):
        """Detect vehicles in a single frame"""
        results = self.model(frame)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id in self.vehicle_classes and confidence > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        vehicle_name = self.vehicle_names[self.vehicle_classes.index(class_id)]
                        
                        detections.append({
                            'class': vehicle_name,
                            'confidence': confidence,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
                        
                        self.detection_stats[vehicle_name] += 1
        
        return detections
    
    def draw_detections(self, frame, detections):
        """Draw bounding boxes and labels on the frame"""
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
        
        return frame
    
    def draw_stats(self, frame):
        """Draw detection statistics on the frame"""
        y_offset = 30
        cv2.putText(frame, "Vehicle Detection Statistics:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        for i, (vehicle, count) in enumerate(self.detection_stats.items()):
            y_offset += 30
            text = f"{vehicle.capitalize()}: {count}"
            cv2.putText(frame, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run_webcam(self, camera_index=0):
        """Run real-time detection on webcam feed"""
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        print("Starting real-time vehicle detection...")
        print("Press 'q' to quit, 'r' to reset statistics")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from camera")
                break
            
            # Detect vehicles
            detections = self.detect_vehicles(frame)
            
            # Draw detections
            frame = self.draw_detections(frame, detections)
            
            # Draw statistics
            frame = self.draw_stats(frame)
            
            # Display frame
            cv2.imshow('Vehicle Detection - Real Time', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.detection_stats.clear()
                print("Statistics reset!")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_video_file(self, video_path):
        """Run detection on a video file"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_delay = int(1000 / fps)  # Delay between frames in milliseconds
        
        print(f"Processing video: {video_path}")
        print("Press 'q' to quit, 'r' to reset statistics, SPACE to pause/resume")
        
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                
                # Detect vehicles
                detections = self.detect_vehicles(frame)
                
                # Draw detections
                frame = self.draw_detections(frame, detections)
                
                # Draw statistics
                frame = self.draw_stats(frame)
            
            # Display frame
            cv2.imshow('Vehicle Detection - Video File', frame)
            
            # Handle key presses
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.detection_stats.clear()
                print("Statistics reset!")
            elif key == ord(' '):
                paused = not paused
                print("Paused" if paused else "Resumed")
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Main function to run the real-time detector"""
    detector = RealTimeVehicleDetector()
    
    print("Vehicle Detection System - Real Time Mode")
    print("1. Webcam detection")
    print("2. Video file detection")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == '1':
        camera_index = input("Enter camera index (default: 0): ").strip()
        camera_index = int(camera_index) if camera_index else 0
        detector.run_webcam(camera_index)
    elif choice == '2':
        video_path = input("Enter path to video file: ").strip()
        detector.run_video_file(video_path)
    else:
        print("Invalid choice. Running webcam detection by default.")
        detector.run_webcam()

if __name__ == "__main__":
    main()
