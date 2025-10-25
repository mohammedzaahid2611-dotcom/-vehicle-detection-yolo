#!/usr/bin/env python3
"""
Create a simple test video with moving shapes for vehicle detection testing
"""

import cv2
import numpy as np
import os

def create_test_video():
    """Create a test video with moving shapes that look like vehicles"""
    
    # Video properties
    width, height = 640, 480
    fps = 30
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('test_vehicles.mp4', fourcc, fps, (width, height))
    
    print(f"Creating test video: {total_frames} frames at {fps} FPS")
    
    for frame_num in range(total_frames):
        # Create black background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some background elements
        cv2.rectangle(frame, (0, height-50), (width, height), (100, 100, 100), -1)  # Road
        cv2.line(frame, (0, height-25), (width, height-25), (255, 255, 255), 2)  # Road line
        
        # Animate moving "vehicles"
        t = frame_num / fps  # Time in seconds
        
        # Car 1 - moving left to right
        car1_x = int((frame_num * 3) % (width + 100)) - 50
        car1_y = height - 100
        cv2.rectangle(frame, (car1_x, car1_y), (car1_x + 80, car1_y + 40), (0, 0, 255), -1)  # Red car
        cv2.rectangle(frame, (car1_x + 10, car1_y - 10), (car1_x + 70, car1_y), (0, 0, 255), -1)  # Car roof
        cv2.circle(frame, (car1_x + 20, car1_y + 35), 8, (0, 0, 0), -1)  # Wheel
        cv2.circle(frame, (car1_x + 60, car1_y + 35), 8, (0, 0, 0), -1)  # Wheel
        
        # Car 2 - moving right to left
        car2_x = width - int((frame_num * 2) % (width + 100)) - 30
        car2_y = height - 150
        cv2.rectangle(frame, (car2_x, car2_y), (car2_x + 70, car2_y + 35), (0, 255, 0), -1)  # Green car
        cv2.rectangle(frame, (car2_x + 10, car2_y - 8), (car2_x + 60, car2_y), (0, 255, 0), -1)  # Car roof
        cv2.circle(frame, (car2_x + 15, car2_y + 30), 7, (0, 0, 0), -1)  # Wheel
        cv2.circle(frame, (car2_x + 55, car2_y + 30), 7, (0, 0, 0), -1)  # Wheel
        
        # Truck - moving slowly
        if frame_num % 2 == 0:  # Move every other frame
            truck_x = int((frame_num * 1) % (width + 150)) - 75
            truck_y = height - 200
            cv2.rectangle(frame, (truck_x, truck_y), (truck_x + 120, truck_y + 50), (255, 0, 0), -1)  # Blue truck
            cv2.rectangle(frame, (truck_x + 20, truck_y - 15), (truck_x + 100, truck_y), (255, 0, 0), -1)  # Truck cab
            cv2.circle(frame, (truck_x + 25, truck_y + 45), 10, (0, 0, 0), -1)  # Wheel
            cv2.circle(frame, (truck_x + 95, truck_y + 45), 10, (0, 0, 0),  -1)  # Wheel
        
        # Motorcycle - fast movement
        if frame_num % 3 == 0:  # Move every 3rd frame
            bike_x = int((frame_num * 4) % (width + 50)) - 25
            bike_y = height - 80
            cv2.circle(frame, (bike_x, bike_y), 12, (0, 255, 255), -1)  # Yellow motorcycle
            cv2.circle(frame, (bike_x - 10, bike_y), 6, (0, 0, 0), -1)  # Wheel
            cv2.circle(frame, (bike_x + 10, bike_y), 6, (0, 0, 0), -1)  # Wheel
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {t:.1f}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write frame
        out.write(frame)
        
        # Progress update
        if frame_num % 30 == 0:
            progress = (frame_num / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_num}/{total_frames} frames)")
    
    # Release video writer
    out.release()
    
    print(f"Test video created: test_vehicles.mp4")
    print(f"Duration: {duration} seconds")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print("You can now upload this video to test vehicle detection!")

if __name__ == "__main__":
    create_test_video()
