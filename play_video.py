#!/usr/bin/env python3
"""
Simple video player to view the processed videos
"""

import cv2
import os
import sys

def play_video(video_path):
    """Play a video file using OpenCV"""
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Playing: {video_path}")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps:.1f}")
    print(f"Duration: {duration:.1f} seconds")
    print(f"Total frames: {total_frames}")
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 'p' to pause/resume")
    print("  - Press 'r' to restart")
    print("  - Press 's' to save current frame")
    print("\nStarting playback...")
    
    paused = False
    frame_count = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video reached")
                break
            
            frame_count += 1
        
        # Add frame info overlay
        info_text = f"Frame: {frame_count}/{total_frames} | Time: {frame_count/fps:.1f}s"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Vehicle Detection Video Player', frame)
        
        # Handle key presses
        key = cv2.waitKey(int(1000/fps)) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            print("Restarted")
        elif key == ord('s'):
            filename = f"frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame {frame_count} as {filename}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Video playback ended.")

def list_videos():
    """List all available video files"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for file in os.listdir('.'):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    if not video_files:
        print("No video files found in current directory.")
        return []
    
    print("Available video files:")
    for i, file in enumerate(video_files, 1):
        size = os.path.getsize(file) / (1024*1024)  # Size in MB
        print(f"  {i}. {file} ({size:.1f} MB)")
    
    return video_files

def main():
    """Main function"""
    print("Vehicle Detection Video Player")
    print("=" * 40)
    
    # List available videos
    video_files = list_videos()
    
    if not video_files:
        return
    
    # Let user choose video
    try:
        choice = input(f"\nEnter video number (1-{len(video_files)}) or press Enter for first video: ").strip()
        
        if choice == "":
            video_index = 0
        else:
            video_index = int(choice) - 1
        
        if 0 <= video_index < len(video_files):
            selected_video = video_files[video_index]
            play_video(selected_video)
        else:
            print("Invalid choice!")
    
    except (ValueError, KeyboardInterrupt):
        print("\nExiting...")

if __name__ == "__main__":
    main()
