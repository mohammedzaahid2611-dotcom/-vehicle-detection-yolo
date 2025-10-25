#!/usr/bin/env python3
"""
Convert existing MP4 videos to AVI format for better compatibility
"""

import cv2
import os
import glob

def convert_mp4_to_avi():
    """Convert all MP4 files in the directory to AVI format"""
    
    # Find all MP4 files
    mp4_files = glob.glob("output_video_*.mp4")
    
    if not mp4_files:
        print("No MP4 files found to convert.")
        return
    
    print(f"Found {len(mp4_files)} MP4 files to convert:")
    for file in mp4_files:
        print(f"  - {file}")
    
    for mp4_file in mp4_files:
        print(f"\nConverting {mp4_file}...")
        
        # Open input video
        cap = cv2.VideoCapture(mp4_file)
        
        if not cap.isOpened():
            print(f"Error: Could not open {mp4_file}")
            continue
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  Properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Create output filename
        avi_file = mp4_file.replace('.mp4', '_converted.avi')
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(avi_file, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            
            # Progress update
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"  Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Release everything
        cap.release()
        out.release()
        
        print(f"  ✅ Converted to: {avi_file}")
    
    print("\n🎉 All conversions complete!")
    print("The AVI files should now work with all video players.")

if __name__ == "__main__":
    convert_mp4_to_avi()
