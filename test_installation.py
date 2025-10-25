#!/usr/bin/env python3
"""
Test script to verify the vehicle detection system installation
"""

import sys
import importlib
import subprocess

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing package imports...")
    
    required_packages = [
        'ultralytics',
        'cv2',
        'numpy',
        'flask',
        'PIL',
        'pandas',
        'plotly'
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                importlib.import_module(package)
            print(f"[OK] {package}")
        except ImportError as e:
            print(f"[FAIL] {package}: {e}")
            failed_imports.append(package)
    
    return failed_imports

def test_yolo_model():
    """Test if YOLO model can be loaded"""
    print("\nTesting YOLO model loading...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("[OK] YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"[FAIL] YOLO model loading failed: {e}")
        return False

def test_opencv():
    """Test OpenCV functionality"""
    print("\nTesting OpenCV...")
    
    try:
        import cv2
        import numpy as np
        
        # Test basic OpenCV operations
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (10, 10), (90, 90), (0, 255, 0), 2)
        
        # Test camera access
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("[OK] Camera access available")
            cap.release()
        else:
            print("[WARN] Camera not available (this is normal if no camera is connected)")
        
        print("[OK] OpenCV working correctly")
        return True
    except Exception as e:
        print(f"[FAIL] OpenCV test failed: {e}")
        return False

def test_flask():
    """Test Flask functionality"""
    print("\nTesting Flask...")
    
    try:
        from flask import Flask
        app = Flask(__name__)
        
        @app.route('/')
        def test():
            return "Test"
        
        print("[OK] Flask working correctly")
        return True
    except Exception as e:
        print(f"[FAIL] Flask test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Vehicle Detection System - Installation Test")
    print("=" * 50)
    
    # Test imports
    failed_imports = test_imports()
    
    # Test YOLO model
    yolo_ok = test_yolo_model()
    
    # Test OpenCV
    opencv_ok = test_opencv()
    
    # Test Flask
    flask_ok = test_flask()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    if failed_imports:
        print(f"[FAIL] Failed imports: {', '.join(failed_imports)}")
        print("  Run: pip install -r requirements.txt")
    else:
        print("[OK] All packages imported successfully")
    
    if yolo_ok:
        print("[OK] YOLO model ready")
    else:
        print("[FAIL] YOLO model not ready")
        print("  Run: pip install ultralytics")
    
    if opencv_ok:
        print("[OK] OpenCV ready")
    else:
        print("[FAIL] OpenCV not ready")
        print("  Run: pip install opencv-python")
    
    if flask_ok:
        print("[OK] Flask ready")
    else:
        print("[FAIL] Flask not ready")
        print("  Run: pip install flask")
    
    # Overall status
    all_ok = not failed_imports and yolo_ok and opencv_ok and flask_ok
    
    if all_ok:
        print("\n[SUCCESS] All tests passed! The system is ready to use.")
        print("\nTo start the web dashboard:")
        print("  python app.py")
        print("\nTo start real-time detection:")
        print("  python realtime_detection.py")
    else:
        print("\n[ERROR] Some tests failed. Please fix the issues above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
