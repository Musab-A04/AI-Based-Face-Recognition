from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_OPENCV_DETECTOR_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
DEFAULT_OPENCV_FACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy installed: {importlib.util.find_spec('numpy') is not None}")
    print(f"DeepFace installed: {importlib.util.find_spec('deepface') is not None}")

    try:
        import cv2
    except ImportError:
        print("OpenCV installed: False")
        print("OpenCV FaceRecognizerSF available: False")
        print("Install dependencies with: pip install -r requirements.txt")
    else:
        print("OpenCV installed: True")
        print(f"OpenCV: {cv2.__version__}")
        print(f"OpenCV FaceDetectorYN available: {hasattr(cv2, 'FaceDetectorYN_create')}")
        print(f"OpenCV FaceRecognizerSF available: {hasattr(cv2, 'FaceRecognizerSF_create')}")
        if not hasattr(cv2, "FaceDetectorYN_create") or not hasattr(cv2, "FaceRecognizerSF_create"):
            print("Install opencv-contrib-python to use the default OpenCV backend.")

    print(f"YuNet detector model path: {DEFAULT_OPENCV_DETECTOR_MODEL_PATH}")
    print(f"YuNet detector model exists: {DEFAULT_OPENCV_DETECTOR_MODEL_PATH.exists()}")
    print(f"SFace model path: {DEFAULT_OPENCV_FACE_MODEL_PATH}")
    print(f"SFace model exists: {DEFAULT_OPENCV_FACE_MODEL_PATH.exists()}")

    if not DEFAULT_OPENCV_DETECTOR_MODEL_PATH.exists():
        print("Download face_detection_yunet_2023mar.onnx into the models/ folder.")
    if not DEFAULT_OPENCV_FACE_MODEL_PATH.exists():
        print("Download face_recognition_sface_2021dec.onnx into the models/ folder.")


if __name__ == "__main__":
    main()
