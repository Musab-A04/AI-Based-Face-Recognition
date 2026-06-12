from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from utils import DEFAULT_OPENCV_DETECTOR_MODEL_PATH, FACE_CASCADE_PATH, FACE_IMAGE_SIZE


@dataclass(slots=True)
class DetectionResult:
    bbox: tuple[int, int, int, int]
    face_image: np.ndarray
    face_data: np.ndarray | None = None


class FaceDetectionError(RuntimeError):
    pass


class NoFaceDetectedError(FaceDetectionError):
    pass


class MultipleFacesDetectedError(FaceDetectionError):
    pass


class FaceDetector:
    def __init__(
        self,
        cascade_path: str = FACE_CASCADE_PATH,
        yunet_model_path: str | Path = DEFAULT_OPENCV_DETECTOR_MODEL_PATH,
    ) -> None:
        self.yunet = None
        self.yunet_model_path = Path(yunet_model_path)
        if self.yunet_model_path.exists() and hasattr(cv2, "FaceDetectorYN_create"):
            self.yunet = cv2.FaceDetectorYN_create(
                str(self.yunet_model_path),
                "",
                (320, 320),
                score_threshold=0.9,
                nms_threshold=0.3,
                top_k=5000,
            )

        self.classifier = cv2.CascadeClassifier(cascade_path)
        if self.classifier.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")

    def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        detections = self._detect_yunet(image)
        if detections:
            return [self._bbox_from_yunet_face(face) for face in detections]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
        )
        return [tuple(map(int, face)) for face in faces]

    def detect_single_face(self, image: np.ndarray) -> DetectionResult:
        detections = self._detect_yunet(image)
        if detections:
            if len(detections) > 1:
                raise MultipleFacesDetectedError("Multiple faces detected. Use an image with one face.")
            face_data = detections[0]
            bbox = self._bbox_from_yunet_face(face_data)
            return DetectionResult(
                bbox=bbox,
                face_image=self.crop_face(image, bbox),
                face_data=face_data,
            )

        faces = self.detect_faces(image)
        if not faces:
            raise NoFaceDetectedError("No face detected.")
        if len(faces) > 1:
            raise MultipleFacesDetectedError("Multiple faces detected. Use an image with one face.")
        bbox = faces[0]
        return DetectionResult(bbox=bbox, face_image=self.crop_face(image, bbox))

    def _detect_yunet(self, image: np.ndarray) -> list[np.ndarray]:
        if self.yunet is None:
            return []
        height, width = image.shape[:2]
        self.yunet.setInputSize((width, height))
        _, faces = self.yunet.detect(image)
        if faces is None:
            return []
        return [face.astype(np.float32) for face in faces]

    @staticmethod
    def _bbox_from_yunet_face(face_data: np.ndarray) -> tuple[int, int, int, int]:
        x, y, w, h = face_data[:4]
        return (int(round(x)), int(round(y)), int(round(w)), int(round(h)))

    @staticmethod
    def crop_face(
        image: np.ndarray,
        bbox: tuple[int, int, int, int],
        output_size: tuple[int, int] = FACE_IMAGE_SIZE,
        margin: float = 0.20,
    ) -> np.ndarray:
        x, y, w, h = bbox
        x_margin = int(w * margin)
        y_margin = int(h * margin)
        x1 = max(0, x - x_margin)
        y1 = max(0, y - y_margin)
        x2 = min(image.shape[1], x + w + x_margin)
        y2 = min(image.shape[0], y + h + y_margin)
        face = image[y1:y2, x1:x2]
        return cv2.resize(face, output_size)
