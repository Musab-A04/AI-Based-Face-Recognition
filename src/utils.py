from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import re
import time

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_DIR = PROJECT_ROOT / "database"
IMAGES_DIR = PROJECT_ROOT / "images"
PROFILE_IMAGES_DIR = IMAGES_DIR / "profiles"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
MODELS_DIR = PROJECT_ROOT / "models"
DB_PATH = DATABASE_DIR / "people.db"

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
FACE_IMAGE_SIZE = (160, 160)
DEFAULT_EMBEDDING_BACKEND = "opencv"
DEFAULT_DEEPFACE_MODEL_NAME = "Facenet512"
DEFAULT_OPENCV_DETECTOR_MODEL_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
DEFAULT_OPENCV_FACE_MODEL_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
DEFAULT_SIMILARITY_THRESHOLD = 0.62
DEFAULT_MATCH_MARGIN = 0.06
DEFAULT_TOP_K_SCORES = 3
DEFAULT_ENROLLMENT_SAMPLES = 5
CAMERA_BACKENDS = {
    "auto": 0,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}
ROTATION_OPTIONS = {"none", "cw", "ccw", "180"}


@dataclass(slots=True)
class MatchResult:
    person_id: int | None
    name: str
    similarity: float
    is_known: bool
    profile: dict[str, Any] | None = None
    runner_up_similarity: float = 0.0
    margin: float = 0.0


def ensure_directories() -> None:
    for path in (DATABASE_DIR, IMAGES_DIR, PROFILE_IMAGES_DIR, EMBEDDINGS_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def to_project_relative_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def timestamp_string() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "person"


def save_embedding(embedding: np.ndarray, prefix: str) -> Path:
    ensure_directories()
    path = EMBEDDINGS_DIR / f"{prefix}_{timestamp_string()}.npy"
    np.save(path, embedding.astype(np.float32))
    return path


def load_embedding(path: str | Path) -> np.ndarray:
    return np.load(resolve_project_path(path))


def save_image(image: np.ndarray, prefix: str) -> Path:
    ensure_directories()
    path = IMAGES_DIR / f"{prefix}_{timestamp_string()}.jpg"
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save image to {path}")
    return path


def save_profile_image(image: np.ndarray, prefix: str) -> Path:
    ensure_directories()
    path = PROFILE_IMAGES_DIR / f"{prefix}_profile_{timestamp_string()}.jpg"
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to save profile image to {path}")
    return path


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    vector = embedding.astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("Embedding norm is zero.")
    return vector / norm


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    a = normalize_embedding(embedding_a)
    b = normalize_embedding(embedding_b)
    return float(np.dot(a, b))


def open_camera(camera_index: int, backend: str = "auto") -> cv2.VideoCapture:
    backend_id = CAMERA_BACKENDS.get(backend)
    if backend_id is None:
        raise ValueError(f"Unsupported camera backend: {backend}")
    if backend_id == 0:
        capture = cv2.VideoCapture(camera_index)
    else:
        capture = cv2.VideoCapture(camera_index, backend_id)

    return capture


def read_camera_frame(capture: cv2.VideoCapture, attempts: int = 30) -> np.ndarray:
    for _ in range(attempts):
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.05)
    raise RuntimeError("Failed to read frame from webcam.")


def rotate_frame(frame: np.ndarray, rotation: str = "none") -> np.ndarray:
    if rotation == "none":
        return frame
    if rotation == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    raise ValueError(f"Unsupported rotation: {rotation}")


def draw_face_label(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    title: str,
    detail_lines: list[str] | None = None,
) -> np.ndarray:
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    lines = [title] + (detail_lines or [])
    text_y = max(25, y - 10)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, text_y + index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return frame


def build_profile_card(
    profile: dict[str, Any],
    similarity: float | None = None,
    runner_up_similarity: float | None = None,
    margin: float | None = None,
) -> np.ndarray:
    card_width = 900
    card_height = 520
    card = np.full((card_height, card_width, 3), 245, dtype=np.uint8)
    cv2.rectangle(card, (0, 0), (card_width - 1, card_height - 1), (50, 50, 50), 2)
    cv2.rectangle(card, (0, 0), (card_width, 70), (32, 76, 120), -1)
    cv2.putText(card, "Recognized Profile", (28, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (255, 255, 255), 2)

    headshot = _load_headshot(profile.get("profile_image_path"))
    card[110:350, 34:274] = headshot
    cv2.rectangle(card, (34, 110), (274, 350), (60, 60, 60), 2)

    lines = [
        ("Name", str(profile.get("name", ""))),
        ("Age", str(profile.get("age", ""))),
        ("Nationality", str(profile.get("nationality", ""))),
        ("Career", str(profile.get("career", ""))),
    ]
    if similarity is not None:
        lines.append(("Similarity", f"{similarity:.3f}"))
    if runner_up_similarity is not None:
        lines.append(("Runner-up", f"{runner_up_similarity:.3f}"))
    if margin is not None:
        lines.append(("Margin", f"{margin:.3f}"))

    x = 325
    y = 112
    max_text_width = card_width - x - 40
    for label, value in lines:
        cv2.putText(card, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (90, 90, 90), 1, cv2.LINE_AA)
        _put_fitted_text(
            card,
            value,
            (x, y + 32),
            max_text_width=max_text_width,
            initial_scale=0.82,
            color=(20, 20, 20),
            thickness=2,
        )
        y += 62
    return card


def _put_fitted_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    max_text_width: int,
    initial_scale: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    scale = initial_scale
    while scale > 0.42:
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        if text_size[0] <= max_text_width:
            break
        scale -= 0.05
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _load_headshot(path_value: Any) -> np.ndarray:
    placeholder = np.full((240, 240, 3), 220, dtype=np.uint8)
    cv2.circle(placeholder, (120, 92), 42, (150, 150, 150), -1)
    cv2.ellipse(placeholder, (120, 190), (78, 58), 0, 180, 360, (150, 150, 150), -1)

    if not path_value:
        return placeholder
    image = cv2.imread(str(path_value))
    if image is None:
        return placeholder

    height, width = image.shape[:2]
    side = min(height, width)
    y1 = max(0, (height - side) // 2)
    x1 = max(0, (width - side) // 2)
    square = image[y1:y1 + side, x1:x1 + side]
    return cv2.resize(square, (240, 240))
