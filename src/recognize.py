from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from database import ProfileDatabase
from detect import DetectionResult, FaceDetector
from utils import (
    DEFAULT_DEEPFACE_MODEL_NAME,
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_MATCH_MARGIN,
    DEFAULT_OPENCV_FACE_MODEL_PATH,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K_SCORES,
    MatchResult,
    cosine_similarity,
    load_embedding,
)

try:
    from deepface import DeepFace
except ImportError:  # pragma: no cover - handled at runtime
    DeepFace = None


class EmbeddingModelError(RuntimeError):
    pass


@dataclass(slots=True)
class RecognitionOutput:
    bbox: tuple[int, int, int, int]
    match: MatchResult


class BaseFaceEmbedder:
    backend_name = "base"

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_embedding_from_detection(
        self,
        image: np.ndarray,
        detection: DetectionResult,
    ) -> np.ndarray:
        return self.get_embedding(detection.face_image)


class DeepFaceEmbedder(BaseFaceEmbedder):
    backend_name = "deepface"

    def __init__(self, model_name: str = DEFAULT_DEEPFACE_MODEL_NAME) -> None:
        self.model_name = model_name
        if DeepFace is None:
            raise EmbeddingModelError(
                "DeepFace is not installed. Install requirements.txt before running recognition."
            )

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        rgb_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        representations = DeepFace.represent(
            img_path=rgb_face,
            model_name=self.model_name,
            detector_backend="skip",
            enforce_detection=False,
        )
        if not representations:
            raise EmbeddingModelError("Embedding extraction returned no data.")
        vector = np.asarray(representations[0]["embedding"], dtype=np.float32)
        if vector.size == 0:
            raise EmbeddingModelError("Embedding vector is empty.")
        return vector


class OpenCVSFaceEmbedder(BaseFaceEmbedder):
    backend_name = "opencv"

    def __init__(self, model_path: str | Path = DEFAULT_OPENCV_FACE_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise EmbeddingModelError(
                "OpenCV SFace model is missing. Put face_recognition_sface_2021dec.onnx "
                f"in {self.model_path.parent} or run with --backend deepface."
            )
        if not hasattr(cv2, "FaceRecognizerSF_create"):
            raise EmbeddingModelError(
                "Your OpenCV build does not include FaceRecognizerSF. "
                "Install opencv-contrib-python or use --backend deepface."
            )
        self.model = cv2.FaceRecognizerSF_create(str(self.model_path), "")

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        resized = cv2.resize(face_image, (112, 112))
        feature = self.model.feature(resized)
        vector = np.asarray(feature, dtype=np.float32).reshape(-1)
        if vector.size == 0:
            raise EmbeddingModelError("OpenCV SFace returned an empty embedding.")
        return vector

    def get_embedding_from_detection(
        self,
        image: np.ndarray,
        detection: DetectionResult,
    ) -> np.ndarray:
        if detection.face_data is not None:
            aligned_face = self.model.alignCrop(image, detection.face_data)
            return self.get_embedding(aligned_face)
        return self.get_embedding(detection.face_image)


def create_embedder(
    backend: str = DEFAULT_EMBEDDING_BACKEND,
    deepface_model: str = DEFAULT_DEEPFACE_MODEL_NAME,
    opencv_model_path: str | Path = DEFAULT_OPENCV_FACE_MODEL_PATH,
) -> BaseFaceEmbedder:
    selected_backend = backend.strip().lower()
    if selected_backend == "opencv":
        return OpenCVSFaceEmbedder(model_path=opencv_model_path)
    if selected_backend == "deepface":
        return DeepFaceEmbedder(model_name=deepface_model)
    raise EmbeddingModelError("Unsupported backend. Use 'opencv' or 'deepface'.")


class FaceRecognizer:
    def __init__(
        self,
        database: ProfileDatabase,
        detector: FaceDetector,
        embedder: BaseFaceEmbedder,
        threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        match_margin: float = DEFAULT_MATCH_MARGIN,
        top_k_scores: int = DEFAULT_TOP_K_SCORES,
    ) -> None:
        self.database = database
        self.detector = detector
        self.embedder = embedder
        self.threshold = threshold
        self.match_margin = match_margin
        self.top_k_scores = top_k_scores

    def recognize_image(self, image: np.ndarray) -> RecognitionOutput:
        detection = self.detector.detect_single_face(image)
        query_embedding = self.embedder.get_embedding_from_detection(image, detection)
        match = self.find_best_match(query_embedding)
        return RecognitionOutput(bbox=detection.bbox, match=match)

    def find_best_match(self, query_embedding: np.ndarray) -> MatchResult:
        samples = self.database.get_all_face_samples()
        if not samples:
            return MatchResult(
                person_id=None,
                name="Unknown",
                similarity=0.0,
                is_known=False,
                profile=None,
            )

        scores_by_person: dict[int, dict[str, Any]] = {}
        for sample in samples:
            embedding_path = Path(sample["embedding_path"])
            if not embedding_path.exists():
                continue
            stored_embedding = load_embedding(embedding_path)
            score = cosine_similarity(query_embedding, stored_embedding)
            person_id = int(sample["person_id"])
            if person_id not in scores_by_person:
                scores_by_person[person_id] = {"sample": sample, "scores": []}
            scores_by_person[person_id]["scores"].append(score)

        if not scores_by_person:
            return MatchResult(
                person_id=None,
                name="Unknown",
                similarity=0.0,
                is_known=False,
                profile=None,
            )

        ranked_people = []
        for person_id, data in scores_by_person.items():
            person_scores = sorted(data["scores"], reverse=True)
            top_scores = person_scores[: max(1, self.top_k_scores)]
            ranked_people.append(
                {
                    "person_id": person_id,
                    "sample": data["sample"],
                    "score": float(np.mean(top_scores)),
                    "best_sample_score": float(person_scores[0]),
                }
            )
        ranked_people.sort(key=lambda item: item["score"], reverse=True)

        if not ranked_people:
            return MatchResult(
                person_id=None,
                name="Unknown",
                similarity=0.0,
                is_known=False,
                profile=None,
            )

        best_match = ranked_people[0]
        runner_up_score = ranked_people[1]["score"] if len(ranked_people) > 1 else 0.0
        margin = best_match["score"] - runner_up_score

        if best_match["score"] < self.threshold or margin < self.match_margin:
            return MatchResult(
                person_id=None,
                name="Unknown",
                similarity=max(best_match["score"], 0.0),
                is_known=False,
                profile=None,
                runner_up_similarity=max(runner_up_score, 0.0),
                margin=max(margin, 0.0),
            )

        sample = best_match["sample"]
        profile = self.database.get_person_by_id(int(sample["person_id"]))
        return MatchResult(
            person_id=int(sample["person_id"]),
            name=str(sample["name"]),
            similarity=best_match["score"],
            is_known=True,
            profile=profile,
            runner_up_similarity=max(runner_up_score, 0.0),
            margin=max(margin, 0.0),
        )
