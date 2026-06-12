from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from database import ProfileDatabase
from detect import FaceDetectionError, FaceDetector
from recognize import BaseFaceEmbedder
from utils import (
    DEFAULT_ENROLLMENT_SAMPLES,
    open_camera,
    read_camera_frame,
    rotate_frame,
    save_embedding,
    save_profile_image,
    slugify,
    timestamp_string,
)


@dataclass(slots=True)
class EnrollmentProfile:
    name: str
    age: int
    nationality: str
    career: str
    role: str = "visitor"


class EnrollmentManager:
    def __init__(
        self,
        database: ProfileDatabase,
        detector: FaceDetector,
        embedder: BaseFaceEmbedder,
    ) -> None:
        self.database = database
        self.detector = detector
        self.embedder = embedder

    def enroll_from_images(
        self,
        profile: EnrollmentProfile,
        images: list[np.ndarray],
    ) -> int:
        if not images:
            raise ValueError("At least one face image is required for enrollment.")

        prepared_samples: list[tuple[np.ndarray, np.ndarray]] = []
        for image in images:
            detection = self.detector.detect_single_face(image)
            embedding = self.embedder.get_embedding_from_detection(image, detection)
            prepared_samples.append((detection.face_image, embedding))

        if not prepared_samples:
            raise FaceDetectionError("Enrollment failed because no valid face samples were collected.")

        created_at = timestamp_string()
        person_id = self.database.insert_person(
            name=profile.name,
            age=profile.age,
            nationality=profile.nationality,
            career=profile.career,
            created_at=created_at,
            role=profile.role,
        )

        prefix = f"{person_id}_{slugify(profile.name)}"
        profile_image_path = save_profile_image(prepared_samples[0][0], prefix)
        self.database.update_profile_image(person_id, str(profile_image_path), source="first_sample")

        for _, embedding in prepared_samples:
            embedding_path = save_embedding(embedding, prefix)
            self.database.add_face_sample(
                person_id=person_id,
                image_path=str(profile_image_path),
                embedding_path=str(embedding_path),
                created_at=timestamp_string(),
            )

        return person_id

    def enroll_from_webcam(
        self,
        profile: EnrollmentProfile,
        sample_count: int = DEFAULT_ENROLLMENT_SAMPLES,
        camera_index: int = 0,
        camera_backend: str = "auto",
        rotation: str = "none",
        mirror: bool = True,
    ) -> int:
        capture = open_camera(camera_index, camera_backend)
        if not capture.isOpened():
            raise RuntimeError("Could not open webcam.")

        images: list[np.ndarray] = []
        try:
            # Give the camera a short warm-up window before the first capture.
            # Some webcams return empty frames for the first few reads after opening.
            for _ in range(15):
                ok, warmup_frame = capture.read()
                if ok and warmup_frame is not None:
                    break
            while len(images) < sample_count:
                frame = read_camera_frame(capture, attempts=60)
                frame = rotate_frame(frame, rotation)
                if mirror:
                    frame = cv2.flip(frame, 1)

                display = frame.copy()
                faces = self.detector.detect_faces(frame)
                for x, y, w, h in faces:
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    display,
                    f"Samples: {len(images)}/{sample_count} | Press SPACE to capture | Q to quit",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Enrollment", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    raise RuntimeError("Enrollment canceled by user.")

                if key == ord(" "):
                    try:
                        self.detector.detect_single_face(frame)
                        images.append(frame.copy())
                    except FaceDetectionError as error:
                        print(f"Capture skipped: {error}")
        finally:
            capture.release()
            cv2.destroyWindow("Enrollment")

        return self.enroll_from_images(profile=profile, images=images)


def prompt_for_profile() -> EnrollmentProfile:
    name = input("Name: ").strip()
    age_text = input("Age: ").strip()
    nationality = input("Nationality: ").strip()
    career = input("Career: ").strip()
    role = input("Role [visitor]: ").strip() or "visitor"

    if not name or not nationality or not career:
        raise ValueError("Name, nationality, and career are required.")
    if not age_text.isdigit():
        raise ValueError("Age must be a valid integer.")
    return EnrollmentProfile(
        name=name,
        age=int(age_text),
        nationality=nationality,
        career=career,
        role=role,
    )
