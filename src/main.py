from __future__ import annotations

import argparse

import cv2

from database import ProfileDatabase
from detect import FaceDetectionError, FaceDetector
from enroll import EnrollmentManager, prompt_for_profile
from recognize import BaseFaceEmbedder, EmbeddingModelError, FaceRecognizer, create_embedder
from utils import (
    DEFAULT_DEEPFACE_MODEL_NAME,
    DEFAULT_EMBEDDING_BACKEND,
    DEFAULT_MATCH_MARGIN,
    DEFAULT_OPENCV_FACE_MODEL_PATH,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K_SCORES,
    build_profile_card,
    draw_face_label,
    ensure_directories,
    open_camera,
    read_camera_frame,
    rotate_frame,
    save_profile_image,
    slugify,
    resolve_project_path,
)


def build_services(args: argparse.Namespace) -> tuple[
    ProfileDatabase, FaceDetector, BaseFaceEmbedder, FaceRecognizer, EnrollmentManager
]:
    ensure_directories()
    database = ProfileDatabase()
    database.initialize()
    detector = FaceDetector()
    embedder = create_embedder(
        backend=args.backend,
        deepface_model=args.deepface_model,
        opencv_model_path=args.opencv_model,
    )
    recognizer = FaceRecognizer(
        database=database,
        detector=detector,
        embedder=embedder,
        threshold=args.threshold,
        match_margin=args.match_margin,
        top_k_scores=args.top_k_scores,
    )
    enroller = EnrollmentManager(database=database, detector=detector, embedder=embedder)
    return database, detector, embedder, recognizer, enroller


def load_image(image_path: str) -> cv2.typing.MatLike:
    image = cv2.imread(str(resolve_project_path(image_path)))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return image


def handle_enroll_image(args: argparse.Namespace) -> None:
    _, _, _, _, enroller = build_services(args)
    profile = prompt_for_profile()
    images = [load_image(image_path) for image_path in args.images]
    person_id = enroller.enroll_from_images(profile=profile, images=images)
    print(f"Enrollment complete. person_id={person_id}")


def handle_enroll_webcam(args: argparse.Namespace) -> None:
    _, _, _, _, enroller = build_services(args)
    profile = prompt_for_profile()
    person_id = enroller.enroll_from_webcam(
        profile=profile,
        sample_count=args.samples,
        camera_index=args.camera_index,
        camera_backend=args.camera_backend,
        rotation=args.rotate,
        mirror=not args.no_mirror,
    )
    print(f"Enrollment complete. person_id={person_id}")


def handle_recognize_image(args: argparse.Namespace) -> None:
    _, _, _, recognizer, _ = build_services(args)
    image = load_image(args.image)
    output = recognizer.recognize_image(image)
    lines = [f"Similarity: {output.match.similarity:.3f}"]
    if output.match.profile:
        lines.extend(
            [
                f"Age: {output.match.profile['age']}",
                f"Nationality: {output.match.profile['nationality']}",
                f"Career: {output.match.profile['career']}",
            ]
        )
    annotated = draw_face_label(image.copy(), output.bbox, output.match.name, lines)
    cv2.imshow("Recognition Result", annotated)
    if output.match.profile:
        cv2.imshow(
            "Recognized Profile",
            build_profile_card(
                output.match.profile,
                output.match.similarity,
            ),
        )
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print(output.match)


def handle_webcam_demo(args: argparse.Namespace) -> None:
    database, _, _, recognizer, enroller = build_services(args)
    capture = open_camera(args.camera_index, args.camera_backend)
    if not capture.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Webcam demo started.")
    print("Keys: q=quit, a=add currently unknown person")

    last_unknown_frame = None
    visible_profile_id = None
    try:
        while True:
            frame = read_camera_frame(capture)
            frame = rotate_frame(frame, args.rotate)
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            display = frame.copy()
            try:
                output = recognizer.recognize_image(frame)
                detail_lines = [
                    f"Similarity: {output.match.similarity:.3f}",
                ]
                if output.match.profile:
                    cv2.imshow(
                        "Recognized Profile",
                        build_profile_card(
                            output.match.profile,
                            output.match.similarity,
                        ),
                    )
                    visible_profile_id = output.match.person_id
                else:
                    detail_lines.append("Press A to enroll")
                    last_unknown_frame = frame.copy()
                    if visible_profile_id is not None:
                        try:
                            cv2.destroyWindow("Recognized Profile")
                        except cv2.error:
                            pass
                        visible_profile_id = None
                display = draw_face_label(display, output.bbox, output.match.name, detail_lines)
            except FaceDetectionError as error:
                if visible_profile_id is not None:
                    try:
                        cv2.destroyWindow("Recognized Profile")
                    except cv2.error:
                        pass
                    visible_profile_id = None
                cv2.putText(
                    display,
                    str(error),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Face Recognition Demo", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("a") and last_unknown_frame is not None:
                print("Unknown face detected. Enter profile details.")
                profile = prompt_for_profile()
                capture.release()
                cv2.destroyWindow("Face Recognition Demo")
                person_id = enroller.enroll_from_webcam(
                    profile=profile,
                    sample_count=args.samples,
                    camera_index=args.camera_index,
                    camera_backend=args.camera_backend,
                    rotation=args.rotate,
                    mirror=not args.no_mirror,
                )
                capture = open_camera(args.camera_index, args.camera_backend)
                if not capture.isOpened():
                    raise RuntimeError("Could not reopen webcam after enrollment.")
                print(f"Added new person with person_id={person_id}")
                if database.get_person_by_id(person_id):
                    last_unknown_frame = None
    finally:
        capture.release()
        cv2.destroyAllWindows()


def handle_init_db(args: argparse.Namespace) -> None:
    ensure_directories()
    database = ProfileDatabase()
    database.initialize()
    print(f"Database ready at: {database.db_path}")


def handle_list_people(args: argparse.Namespace) -> None:
    database = ProfileDatabase()
    database.initialize()
    people = database.get_all_people()
    if not people:
        print("No people enrolled yet.")
        return
    for person in people:
        print(
            f"{person['person_id']}: {person['name']} | "
            f"age={person['age']} | nationality={person['nationality']} | "
            f"career={person['career']} | samples={person['sample_count']}"
        )


def handle_delete_person(args: argparse.Namespace) -> None:
    database = ProfileDatabase()
    database.initialize()
    deleted_count = database.delete_person(args.person_id)
    if deleted_count == 0:
        print(f"No person found with person_id={args.person_id}.")
        return
    print(f"Deleted person_id={args.person_id} and related database records.")


def handle_set_headshot(args: argparse.Namespace) -> None:
    database = ProfileDatabase()
    database.initialize()
    person = database.get_person_by_id(args.person_id)
    if person is None:
        raise ValueError(f"No person found with person_id={args.person_id}.")

    image = load_image(args.image)
    prefix = f"{args.person_id}_{slugify(person['name'])}"
    profile_image_path = save_profile_image(image, prefix)
    database.update_profile_image(args.person_id, str(profile_image_path))
    print(f"Updated headshot for {person['name']}: {profile_image_path}")


def handle_camera_test(args: argparse.Namespace) -> None:
    capture = open_camera(args.camera_index, args.camera_backend)
    if not capture.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Camera test started. Press Q to quit.")
    try:
        while True:
            frame = read_camera_frame(capture)
            frame = rotate_frame(frame, args.rotate)
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)
            cv2.putText(
                frame,
                f"index={args.camera_index} backend={args.camera_backend} | Q to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Camera Test", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        try:
            cv2.destroyWindow("Camera Test")
        except cv2.error:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local face recognition demo")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Cosine similarity threshold for known/unknown classification.",
    )
    parser.add_argument(
        "--match-margin",
        type=float,
        default=DEFAULT_MATCH_MARGIN,
        help="Required score gap between best and second-best person.",
    )
    parser.add_argument(
        "--top-k-scores",
        type=int,
        default=DEFAULT_TOP_K_SCORES,
        help="Number of best sample scores to average per person.",
    )
    parser.add_argument(
        "--backend",
        choices=("opencv", "deepface"),
        default=DEFAULT_EMBEDDING_BACKEND,
        help="Embedding backend to use.",
    )
    parser.add_argument(
        "--opencv-model",
        default=str(DEFAULT_OPENCV_FACE_MODEL_PATH),
        help="Path to OpenCV SFace ONNX model.",
    )
    parser.add_argument(
        "--deepface-model",
        default=DEFAULT_DEEPFACE_MODEL_NAME,
        help="DeepFace model name when using --backend deepface.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_db = subparsers.add_parser("init-db", help="Create the local SQLite database and folders.")
    init_db.set_defaults(func=handle_init_db)

    list_people = subparsers.add_parser("list-people", help="List enrolled people from the local database.")
    list_people.set_defaults(func=handle_list_people)

    delete_person = subparsers.add_parser("delete-person", help="Delete one enrolled person from the database.")
    delete_person.add_argument("person_id", type=int)
    delete_person.set_defaults(func=handle_delete_person)

    set_headshot = subparsers.add_parser("set-headshot", help="Set a nicer profile headshot for one person.")
    set_headshot.add_argument("person_id", type=int)
    set_headshot.add_argument("--image", required=True, help="Path to the headshot image.")
    set_headshot.set_defaults(func=handle_set_headshot)

    camera_test = subparsers.add_parser("camera-test", help="Preview one camera without recognition.")
    camera_test.add_argument("--camera-index", type=int, default=0)
    camera_test.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"), default="auto")
    camera_test.add_argument("--rotate", choices=("none", "cw", "ccw", "180"), default="none")
    camera_test.add_argument("--no-mirror", action="store_true", help="Show raw webcam orientation.")
    camera_test.set_defaults(func=handle_camera_test)

    enroll_image = subparsers.add_parser("enroll-image", help="Enroll a new person from a single image.")
    enroll_image.add_argument("--images", nargs="+", required=True, help="One or more image paths.")
    enroll_image.set_defaults(func=handle_enroll_image)

    enroll_webcam = subparsers.add_parser("enroll-webcam", help="Enroll a new person from webcam samples.")
    enroll_webcam.add_argument("--camera-index", type=int, default=0)
    enroll_webcam.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"), default="auto")
    enroll_webcam.add_argument("--rotate", choices=("none", "cw", "ccw", "180"), default="none")
    enroll_webcam.add_argument("--samples", type=int, default=5)
    enroll_webcam.add_argument("--no-mirror", action="store_true", help="Show raw webcam orientation.")
    enroll_webcam.set_defaults(func=handle_enroll_webcam)

    recognize_image = subparsers.add_parser("recognize-image", help="Recognize one face from an image.")
    recognize_image.add_argument("--image", required=True, help="Path to image file.")
    recognize_image.set_defaults(func=handle_recognize_image)

    webcam_demo = subparsers.add_parser("webcam-demo", help="Run live webcam recognition demo.")
    webcam_demo.add_argument("--camera-index", type=int, default=0)
    webcam_demo.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"), default="auto")
    webcam_demo.add_argument("--rotate", choices=("none", "cw", "ccw", "180"), default="none")
    webcam_demo.add_argument("--samples", type=int, default=5, help="Samples to collect during live enrollment.")
    webcam_demo.add_argument("--no-mirror", action="store_true", help="Show raw webcam orientation.")
    webcam_demo.set_defaults(func=handle_webcam_demo)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FaceDetectionError, FileNotFoundError, RuntimeError, ValueError, EmbeddingModelError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
