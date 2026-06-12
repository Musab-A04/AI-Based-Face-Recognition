# Face Recognition Project

Local Python face-recognition demo for a student AI project.

The system follows this pipeline:

`camera/image input -> face detection -> face cropping -> pretrained CNN embedding extraction -> similarity comparison -> SQLite profile lookup -> on-screen display`

It recognizes a face first, then retrieves stored profile fields from the local database. It does not predict age, nationality, or career from facial features.

## Features

- Detect one face from a webcam frame or image
- Extract face embeddings with a pretrained OpenCV SFace model by default
- Optionally use DeepFace as an alternate embedding backend
- Compare against stored embeddings with cosine similarity
- Return `Unknown` when the similarity score is below a configurable threshold
- Store profile data locally in SQLite
- Store multiple face embeddings per person while keeping one retained headshot image
- Enroll a new person from image files or webcam samples
- Add an unknown person live during the webcam demo
- Display name, age, nationality, career, and similarity score
- Show a separate profile card window with a stored headshot when a person is recognized

## Project structure

```text
FaceRecognitionProject/
  app.py
  app.pyw
  requirements.txt
  README.md
  database/
  images/
  embeddings/
  models/
  src/
    main.py
    detect.py
    recognize.py
    enroll.py
    database.py
    utils.py
    ui.py
    check_setup.py
```

## Modules

- `app.py`: GUI launcher for normal use
- `app.pyw`: Windows no-console GUI launcher when Python file associations support it
- `src/ui.py`: Tkinter desktop interface
- `src/database.py`: SQLite schema and CRUD functions
- `src/detect.py`: face detection, face cropping, detection errors
- `src/recognize.py`: pretrained embedding extraction and matching
- `src/enroll.py`: enrollment logic and webcam sample capture
- `src/utils.py`: paths, constants, similarity helpers, drawing helpers
- `src/main.py`: CLI entrypoint and live demo
- `src/check_setup.py`: environment and model-file checker

## Database design

The local SQLite database contains:

### `people`

- `person_id`
- `name`
- `age`
- `nationality`
- `career`
- `profile_image_path`
- `created_at`

### `face_samples`

- `sample_id`
- `person_id`
- `image_path`
- `embedding_path`
- `created_at`

This allows multiple embeddings to be stored for the same person while keeping one retained headshot image for display.

## Setup

## 1. Python version

The current workspace has Python `3.13`.

The default backend uses OpenCV SFace, which is more practical for this machine. DeepFace is optional; if you enable it, Python `3.11` or `3.12` is usually easier than Python `3.13`.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Main dependencies:

- `opencv-contrib-python`
- `numpy`

## 3. Add the pretrained SFace model

For the default `opencv` backend, place these pretrained model files in `models/`:

```text
models/face_detection_yunet_2023mar.onnx
models/face_recognition_sface_2021dec.onnx
```

Keep this file local so the demo can run offline after setup.

## 4. Check setup

```bash
python src/check_setup.py
```

This prints the Python version, OpenCV version, whether `FaceRecognizerSF` is available, and whether the SFace model file exists.

## 5. Run the GUI

The GUI is the preferred classroom demo path:

```bash
python app.py
```

On Windows, `app.pyw` can be used as a no-console launcher if `.pyw` files are associated with Python:

```bash
app.pyw
```

The GUI automatically creates the database and missing tables on startup. You do not need to run `init-db` for normal use.

The GUI is organized into tabs:

- `Live Demo`: main presentation screen with embedded camera preview, recognition status, similarity, saved headshot, and structured profile fields
- `Enroll`: enrollment entry point and registered people summary
- `People`: registered people table
- `Status`: database/status summary and refresh button
- `Advanced`: camera and matching settings
- `Exit`: clean app shutdown

Live Demo action:

- `Start Recognition / Demo`: starts live recognition and updates the GUI with stored profile data

Enrollment, people listing, status refresh, settings, and exit are available from the top tabs.

Camera settings are available in the GUI:

- camera index
- backend
- rotation
- sample count
- threshold
- margin
- mirror preview

## CLI fallback

The old command-line interface is still available for testing and fallback use.

Initialize the database manually:

```bash
python src/main.py init-db
```

Inspect enrolled profiles:

```bash
python src/main.py list-people
```

Delete an old or bad enrollment:

```bash
python src/main.py delete-person PERSON_ID
```

Replace the stored headshot with a nicer image:

```bash
python src/main.py set-headshot PERSON_ID --image images/professional_headshot.jpg
```

## How recognition works

1. Detect a face with OpenCV YuNet when the model file is available.
2. Align the face with SFace landmark alignment.
3. Generate a pretrained embedding using OpenCV SFace by default.
4. Compare the query embedding with stored embeddings using cosine similarity.
5. If the best score is above the threshold, mark as known.
6. Otherwise, return `Unknown`.
7. If known, fetch profile data from SQLite and display it.

## Threshold logic

The matching threshold is defined in [utils.py](./src/utils.py) as:

- `DEFAULT_SIMILARITY_THRESHOLD = 0.62`

You can also override it from the command line:

```bash
python src/main.py --threshold 0.50 recognize-image --image images/test.jpg
```

If the best similarity score is below the threshold, the system returns `Unknown`.

You can choose the backend from the command line:

```bash
python src/main.py --backend opencv recognize-image --image images/test.jpg
python src/main.py --backend deepface recognize-image --image images/test.jpg
```

## Enrollment

Enrollment stores profile data only after the person is intentionally added.

### Enroll from image files

```bash
python src/main.py enroll-image --images images/person1_1.jpg images/person1_2.jpg
```

The program will ask for:

- `Name`
- `Age`
- `Nationality`
- `Career`

Then it will:

1. detect the face in each image
2. generate embeddings
3. save cropped face images
4. save embedding `.npy` files
5. write the profile and sample records to SQLite
6. keep only the first captured face image on disk for display, while storing all embeddings

### Enroll from webcam

```bash
python src/main.py enroll-webcam --samples 5
```

Controls:

- `SPACE`: capture a sample
- `Q`: cancel

Recommended: collect 3 to 5 clean samples with slightly different angles and expressions.

## Recognition

### Recognize from an image

```bash
python src/main.py recognize-image --image images/test.jpg
```

The app shows the detected face with:

- name or `Unknown`
- similarity score
- age
- nationality
- career

If the person is known, a second `Recognized Profile` window opens with the stored headshot and profile fields from SQLite.

## Live demo mode

Run the webcam demo:

```bash
python src/main.py webcam-demo --samples 5
```

Controls:

- `Q`: quit
- `A`: enroll the currently unknown person

Live demo flow:

1. start webcam recognition
2. if the face is known, show stored profile info
3. if the face is unknown, show `Unknown`
4. press `A` to add that person
5. enter the profile fields
6. collect webcam samples
7. return to the demo and recognize the newly enrolled person
8. show a separate profile card window for known people

## Images and data needed

You need your own face images for testing and enrollment.

Recommended dataset for a demo:

- one folder of sample images for known people
- at least 2 to 5 images per enrolled person
- one separate image or live webcam view for testing recognition

Good sample images should have:

- clear face visibility
- reasonable lighting
- limited blur
- limited occlusion
- mostly frontal pose for best results

## Error handling included

The code handles:

- no face detected
- multiple faces detected
- invalid image path
- webcam open/read failure
- invalid enrollment input
- missing embeddings in the database
- no match found
- missing embedding backend or model file

## Important limitations

- Age, nationality, and career are retrieved from the local database after identity recognition. They are not predicted from the face.
- Accuracy depends heavily on enrollment image quality.
- Lighting, pose, blur, and occlusion affect performance.
- The project uses pretrained face embeddings instead of training a CNN from scratch.
- The default OpenCV backend requires local YuNet and SFace ONNX models in `models/`.
- This is not a liveness detector. A photo shown to the webcam may still be processed as a face.
- The optional DeepFace backend may require model-weight download before fully offline use.
- Recognition quality depends on threshold tuning and the number of stored samples.

## Suggested improvements

- switch to a stronger detector such as RetinaFace or MTCNN
- add a small Streamlit UI
- keep one retained enrollment photo per person and store the rest only as embeddings
- add a person list view from SQLite
- support deleting or updating profiles
- average embeddings per person for faster matching

## Current status

This repository now includes the initial end-to-end scaffold and core recognition pipeline. The code structure is ready for dependency installation, local testing, and class-demo refinement.
