# Assignment 3 — UAV Drone Detection and Tracking

**Student:** Pablo Monteros  
**Course:** Computer Vision  
**Assignment:** UAV Drone Detection and Tracking

## Overview

This project detects and tracks drones in video using a deep learning detector and a Kalman filter.  
The pipeline processes video frames, saves frames with drone detections, stores detections in Parquet format, and generates tracked output videos with bounding boxes and 2D trajectories.

## Input Videos

The pipeline was tested on the two required drone videos provided in the assignment.

## Task 1: Drone Detection

### Detector choice
I used a zero-shot object detection model based on **Grounding DINO** with the prompt:

- `drone`
- `quadcopter`
- `uav`

This was chosen as a fast baseline to detect drones without additional training.

### Detection configuration
Key settings used:
- confidence filtering
- text prompt filtering for drone-related labels
- bounding box size filtering to reject implausible detections
- frame-by-frame processing over extracted video frames

### Output
Frames with at least one accepted detection were saved in:

- `assignments/assignment-3/detections/video1/`
- `assignments/assignment-3/detections/video2/`

Structured detections were saved in:

- `assignments/assignment-3/results/detections.parquet`

The Parquet file includes:
- `video_id`
- `frame_file`
- `frame_index`
- `timestamp_sec`
- `class_label`
- `bounding_box`
- `confidence_score`

## Task 2: Kalman Filter Tracking

### State design
I used a Kalman filter with state:

- `(x, y, vx, vy)`

where:
- `x, y` = 2D image-plane center of the detected drone bounding box
- `vx, vy` = estimated velocity in pixels per frame

### Measurement model
The measurement vector contains:

- `(x, y)`

from the center of the detector bounding box.

### Tracking logic
For each frame:
1. predict the next state
2. update with the current detection if the detection is plausible
3. continue predicting for a short number of missed frames
4. reset the track if too many consecutive frames are missed

### Stabilization improvements
To reduce false positives and trajectory jumps, I added:
- stricter confidence thresholds
- bounding box size filtering
- distance gating between the Kalman prediction and new detections
- reset logic after too many missed frames

### Output videos
Tracked videos were generated as:

- `assignments/assignment-3/results/video1_tracked.mp4`
- `assignments/assignment-3/results/video2_tracked.mp4`

Each output video overlays:
- detector bounding boxes
- estimated 2D trajectory as a polyline

## Failure Cases

A main failure case occurs when the drone becomes very small, leaves the frame, or is confused with visually similar background objects such as birds or tree/horizon structures.

In these situations:
- the detector may produce false positives
- the tracker may temporarily lose the drone
- the trajectory may drift if bad detections are accepted

The added gating and reset logic reduced these errors substantially, but some difficult scenes still remain challenging.

## How to Run

### 1. Extract frames
Frames were extracted from the input videos at 5 fps.

### 2. Run detection
Example command:

```bash
docker compose run --rm torch.dev.gpu /bin/sh -lc ".venv/bin/python assignments/assignment-3/src/detect_drones.py"


YouTube Links

Video 1: https://youtu.be/MlQhMC3L2Po

Video 2: https://youtu.be/jJS7w30CsB0



