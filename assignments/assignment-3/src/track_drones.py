from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter

ROOT = Path("assignments/assignment-3")
FRAMES_DIR = ROOT / "frames"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DETECTIONS_PATH = RESULTS_DIR / "detections.parquet"
df = pd.read_parquet(DETECTIONS_PATH)

# strongest detection per frame
df = df.sort_values(["video_id", "frame_index", "confidence_score"], ascending=[True, True, False])
df = df.groupby(["video_id", "frame_index"], as_index=False).first()

def bbox_center(box):
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)

def make_kf():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 1.0
    kf.F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1,  0],
        [0, 0, 0,  1],
    ], dtype=float)
    kf.H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)
    kf.P *= 200.0
    kf.R = np.array([[25.0, 0.0], [0.0, 25.0]])
    kf.Q = np.eye(4) * 0.5
    return kf

MAX_GATING_DISTANCE = 120
MAX_MISSED_FRAMES = 8

for video_id in ["video2"]:   # debug only short video first
    video_df = df[df["video_id"] == video_id].copy()
    frame_dir = FRAMES_DIR / video_id
    frame_paths = sorted(frame_dir.glob("*.jpg"))
    if not frame_paths:
        continue

    first_img = cv2.imread(str(frame_paths[0]))
    h, w = first_img.shape[:2]

    out_path = RESULTS_DIR / f"{video_id}_tracked_debug.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (w, h)
    )

    kf = None
    trajectory = []
    missed_frames = 0

    det_map = {int(r.frame_index): r for r in video_df.itertuples(index=False)}

    for frame_path in frame_paths:
        frame_name = frame_path.stem
        frame_index = int(frame_name.split("_")[-1])
        img = cv2.imread(str(frame_path))

        accepted_detection = False
        row = None
        box = None

        if kf is not None:
            kf.predict()

        if frame_index in det_map:
            row = det_map[frame_index]
            box = row.bounding_box
            center = bbox_center(box)

            if kf is None:
                kf = make_kf()
                kf.x = np.array([center[0], center[1], 0.0, 0.0], dtype=float)
                accepted_detection = True
                missed_frames = 0
            else:
                pred_center = np.array([kf.x[0], kf.x[1]])
                dist = np.linalg.norm(center - pred_center)

                if dist < MAX_GATING_DISTANCE:
                    kf.update(center)
                    accepted_detection = True
                    missed_frames = 0
                else:
                    missed_frames += 1
        else:
            if kf is not None:
                missed_frames += 1

        if kf is not None and missed_frames > MAX_MISSED_FRAMES:
            kf = None
            trajectory = []
            missed_frames = 0

        if accepted_detection and box is not None:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{row.class_label}: {row.confidence_score:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if kf is not None:
            cx, cy = int(kf.x[0]), int(kf.x[1])
            trajectory.append((cx, cy))
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)

            if len(trajectory) > 1:
                pts = np.array(trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [pts], False, (255, 0, 0), 2)

        writer.write(img)

    writer.release()
    print(f"Saved tracked video: {out_path}")