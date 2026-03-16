from pathlib import Path
import re
import cv2
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

ROOT = Path("assignments/assignment-3")
FRAMES_DIR = ROOT / "frames"
OUT_DIR = ROOT / "detections"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
TEXT_PROMPT = "drone. quadcopter. uav."
BOX_THRESHOLD = 0.40
TEXT_THRESHOLD = 0.30
EXTRACTED_FPS = 5.0

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
model.eval()

rows = []

def frame_idx_from_name(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else -1

for video_dir in sorted(FRAMES_DIR.iterdir()):
    if not video_dir.is_dir():
        continue

    video_id = video_dir.name
    out_video_dir = OUT_DIR / video_id
    out_video_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(video_dir.glob("*.jpg"))
    print(f"{video_id}: scanning {len(frame_paths)} frames")

    for frame_path in frame_paths:
        image = Image.open(frame_path).convert("RGB")

        inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=target_sizes,
        )[0]


        boxes = results["boxes"].detach().cpu().tolist()
        scores = results["scores"].detach().cpu().tolist()
        labels = results["labels"]

        if len(boxes) == 0:
            continue

        frame_index = frame_idx_from_name(frame_path)
        timestamp_sec = frame_index / EXTRACTED_FPS if frame_index >= 0 else None

        img_bgr = cv2.imread(str(frame_path))
        kept = 0

        for box, score, label in zip(boxes, scores, labels):
            label_str = str(label).strip().lower()
            if float(score) < 0.55:
                continue
            if not any(k in label_str for k in ["drone", "quadcopter", "uav"]):
                continue

            x1, y1, x2, y2 = map(int, box)

            bw = x2 - x1
            bh = y2 - y1
            area = bw * bh
            img_h, img_w = img_bgr.shape[:2]
            img_area = img_h * img_w

            if area < 20:
                continue

            if area > 0.01 * img_area:
                continue

            kept += 1

            rows.append(
                {
                    "video_id": video_id,
                    "frame_file": frame_path.name,
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                    "class_label": label_str,
                    "bounding_box": [x1, y1, x2, y2],
                    "confidence_score": float(score),
                }
            )

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img_bgr,
                f"{label_str}: {score:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if kept > 0:
            out_path = out_video_dir / frame_path.name
            cv2.imwrite(str(out_path), img_bgr)

df = pd.DataFrame(rows)
parquet_path = ROOT / "results" / "detections.parquet"
parquet_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(parquet_path, index=False)

print(f"Saved {len(df)} detections to {parquet_path}")
print(df.head())