from ultralytics import YOLO
import cv2
import numpy as np

MODEL_PATH = "model/runs/detect/train3/weights/best.pt"
VIDEO_PATH = "data/videos/input.mp4"
FRAME_PATH = "data/frames/output.png"
FRAME_IDX = 1000    #select a frame

conf = 0.40

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_IDX)

ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError(f"Could not read frame {FRAME_IDX}")

results = model.predict(frame, conf=conf, verbose=False)[0]

for box in results.boxes:
    x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
    cls = int(box.cls[0])
    conf_val = float(box.conf[0])

    label = f"{model.names[cls]} {conf_val:.2f}"
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.putText(frame, label, (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imwrite(FRAME_PATH, frame)
print(f"Success: {FRAME_PATH}")