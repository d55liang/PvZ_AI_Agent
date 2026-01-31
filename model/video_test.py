from ultralytics import YOLO
import cv2
import numpy as np 

MODEL_PATH = "model/runs/detect/train3/weights/best.pt"
INPUT_VIDEO = "data/videos/input.mp4"
OUTPUT_VIDEO = "data/videos/output.mp4"

conf = 0.25

model = YOLO(MODEL_PATH)

cpt = cv2.VideoCapture(INPUT_VIDEO)

fps = cpt.get(cv2.CAP_PROP_FPS)
w = int(cpt.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cpt.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

while True:
    retval, frame = cpt.read()
    if not retval:
        break

    results = model.predict(frame, conf=conf, verbose=False)[0]

    for box in results.boxes:   
        x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls[0])
        conf = float(box.conf[0])
    
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

    output.write(frame)

cpt.release()
output.release()
print(f"Success: {OUTPUT_VIDEO}")