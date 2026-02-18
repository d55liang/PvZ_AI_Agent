import argparse

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run ONNX YOLO inference on a video.")
    parser.add_argument(
        "--model",
        default="model/runs/detect/train5/weights/best.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument("--input", default="data/videos/input.mp4", help="Input video path")
    parser.add_argument("--output", default="data/videos/output_onnx.mp4", help="Output video path")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    args = parser.parse_args()

    model = YOLO(args.model)
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        for box in results.boxes:
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            conf_val = float(box.conf[0])
            label = f"{model.names[cls]} {conf_val:.2f}"
            cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(frame, label, (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Success: {args.output}")


if __name__ == "__main__":
    main()
