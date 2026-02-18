import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export a YOLO .pt model to ONNX.")
    parser.add_argument(
        "--weights",
        default="model/runs/detect/train5/weights/best.pt",
        help="Path to YOLO .pt weights",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Export image size")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset version")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input shape for ONNX export",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export in FP16 (requires compatible hardware/runtime)",
    )
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Missing weights file: {weights}")

    model = YOLO(str(weights))
    out_path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        dynamic=args.dynamic,
        half=args.half,
    )
    print(f"ONNX export complete: {out_path}")


if __name__ == "__main__":
    main()
