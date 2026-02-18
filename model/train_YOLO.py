import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for PvZ detection.")
    parser.add_argument("--model", default="yolov8m", help="Base model or weights path")
    parser.add_argument("--data", default="model/data.yaml", help="Dataset yaml path")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs")
    args = parser.parse_args()

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs)


if __name__ == "__main__":
    main()