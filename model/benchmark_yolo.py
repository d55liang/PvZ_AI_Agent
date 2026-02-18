import argparse
import statistics
import time

import cv2
from ultralytics import YOLO


def load_frames(video_path, n_frames):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames


def benchmark(model_path, frames, imgsz, conf, warmup):
    model = YOLO(model_path)
    timings = []
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        _ = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
        t1 = time.perf_counter()
        if i >= warmup:
            timings.append((t1 - t0) * 1000.0)
    return timings


def main():
    parser = argparse.ArgumentParser(description="Benchmark YOLO .pt/.onnx latency and FPS.")
    parser.add_argument("--video", default="data/videos/input.mp4", help="Input video for benchmark")
    parser.add_argument("--model", required=True, help="Model path (.pt or .onnx)")
    parser.add_argument("--imgsz", type=int, default=416, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    parser.add_argument("--frames", type=int, default=140, help="Total frames to use")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup frames to skip in metrics")
    args = parser.parse_args()

    frames = load_frames(args.video, args.frames)
    if len(frames) <= args.warmup:
        raise RuntimeError(f"Need > warmup frames, got {len(frames)}")

    timings = benchmark(args.model, frames, args.imgsz, args.conf, args.warmup)
    avg_ms = statistics.mean(timings)
    p50_ms = statistics.median(timings)
    fps = 1000.0 / avg_ms

    print(f"model={args.model}")
    print(f"imgsz={args.imgsz} conf={args.conf}")
    print(f"frames_total={len(frames)} warmup={args.warmup} measured={len(timings)}")
    print(f"avg_latency_ms={avg_ms:.2f}")
    print(f"p50_latency_ms={p50_ms:.2f}")
    print(f"fps={fps:.2f}")


if __name__ == "__main__":
    main()
