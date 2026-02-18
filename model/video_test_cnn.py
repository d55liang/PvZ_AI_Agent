import argparse
import json
from pathlib import Path

import cv2
import torch
from torchvision import transforms

try:
    from model.cell_state_classifier import CellStateCNN
except ModuleNotFoundError:
    from cell_state_classifier import CellStateCNN


def compute_grid_edges(width, height, rows, cols):
    x_edges = [int(round(i * width / cols)) for i in range(cols + 1)]
    y_edges = [int(round(i * height / rows)) for i in range(rows + 1)]
    return x_edges, y_edges


def resolve_lawn_rect(frame_w, frame_h, lawn_json_path, roi_json_path):
    if lawn_json_path is None:
        return 0, 0, frame_w, frame_h

    lawn = json.load(open(lawn_json_path, "r"))
    (lx0, ly0) = lawn["top_left"]
    (lx1, ly1) = lawn["bottom_right"]

    if 0 <= lx0 < lx1 <= frame_w and 0 <= ly0 < ly1 <= frame_h:
        return int(lx0), int(ly0), int(lx1), int(ly1)

    if roi_json_path is not None:
        roi = json.load(open(roi_json_path, "r"))
        rx, ry = int(roi["left"]), int(roi["top"])
        rw, rh = int(roi["width"]), int(roi["height"])

        lx0_local = int(round(((lx0 - rx) / max(rw, 1)) * frame_w))
        ly0_local = int(round(((ly0 - ry) / max(rh, 1)) * frame_h))
        lx1_local = int(round(((lx1 - rx) / max(rw, 1)) * frame_w))
        ly1_local = int(round(((ly1 - ry) / max(rh, 1)) * frame_h))

        lx0_local = max(0, min(frame_w - 1, lx0_local))
        ly0_local = max(0, min(frame_h - 1, ly0_local))
        lx1_local = max(lx0_local + 1, min(frame_w, lx1_local))
        ly1_local = max(ly0_local + 1, min(frame_h, ly1_local))
        return lx0_local, ly0_local, lx1_local, ly1_local

    raise ValueError(
        "Lawn coords do not fit this video frame. Provide --roi-json for conversion "
        "or pass an empty --lawn-json to use full frame."
    )


def short_label(name, max_len=10):
    if len(name) <= max_len:
        return name
    return name[: max_len - 1] + "."


def main():
    parser = argparse.ArgumentParser(description="Run cell-state CNN on a video and visualize per-cell predictions.")
    parser.add_argument("--checkpoint", default="model/runs/cnn/best.pt", help="Path to CNN checkpoint")
    parser.add_argument("--input", default="data/videos/input.mp4", help="Input video path")
    parser.add_argument("--output", default="data/videos/output_cnn.mp4", help="Output video path")
    parser.add_argument("--rows", type=int, default=5, help="Grid rows")
    parser.add_argument("--columns", type=int, default=9, help="Grid columns")
    parser.add_argument("--lawn-json", default="configs/lawn.json", help="Path to lawn json; set '' to use full frame")
    parser.add_argument("--roi-json", default="configs/roi.json", help="Path to roi json for absolute->local conversion")
    parser.add_argument("--margin-left", type=int, default=None, help="Override lawn left margin in pixels")
    parser.add_argument("--margin-right", type=int, default=None, help="Override lawn right margin in pixels")
    parser.add_argument("--margin-top", type=int, default=None, help="Override lawn top margin in pixels")
    parser.add_argument("--margin-bottom", type=int, default=None, help="Override lawn bottom margin in pixels")
    parser.add_argument("--show-conf", action="store_true", help="Overlay confidence for each cell")
    parser.add_argument(
        "--save-boards-json",
        default=None,
        help="Optional path to save per-frame board predictions as JSON",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    num_classes = len(class_to_idx)
    img_size = int(ckpt.get("img_size", 96))

    model = CellStateCNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if any(v is not None for v in [args.margin_left, args.margin_right, args.margin_top, args.margin_bottom]):
        left = args.margin_left if args.margin_left is not None else 0
        right = args.margin_right if args.margin_right is not None else 0
        top = args.margin_top if args.margin_top is not None else 0
        bottom = args.margin_bottom if args.margin_bottom is not None else 0
        lawn_x0 = max(0, left)
        lawn_y0 = max(0, top)
        lawn_x1 = min(width, width - max(0, right))
        lawn_y1 = min(height, height - max(0, bottom))
    else:
        lawn_json_path = args.lawn_json if args.lawn_json else None
        roi_json_path = args.roi_json if args.roi_json else None
        lawn_x0, lawn_y0, lawn_x1, lawn_y1 = resolve_lawn_rect(width, height, lawn_json_path, roi_json_path)

    if lawn_x1 <= lawn_x0 or lawn_y1 <= lawn_y0:
        raise ValueError(f"Invalid lawn rectangle: ({lawn_x0}, {lawn_y0}) -> ({lawn_x1}, {lawn_y1})")

    print(f"Using lawn rect: x=({lawn_x0},{lawn_x1}) y=({lawn_y0},{lawn_y1})")
    lawn_w = lawn_x1 - lawn_x0
    lawn_h = lawn_y1 - lawn_y0
    x_edges, y_edges = compute_grid_edges(lawn_w, lawn_h, args.rows, args.columns)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    all_boards = []
    frame_idx = 0

    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            crops = []
            for r in range(args.rows):
                for c in range(args.columns):
                    x0, x1 = x_edges[c], x_edges[c + 1]
                    y0, y1 = y_edges[r], y_edges[r + 1]
                    crop_bgr = frame[lawn_y0 + y0 : lawn_y0 + y1, lawn_x0 + x0 : lawn_x0 + x1]
                    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                    crops.append(tf(crop_rgb))

            batch = torch.stack(crops, dim=0).to(device, non_blocking=True)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)
            conf_vals, pred_idxs = torch.max(probs, dim=1)

            board = []
            k = 0
            for r in range(args.rows):
                row = []
                for c in range(args.columns):
                    pred_idx = int(pred_idxs[k].item())
                    conf = float(conf_vals[k].item())
                    cls_name = idx_to_class[pred_idx]
                    row.append({"class": cls_name, "confidence": conf})

                    x0, x1 = lawn_x0 + x_edges[c], lawn_x0 + x_edges[c + 1]
                    y0, y1 = lawn_y0 + y_edges[r], lawn_y0 + y_edges[r + 1]

                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)
                    text = short_label(cls_name)
                    if args.show_conf:
                        text = f"{text} {conf:.2f}"
                    cv2.putText(
                        frame,
                        text,
                        (x0 + 3, min(y1 - 4, y0 + 14)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    k += 1
                board.append(row)

            cv2.rectangle(frame, (lawn_x0, lawn_y0), (lawn_x1, lawn_y1), (255, 255, 0), 2)
            writer.write(frame)
            all_boards.append({"frame": frame_idx, "board": board})
            frame_idx += 1

    cap.release()
    writer.release()

    if args.save_boards_json:
        boards_path = Path(args.save_boards_json)
        boards_path.parent.mkdir(parents=True, exist_ok=True)
        with boards_path.open("w") as f:
            json.dump(all_boards, f)
        print(f"Saved board predictions: {boards_path}")

    print(f"Success: {out_path}")


if __name__ == "__main__":
    main()
