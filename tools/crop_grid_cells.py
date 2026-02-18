import argparse
from pathlib import Path

import cv2


def compute_image_grid_edges(width, height, rows, cols):
    x_edges = [int(round(i * width / cols)) for i in range(cols + 1)]
    y_edges = [int(round(i * height / rows)) for i in range(rows + 1)]
    return x_edges, y_edges


def main():
    parser = argparse.ArgumentParser(description="Crop each image into rows x columns equal cells.")
    parser.add_argument(
        "--image",
        default=None,
        help="Path to a single full lawn image (overrides --input-dir)",
    )
    parser.add_argument(
        "--input-dir",
        default="data/frames",
        help="Directory containing full lawn images",
    )
    parser.add_argument("--rows", type=int, default=5, help="Number of grid rows")
    parser.add_argument("--columns", type=int, default=9, help="Number of grid columns")
    parser.add_argument(
        "--output",
        default="data/cnn_cells/uncategorized",
        help="Output directory for crops",
    )
    parser.add_argument(
        "--ext",
        default="png",
        help="Image extension for crops (png or jpg)",
    )

    args = parser.parse_args()
    output_root = Path(args.output)
    rows = args.rows
    cols = args.columns

    if rows <= 0 or cols <= 0:
        raise ValueError("--rows and --columns must be positive integers.")

    if args.image is None:
        frames_dir = Path(args.input_dir)
        candidates = list(frames_dir.glob("*.png")) + list(frames_dir.glob("*.jpg"))
        if not candidates:
            raise FileNotFoundError(f"No images found in {frames_dir}. Pass --image.")
        image_paths = candidates
    else:
        image_paths = [Path(args.image)]

    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        h, w = img.shape[:2]
        x_edges, y_edges = compute_image_grid_edges(w, h, rows, cols)
        out_dir = output_root / image_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        for r in range(rows):
            for c in range(cols):
                x0 = x_edges[c]
                y0 = y_edges[r]
                x1 = x_edges[c + 1]
                y1 = y_edges[r + 1]

                x0 = max(0, x0)
                y0 = max(0, y0)
                x1 = min(w, x1)
                y1 = min(h, y1)

                crop = img[y0:y1, x0:x1]
                out_path = out_dir / f"r{r}_c{c}.{args.ext}"
                cv2.imwrite(str(out_path), crop)

        print(f"Saved crops to: {out_dir}")


if __name__ == "__main__":
    main()
