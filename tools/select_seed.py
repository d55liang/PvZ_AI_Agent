import json
import os

import cv2
import numpy as np
from mss import mss

SEED_PATH = "configs/seed.json"
SLOT_NAMES = ["slot_1", "slot_2", "slot_3", "slot_4", "slot_5", "slot_6"]
ACTIVE_PLANT_SLOTS = {
    "sun_flower": "slot_1",
    "pea_shooter": "slot_2",
    "snow_pea": "slot_3",
    "wall_nut": "slot_4",
}
TARGET_SEED_WIDTH = 350
TARGET_SEED_HEIGHT = 70

slot_count = len(SLOT_NAMES)

points = []

def mouse_click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((int(x), int(y)))
        print(f"Clicked ({int(x)}, {int(y)})")

def build_seed_config(left, top, right, bottom, monitor_left, monitor_top):
    width = right - left
    height = bottom - top

    if width <= 0 or height <= 0:
        raise ValueError("Invalid seed region size.")

    slot_width = width / slot_count
    y_center = top + (height / 2.0)

    slots = {}
    for i, name in enumerate(SLOT_NAMES):
        x_center = left + ((i + 0.5) * slot_width)
        slots[name] = [int(x_center + monitor_left), int(y_center + monitor_top)]

    return {
        "seed_region": {
            "left": int(left + monitor_left),
            "top": int(top + monitor_top),
            "width": int(width),
            "height": int(height),
        },
        "slot_names": SLOT_NAMES,
        "active_plant_slots": ACTIVE_PLANT_SLOTS,
        "slot_centers": slots,
    }

def tighten_seed_rect(left, top, right, bottom):
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return left, top, right, bottom

    target_w = min(width, TARGET_SEED_WIDTH)
    target_h = min(height, TARGET_SEED_HEIGHT)

    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0

    new_left = int(round(cx - target_w / 2.0))
    new_right = int(round(cx + target_w / 2.0))
    new_top = int(round(cy - target_h / 2.0))
    new_bottom = int(round(cy + target_h / 2.0))
    return new_left, new_top, new_right, new_bottom

def main():
    window_name = (
        "Click top-left and bottom-right of the 6-slot seed bar. "
        "Press: r = reset, q = exit, s = save"
    )

    with mss() as sct:
        monitor = sct.monitors[0]
        frame = np.array(sct.grab(monitor))[:, :, :3]

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_click)

    while True:
        sc = frame.copy()

        for i, (x, y) in enumerate(points):
            cv2.circle(sc, (x, y), 2, (0, 255, 255), 2)
            cv2.putText(
                sc,
                str(i + 1),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

        if len(points) == 2:
            (x0, y0), (x1, y1) = points[0], points[1]
            left, right = sorted([x0, x1])
            top, bottom = sorted([y0, y1])
            left, top, right, bottom = tighten_seed_rect(left, top, right, bottom)

            cv2.rectangle(sc, (left, top), (right, bottom), (0, 255, 255), 1)

            slot_width = (right - left) / slot_count
            for i in range(1, slot_count):
                x_line = int(left + i * slot_width)
                cv2.line(sc, (x_line, top), (x_line, bottom), (255, 255, 0), 1)

        cv2.imshow(window_name, sc)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            points.clear()
            print("Reset success")
        elif key == ord("q"):
            print("Exit success")
            break
        elif key == ord("s"):
            if len(points) < 2:
                print("Need to select 2 points")
                continue

            (x0, y0), (x1, y1) = points[0], points[1]
            left, right = sorted([x0, x1])
            top, bottom = sorted([y0, y1])
            left, top, right, bottom = tighten_seed_rect(left, top, right, bottom)

            try:
                seed_cfg = build_seed_config(
                    left=left,
                    top=top,
                    right=right,
                    bottom=bottom,
                    monitor_left=monitor["left"],
                    monitor_top=monitor["top"],
                )
            except ValueError as e:
                print(str(e))
                continue

            os.makedirs(os.path.dirname(SEED_PATH), exist_ok=True)
            with open(SEED_PATH, "w") as f:
                json.dump(seed_cfg, f, indent=2)

            print(f"Save success: {SEED_PATH}")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
