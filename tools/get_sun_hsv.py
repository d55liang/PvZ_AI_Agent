import os
import json
import cv2
import numpy as np

IMG_PATH = "data/samples/sun_hsv.png"
HSV_PATH = "configs/hsv.json"


def get_sun_hsv():
    sun_bgr = cv2.imread(IMG_PATH)
    if sun_bgr is None:
        raise FileNotFoundError(f"Missing image: {IMG_PATH}")

    sun_hsv = cv2.cvtColor(sun_bgr, cv2.COLOR_BGR2HSV)

    h = sun_hsv[:, :, 0].astype(np.int32)
    s = sun_hsv[:, :, 1].astype(np.int32)
    v = sun_hsv[:, :, 2].astype(np.int32)

    lower = [int(h.min()), int(s.min()), int(v.min())]
    upper = [int(h.max()), int(s.max()), int(v.max())]

    print("HSV lower:", lower)
    print("HSV upper:", upper)

    os.makedirs(os.path.dirname(HSV_PATH), exist_ok=True)

    data = {}
    if os.path.exists(HSV_PATH):
        with open(HSV_PATH, "r") as f:
            data = json.load(f)

    data["Sun"] = {"lower": lower, "upper": upper}

    with open(HSV_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print("Sun HSV saved successfully")


if __name__ == "__main__":
    get_sun_hsv()
