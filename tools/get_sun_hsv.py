import numpy as np
import cv2
import json
import os

IMG_PATH = "data/samples/sun_hsv.png"
HSV_PATH = "configs/hsv.json"

def get_sun_hsv():
    sun_bgr = cv2.imread(IMG_PATH)
    if sun_bgr is None:
        raise FileNotFoundError(f"Could not read: {IMG_PATH}")

    sun_hsv = cv2.cvtColor(sun_bgr, cv2.COLOR_BGR2HSV)

    H = sun_hsv[:, :, 0].astype(np.int32)
    S = sun_hsv[:, :, 1].astype(np.int32)
    V = sun_hsv[:, :, 2].astype(np.int32)

    bright = V > 160
    if bright.sum() < 10:
        bright = np.ones_like(V, dtype=bool)

    Hs = H[bright]
    Ss = S[bright]
    Vs = V[bright]

    h_lo, h_hi = np.percentile(Hs, [35, 95]).astype(int)
    s_lo, s_hi = np.percentile(Ss, [35, 95]).astype(int)
    v_lo, v_hi = np.percentile(Vs, [35, 95]).astype(int)

    h_lo = max(h_lo - 5, 0)
    h_hi = min(h_hi + 15, 179)
    s_lo = max(s_lo - 60, 0)
    v_lo = max(v_lo - 60, 0)

    lower = [h_lo, s_lo, v_lo]
    upper = [h_hi, 255, 255]

    print("HSV lower:", lower)
    print("HSV upper:", upper)

    os.makedirs(os.path.dirname(HSV_PATH), exist_ok=True)

    data = {}

    if os.path.exists(HSV_PATH):
        with open(HSV_PATH, 'r') as f:
            data = json.load(f)
    
    data["Sun"] = {
        "lower" : [int(x) for x in lower],
        "upper" : [int(y) for y in upper]
    }

    with open(HSV_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    print("Sun HSV saved successlly")

if __name__ == "__main__":
    get_sun_hsv()