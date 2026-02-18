import numpy as np
import cv2
import json

HSV_PATH = "configs/hsv.json"


def get_hsv_lo_hi(name):
    hsv = json.load(open(HSV_PATH, "r"))
    ranges = hsv[name]
    lower = np.array(ranges["lower"], dtype=np.uint8)
    upper = np.array(ranges["upper"], dtype=np.uint8)
    return lower, upper


def detect_sun(
    lawn_bgr,
    area_min=600,
    area_max=8000,
    peak_rel=0.35,
    kernel_size=3,
    circularity_min=0.30,
    extent_min=0.40,
    contour_area_min=600,
    bbox_min_side=14,
):
    lawn_hsv = cv2.cvtColor(lawn_bgr, cv2.COLOR_BGR2HSV)

    lower, upper = get_hsv_lo_hi("Sun")

    mask = cv2.inRange(lawn_hsv, lower, upper)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() <= 0:
        return []

    _, peaks = cv2.threshold(dist, dist.max() * peak_rel, 255, 0)
    peaks = peaks.astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(peaks)

    centers = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < area_min or area > area_max:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        if w < bbox_min_side or h < bbox_min_side:
            continue

        comp_mask = (labels[y : y + h, x : x + w] == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        cnt_area = cv2.contourArea(cnt)
        if cnt_area < contour_area_min:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter <= 0:
            continue
        circularity = (4 * np.pi * cnt_area) / (perimeter * perimeter)
        if circularity < circularity_min:
            continue

        extent = cnt_area / float(w * h)
        if extent < extent_min:
            continue

        cx, cy = centroids[i]
        centers.append((int(cx), int(cy)))

    return centers


def draw_debug(lawn_bgr, centers):
    sc = lawn_bgr.copy()
    for (x, y) in centers:
        cv2.circle(sc, (x, y), 10, (0, 0, 255), 2)
        cv2.circle(sc, (x, y), 1, (0, 0, 255), -1)
    return sc
