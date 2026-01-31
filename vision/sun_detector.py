import numpy as np
import cv2
import json
from mss import mss

HSV_PATH = "configs/hsv.json"

def get_hsv_lo_hi(name):
    hsv = json.load(open(HSV_PATH, 'r'))
    ranges = hsv[name]
    lower = np.array(ranges["lower"], dtype=np.uint8)
    upper = np.array(ranges["upper"], dtype=np.uint8)
    return lower, upper

def detect_sun(lawn_bgr):
    lawn_hsv = cv2.cvtColor(lawn_bgr, cv2.COLOR_BGR2HSV)
    
    lower, upper = get_hsv_lo_hi("Sun")

    '''
    lower = np.array([75, 40, 170], dtype=np.uint8)
    upper = np.array([115, 255, 255], dtype=np.uint8)
    '''

    mask = cv2.inRange(lawn_hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area <= 2000: continue
        if area >= 3000: continue

        '''
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter)
        if circularity < 0.2: continue
        '''

        (x, y), _ = cv2.minEnclosingCircle(cnt)
        centers.append((int(x), int(y)))
    
    return centers
        
def draw_debug(lawn_bgr, centers):
    sc = lawn_bgr.copy()
    for (x, y) in centers:
        cv2.circle(sc, (x, y), 10, (0, 0, 255), 2)
        cv2.circle(sc, (x, y), 1, (0, 0, 255), -1)
    return sc