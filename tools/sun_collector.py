import json
import numpy as np 
import pyautogui
import cv2
import time
from mss import mss
from collections import deque
from vision.sun_detector import detect_sun, draw_debug

LAWN_PATH = "configs/lawn.json"

pyautogui.FALSESAFE = True

def dist(p0, p1):
    dist_sqr = pow(p1[0] - p0[0], 2) + pow(p1[1] - p0[1], 2)
    return pow(dist_sqr, 0.5)

def main():

    lawn = json.load(open(LAWN_PATH, 'r'))
    left, top = lawn['top_left']
    right, bottom = lawn['bottom_right']

    monitor = {"left" : left,
               "top" : top,
               "width" : right - left,
               "height" : bottom - top}

    recent_suns = deque(maxlen=15)
    click_pause = 0.05

    # For testing and debugging purporses
    print("Sun collector starting in 5 seconds")
    time.sleep(5.0)

    while True:
        with mss() as sct:
            frame = np.array(sct.grab(monitor))[:, :, :3]

        lawn_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        centers = detect_sun(lawn_bgr)

        draw_debug(lawn_bgr, centers)

        mouse_x, mouse_y = pyautogui.position()
        if not (left <= mouse_x <= right and top <= mouse_y <= bottom):
            time.sleep(0.05)
            continue

        for (x, y) in centers:
            sun_x = left + x
            sun_y = top + y 

            pyautogui.click(sun_x, sun_y)
            recent_suns.append((sun_x, sun_y))
            pyautogui.moveTo(right - 20, bottom - 30)
            time.sleep(click_pause)

if __name__ == "__main__":
    main()
