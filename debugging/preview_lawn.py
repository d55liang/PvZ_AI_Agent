import json
import numpy as np 
import cv2
import time
from mss import mss

GRID_PATH = "configs/grid.json"
LAWN_PATH = "configs/lawn.json"

def main():
    grid = json.load(open(GRID_PATH, 'r'))
    lawn = json.load(open(LAWN_PATH, 'r'))

    left, top = lawn['top_left']
    right, bottom = lawn['bottom_right']

    roi = {"top" : top,
           "left" : left,
           "width" : right - left,
           "height" : bottom - top}

    with mss() as sct:
        while True:
            frame = np.array(sct.grab(roi))[:, :, :3]
            sc = frame.copy()
            cv2.imshow("PvZ Lawn Preview (Press q to quit)", sc)

            key = cv2.waitKey(1)
            key = key & 0xFF
            if key == ord('q'):
                print("Exit success")
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()