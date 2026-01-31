import json
import numpy as np 
import cv2
from mss import mss
import os

ROI_PATH = "configs/roi.json"

points = []

def mouse_click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((int(x), int(y)))
        print(f"Clicked ({int(x)}, {int(y)})")

def main():
    with mss() as sct:
        monitor = sct.monitors[0]
        frame = np.array(sct.grab(monitor))[:, :, :3]

    cv2.namedWindow("Click top left and bottom right of the PvZ window." 
                        "Press: r = reset, q = exit, s = save", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click top left and bottom right of the PvZ window." 
                        "Press: r = reset, q = exit, s = save", mouse_click)

    while True:
        sc = frame.copy()

        for i, (x, y) in enumerate(points):
            cv2.circle(sc, (x, y), 2, (0, 255, 0), 2)
            cv2.putText(sc, str(i + 1), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if len(points) == 2:
            (x0, y0), (x1, y0) = points[0], points[1]
            left, right =  sorted([x0, x1])
            top, bottom = sorted([y0, y0])
            cv2.rectangle(sc, (left, top), (right, bottom), (0, 255, 0), 1)

        cv2.imshow("Click top left and bottom right of the PvZ window." 
                        "Press: r = reset, q = exit, s = save", sc)

        key = cv2.waitKey(1)
        key = key & 0xFF

        if key == ord('r'):
            points.clear()
            print("Reset success")
        elif key == ord('q'):
            print("Exit success")
            break
        elif key == ord('s'):
            if len(points) < 2:
                print("Need to select 2 points")
                continue

            (x0, y0), (x1, y1) = points[0], points[1]
            left, right =  sorted([x0, x1])
            top, bottom = sorted([y0, y1])

            roi = {"left" : int(monitor['left'] + left),
                   "top" : int(monitor['top'] + top),
                   "width" : int(right - left),
                   "height" : int(bottom - top)}

            os.makedirs(os.path.dirname(ROI_PATH), exist_ok=True)

            with open(ROI_PATH, "w") as f:
                json.dump(roi, f, indent=2)
            
            print("Save success")
            
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()