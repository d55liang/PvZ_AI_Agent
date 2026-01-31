import json
import numpy as np 
import cv2
from mss import mss
import os

GRID_PATH = "configs/grid.json"

points = []

def mouse_click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Clicked ({int(x)}, {int(y)})")

def main():
    with mss() as sct:
        monitor = sct.monitors[0]
        frame = np.array(sct.grab(monitor))[:, :, :3]
    
    cv2.namedWindow("Click center of top-left and bottom-right grids." 
                        "Press: r = reset, q = exit, s = save", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click center of top-left and bottom-right grids." 
                        "Press: r = reset, q = exit, s = save", mouse_click)

    while True:
        sc = frame.copy()
        
        for i, (x, y) in enumerate(points):
            cv2.circle(sc, (x, y), 2, (255, 0, 0), 2)
            cv2.putText(sc, f"{i + 1}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if len(points) == 2:
            (x0, y0), (x1, y1) = points[0], points[1]
            left, right = sorted([x0, x1])
            top, bottom = sorted([y0, y1])
            cv2.rectangle(sc, (left, top), (right, bottom), (0, 0, 255), 1)

        cv2.imshow("Click center of top-left and bottom-right grids." 
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
            left, right = sorted([x0, x1])
            top, bottom = sorted([y0, y1])

            grid = {"top_left" : (x0, y0),
                    "bottom_right" : (x1, y1),
                    "rows" : 5,
                    "columns" : 9}
            
            os.makedirs(os.path.dirname(GRID_PATH), exist_ok=True)

            with open(GRID_PATH, "w") as f:
                json.dump(grid, f, indent=2)
            
            print("Save success")

            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()