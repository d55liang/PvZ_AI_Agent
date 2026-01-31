import json
import numpy as np 
import cv2
import time
from mss import mss

ROI_PATH = "configs/roi.json"

def main():
    roi = json.load(open(ROI_PATH, "r"))

    fps_x = roi['left']
    fps_y = roi['top'] + roi['height']

    with mss() as sct:
        frames = 0
        start_time = time.time()
        fps = 0.0

        while True:
            frame = np.array(sct.grab(roi))[:, :, :3]
            frames += 1
            end_time = time.time()

            if end_time - start_time >= 1:
                elapsed_time = end_time - start_time
                fps = frames / elapsed_time
                frames = 0
                start_time = end_time
            
            sc = frame.copy()
            cv2.putText(sc, f"fps: {fps:.1f}", (fps_x, fps_y), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("PvZ ROI Preview (Press q to exit)", sc)

            key = cv2.waitKey(1)
            key = key & 0xFF
            if key == ord('q'):
                print("Exit success")
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()