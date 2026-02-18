import cv2
import numpy as np

IMG_PATH = "data/frames/sun_overlap.png"


def main():
    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"Missing image: {IMG_PATH}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        b, g, r = img[y, x]
        h, s, v = hsv[y, x]
        print(f"xy=({x},{y}) BGR=({b},{g},{r}) HSV=({h},{s},{v})")

    cv2.namedWindow("pick_hsv", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("pick_hsv", on_mouse)

    print("Click on the image to print HSV. Press ESC to quit.")
    while True:
        cv2.imshow("pick_hsv", img)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
