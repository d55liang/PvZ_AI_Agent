import json
import time
import pyautogui

pyautogui.FAILSAFE = True

ROI_PATH = "configs/roi.json"
GRID_PATH = "configs/grid.json"

roi = json.load(open(ROI_PATH, 'r'))
grid = json.load(open(GRID_PATH, 'r'))

def get_center(grid, row, col):
    x0, y0 = grid['top_left']
    x1, y1 = grid['bottom_right']
    rows = grid['rows']
    columns = grid['columns']

    delta_x = (x1 - x0) / (columns - 1)
    delta_y = (y1 - y0) / (rows - 1)

    return int(x0 + (col * delta_x)), int(y0 + (row * delta_y))

def main():
    print("In PvZ, manually select a plant first")
    time.sleep(5.0)

    tests = [(4, 8)]

    for (row, col) in tests:
        center_x, center_y = get_center(grid, row, col)

        pyautogui.moveTo(center_x, center_y)
        pyautogui.click()
        time.sleep(2.0)

if __name__ == "__main__":
    main()