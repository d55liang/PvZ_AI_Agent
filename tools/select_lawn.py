import json
import os

GRID_PATH = "configs/grid.json"
LAWN_PATH = "configs/lawn.json"

def get_lawn(grid):
    left, top = grid['top_left']
    right, bottom = grid['bottom_right']
    rows = grid['rows']
    columns = grid['columns']

    delta_x = (right - left) / (columns - 1)
    delta_y = (bottom - top) / (rows - 1) 

    lawn_left = left - (delta_x / 2)
    lawn_right = right + (delta_x / 2)
    lawn_top = top - (delta_y / 2)
    lawn_bottom = bottom + (delta_y / 2)

    lawn = {"top_left" : (int(lawn_left), int(lawn_top)),
            "bottom_right" : (int(lawn_right), int(lawn_bottom))}

    os.makedirs(os.path.dirname(LAWN_PATH), exist_ok=True)

    with open(LAWN_PATH, 'w') as f:
        json.dump(lawn, f, indent=2)
    print("Lawn selection success")

if __name__ == "__main__":
    grid = json.load(open(GRID_PATH, 'r'))
    get_lawn(grid)