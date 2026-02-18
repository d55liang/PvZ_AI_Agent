import time
import pyautogui

pyautogui.FAILSAFE = True

class ActionExecutor:
    def __init__(self, slot_centers: dict[str, tuple[int, int]],
                 cell_centers: dict[tuple[int, int], tuple[int, int]]):
        self.slot_centers = slot_centers
        self.cell_centers = cell_centers

    def select_seed(self, plant: str):
        x, y = self.slot_centers[plant]
        pyautogui.click(x, y)
        time.sleep(0.3)

    def place_plant(self, row: int, col: int):
        x, y = self.cell_centers[(row, col)]
        pyautogui.click(x, y)
        time.sleep(0.3)
    
    def plant(self, plant: str, row: int, col: int):
        self.select_seed(plant)
        self.place_plant(row, col)

    def cancel_selection(self):
        # In PvZ, right click cancels a selected seed packet.
        pyautogui.rightClick()
        time.sleep(0.05)
