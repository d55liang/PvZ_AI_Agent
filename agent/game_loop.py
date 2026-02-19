import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyautogui
import torch
from mss import mss
from torchvision import transforms
from ultralytics import YOLO

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from agent.actions import ActionExecutor
from agent.logic import (
    SHOOTER_COLS,
    choose_plant_decision,
    compute_lane_gap,
    pick_priority_row,
    pick_zombie_presence_row,
)
from agent.state import GameState, PlantObs, ZombieObs
from model.cell_state_classifier import CellStateCNN
from vision.sun_detector import detect_sun

GRID_PATH = "configs/grid.json"
LAWN_PATH = "configs/lawn.json"
SEED_PATH = "configs/seed.json"

DEFAULT_YOLO = "model/runs/detect/train5/weights/best.onnx"
FALLBACK_YOLO_PT = "model/runs/detect/train5/weights/best.pt"
DEFAULT_CNN = "model/runs/cnn/best.pt"
NUM_ROWS = 5
NUM_COLS = 9
LAWN_CAPTURE_UPSHIFT_PX = 23

PLANT_ORDER_FOR_SLOTS = ["sun_flower", "pea_shooter", "snow_pea", "wall_nut"]
PLANT_TO_DEFAULT_SLOT = {
    "sun_flower": "slot_1",
    "pea_shooter": "slot_2",
    "snow_pea": "slot_3",
    "wall_nut": "slot_4",
}
SUN_CLICK_PAUSE = 0.03
SUN_DUPLICATE_DIST = 20
OCCUPANCY_MIN_CONF = 0.60
OCCUPANCY_CONFIRM_FRAMES = 2
CNN_SOFT_OCCUPY_CONF = 0.90
CNN_SOFT_EMPTY_VETO_CONF = 0.95

SUN_CREDIT_RATIO = 0.25
SUN_RECENT_BLOCK_DIST = 28
SUN_RECENT_BLOCK_FRAMES = 8
PLANT_CLICK_BLOCK_FRAMES = 12
PLANT_COMMIT_DELAY_SEC = 0.35
SHOOTER_COMMIT_REQUIRES_OCCUPIED = True
ZOMBIE_PRESENCE_HOLD_FRAMES = 18
COMBAT_FALLBACK_MIN_GAP = 0.15
PLANT_CONFIRM_FRAMES = 2
PLANT_CONFIRM_TIMEOUT_SEC = 1.8

ZOMBIE_LABEL_MAP = {
    "ordinary zombie": "ordinary",
    "ordinary zombies": "ordinary",
    "flag zombie": "flag",
    "pole zombie": "pole_vaulting",
    "pole vaulting zombie": "pole_vaulting",
    "conehead zombie": "conehead",
    "bucket zombie": "buckethead",
    "buckethead zombie": "buckethead",
}

pyautogui.FAILSAFE = True


def normalize_label(raw_label: str) -> str | None:
    key = raw_label.strip().lower()
    if key in ZOMBIE_LABEL_MAP:
        return ZOMBIE_LABEL_MAP[key]

    if "zombie" not in key:
        return None
    if "bucket" in key:
        return "buckethead"
    if "cone" in key:
        return "conehead"
    if "flag" in key:
        return "flag"
    if "pole" in key:
        return "pole_vaulting"
    return "ordinary"


def normalize_plant_label(raw_label: str) -> str | None:
    key = raw_label.strip().lower()
    if "zombie" in key:
        return None
    if "cart" in key or "mower" in key:
        return None
    if "bullet" in key or "projectile" in key:
        return None

    if "sunflower" in key:
        return "sun_flower"
    if "snow" in key or "ice" in key or "cold" in key:
        if "pea" in key:
            return "snow_pea"
    if "pea" in key:
        return "pea_shooter"
    if "wall" in key and "nut" in key:
        return "wall_nut"
    if "wallnut" in key or "walnut" in key:
        return "wall_nut"
    return None


def is_plant_label(raw_label: str) -> bool:
    if normalize_plant_label(raw_label) is not None:
        return True
    key = raw_label.strip().lower()
    if "zombie" in key or "cart" in key or "mower" in key or "bullet" in key or "projectile" in key:
        return False
    return any(
        k in key
        for k in ["potato", "mushroom", "flower", "repeater", "chomper", "cabbage", "plant"]
    )


def load_lawn_region() -> dict[str, int]:
    lawn = json.load(open(LAWN_PATH, "r"))
    x0, y0 = lawn["top_left"]
    x1, y1 = lawn["bottom_right"]
    left, right = sorted([int(x0), int(x1)])
    top, bottom = sorted([int(y0), int(y1)])
    # Capture a bit above the calibrated lawn to catch top-row zombie heads.
    capture_top = max(0, top - LAWN_CAPTURE_UPSHIFT_PX)
    capture_height = bottom - capture_top
    return {"left": left, "top": capture_top, "width": right - left, "height": capture_height}


def load_seed_slot_centers() -> dict[str, tuple[int, int]]:
    if not Path(SEED_PATH).exists():
        raise FileNotFoundError(
            f"Missing {SEED_PATH}. Run: python tools/select_seed.py"
        )

    data = json.load(open(SEED_PATH, "r"))
    slot_centers = data.get("slot_centers", {})
    active_plant_slots = data.get("active_plant_slots", PLANT_TO_DEFAULT_SLOT)
    plant_to_slot: dict[str, tuple[int, int]] = {}

    for plant in PLANT_ORDER_FOR_SLOTS:
        slot_name = active_plant_slots.get(plant, PLANT_TO_DEFAULT_SLOT[plant])
        if slot_name not in slot_centers:
            raise KeyError(f"Missing {slot_name} in {SEED_PATH}")
        x, y = slot_centers[slot_name]
        plant_to_slot[plant] = (int(x), int(y))

    return plant_to_slot


def build_cell_centers_absolute() -> dict[tuple[int, int], tuple[int, int]]:
    grid = json.load(open(GRID_PATH, "r"))
    (x0, y0) = grid["top_left"]
    (x1, y1) = grid["bottom_right"]
    rows = int(grid["rows"])
    cols = int(grid["columns"])

    if rows != NUM_ROWS or cols != NUM_COLS:
        raise ValueError(f"Expected {NUM_ROWS}x{NUM_COLS} grid, got {rows}x{cols}")

    dx = (x1 - x0) / max(cols - 1, 1)
    dy = (y1 - y0) / max(rows - 1, 1)

    centers: dict[tuple[int, int], tuple[int, int]] = {}
    for r in range(rows):
        for c in range(cols):
            cx = int(round(x0 + c * dx))
            cy = int(round(y0 + r * dy))
            centers[(r, c)] = (cx, cy)
    return centers


def build_cell_boxes_lawn_local(
    lawn_region: dict[str, int],
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    centers_abs = build_cell_centers_absolute()
    rows = NUM_ROWS
    cols = NUM_COLS

    x_vals = [centers_abs[(0, c)][0] for c in range(cols)]
    y_vals = [centers_abs[(r, 0)][1] for r in range(rows)]
    dx = (x_vals[-1] - x_vals[0]) / max(cols - 1, 1)
    dy = (y_vals[-1] - y_vals[0]) / max(rows - 1, 1)
    half_w = int(round(dx / 2.0))
    half_h = int(round(dy / 2.0))

    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for (r, c), (cx_abs, cy_abs) in centers_abs.items():
        cx = cx_abs - lawn_region["left"]
        cy = cy_abs - lawn_region["top"]
        x0 = max(0, cx - half_w)
        x1 = min(lawn_region["width"], cx + half_w)
        y0 = max(0, cy - half_h)
        y1 = min(lawn_region["height"], cy + half_h)
        boxes[(r, c)] = (x0, y0, x1, y1)
    return boxes


def y_to_row(y: float, lawn_height: int) -> int:
    if lawn_height <= 0:
        return 0
    row_h = lawn_height / NUM_ROWS
    return max(0, min(NUM_ROWS - 1, int(y // row_h)))


def detect_zombies(
    model: YOLO, lawn_bgr: np.ndarray, conf: float, imgsz: int
) -> tuple[list[ZombieObs], dict[str, int], list[tuple[float, float]], list[tuple[float, float, str]]]:
    result = model.predict(lawn_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    zombies: list[ZombieObs] = []
    raw_class_counts: dict[str, int] = {}
    plant_points: list[tuple[float, float]] = []
    typed_plant_dets: list[tuple[float, float, str]] = []

    for box in result.boxes:
        cls_id = int(box.cls[0])
        raw_name = str(model.names[cls_id])
        raw_class_counts[raw_name] = raw_class_counts.get(raw_name, 0) + 1
        x0, y0, x1, y1 = map(float, box.xyxy[0].tolist())
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

        if is_plant_label(raw_name):
            plant_points.append((cx, cy))
        plant_type = normalize_plant_label(raw_name)
        if plant_type is not None:
            typed_plant_dets.append((cx, cy, plant_type))

        zombie_type = normalize_label(raw_name)
        if zombie_type is None:
            continue

        row = y_to_row(cy, lawn_bgr.shape[0])
        zombies.append(ZombieObs(row=row, x=cx, zombie_type=zombie_type))

    return zombies, raw_class_counts, plant_points, typed_plant_dets


def load_cnn_model(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    if "empty" not in class_to_idx:
        raise ValueError("CNN class_to_idx must contain 'empty' class for occupancy mode.")
    empty_idx = int(class_to_idx["empty"])
    num_classes = len(class_to_idx)
    img_size = int(ckpt.get("img_size", 96))

    model = CellStateCNN(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    return model, tf, empty_idx, device


def classify_board(
    lawn_bgr: np.ndarray,
    cell_boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    cnn_model: CellStateCNN,
    tf,
    empty_idx: int,
    device):
    crops = []
    rc_order = []
    for r in range(NUM_ROWS):
        for c in range(NUM_COLS):
            x0, y0, x1, y1 = cell_boxes[(r, c)]
            crop_bgr = lawn_bgr[y0:y1, x0:x1]
            crop_rgb = crop_bgr[:, :, ::-1]
            crops.append(tf(crop_rgb))
            rc_order.append((r, c))

    batch = torch.stack(crops, dim=0).to(device, non_blocking=True)
    with torch.no_grad():
        logits = cnn_model(batch)
        probs = torch.softmax(logits, dim=1)
        conf_vals, pred = torch.max(probs, dim=1)

    board: dict[tuple[int, int], str] = {}
    non_empty_conf: dict[tuple[int, int], float] = {}
    empty_conf: dict[tuple[int, int], float] = {}
    for i, (r, c) in enumerate(rc_order):
        pred_idx = int(pred[i].item())
        conf = float(conf_vals[i].item())
        is_empty = pred_idx == empty_idx
        if (not is_empty) and conf < OCCUPANCY_MIN_CONF:
            is_empty = True
        board[(r, c)] = "empty" if is_empty else "non_empty"
        non_empty_conf[(r, c)] = 0.0 if is_empty else conf
        empty_conf[(r, c)] = conf if is_empty else 0.0
    return board, non_empty_conf, empty_conf


def build_yolo_occupancy_board(
    plant_points: list[tuple[float, float]],
    cell_boxes: dict[tuple[int, int], tuple[int, int, int, int]],
) -> dict[tuple[int, int], str]:
    board = {(r, c): "empty" for r in range(NUM_ROWS) for c in range(NUM_COLS)}
    for cx, cy in plant_points:
        x = int(cx)
        y = int(cy)
        for (r, c), (x0, y0, x1, y1) in cell_boxes.items():
            if x0 <= x < x1 and y0 <= y < y1:
                board[(r, c)] = "non_empty"
                break
    return board


def build_yolo_typed_cell_map(
    typed_plant_dets: list[tuple[float, float, str]],
    cell_boxes: dict[tuple[int, int], tuple[int, int, int, int]],
) -> dict[tuple[int, int], list[str]]:
    cell_map: dict[tuple[int, int], list[str]] = {}
    for cx, cy, plant_type in typed_plant_dets:
        x = int(cx)
        y = int(cy)
        for (r, c), (x0, y0, x1, y1) in cell_boxes.items():
            if x0 <= x < x1 and y0 <= y < y1:
                cell_map.setdefault((r, c), []).append(plant_type)
                break
    return cell_map


def _dist2(a: tuple[int, int], b: tuple[int, int]) -> int:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _decay_cell_ttl(cell_ttl: dict[tuple[int, int], int]) -> None:
    for rc in list(cell_ttl.keys()):
        cell_ttl[rc] -= 1
        if cell_ttl[rc] <= 0:
            cell_ttl.pop(rc, None)


def collect_suns(
    lawn_bgr: np.ndarray,
    lawn_region: dict[str, int],
    recent_click: tuple[int, int] | None = None,
) -> tuple[int, int, tuple[int, int] | None]:
    centers = detect_sun(lawn_bgr)
    if not centers:
        return 0, 0, None

    clicked: list[tuple[int, int]] = []
    min_dist2 = SUN_DUPLICATE_DIST * SUN_DUPLICATE_DIST
    recent_dist2 = SUN_RECENT_BLOCK_DIST * SUN_RECENT_BLOCK_DIST
    collected = 0
    last_clicked: tuple[int, int] | None = None

    for x_local, y_local in centers:
        x_abs = lawn_region["left"] + int(x_local)
        y_abs = lawn_region["top"] + int(y_local)

        if recent_click is not None and _dist2((x_abs, y_abs), recent_click) < recent_dist2:
            continue

        if any(_dist2((x_abs, y_abs), p) < min_dist2 for p in clicked):
            continue

        pyautogui.click(x_abs, y_abs)
        clicked.append((x_abs, y_abs))
        last_clicked = (x_abs, y_abs)
        collected += 1
        time.sleep(SUN_CLICK_PAUSE)

    return len(centers), collected, last_clicked


def run(yolo_path: str, cnn_path: str, conf: float, imgsz: int, show: bool = False):
    selected_yolo_path = yolo_path
    if yolo_path.endswith(".onnx") and not Path(yolo_path).exists() and Path(FALLBACK_YOLO_PT).exists():
        selected_yolo_path = FALLBACK_YOLO_PT
    yolo = YOLO(selected_yolo_path)
    print(f"YOLO model: {selected_yolo_path}", flush=True)
    cnn_model, tf, empty_idx, device = load_cnn_model(cnn_path)

    lawn_region = load_lawn_region()
    cell_boxes = build_cell_boxes_lawn_local(lawn_region)
    slot_centers = load_seed_slot_centers()
    cell_centers = build_cell_centers_absolute()
    actions = ActionExecutor(slot_centers=slot_centers, cell_centers=cell_centers)

    state = GameState()
    state.new_game()
    pending_plant: tuple[str, int, int, float, float] | None = None
    pending_confirm_streak = 0
    placed_plants: dict[tuple[int, int], str] = {}
    sun_credit_buffer = 0.0
    recent_plant_click_ttl: dict[tuple[int, int], int] = {}
    recent_sun_click: tuple[int, int] | None = None
    recent_sun_click_ttl = 0
    zombie_hold_row: int | None = None
    zombie_hold_ttl = 0
    occ_streak: dict[tuple[int, int], int] = {(r, c): 0 for r in range(NUM_ROWS) for c in range(NUM_COLS)}
    empty_streak: dict[tuple[int, int], int] = {(r, c): 0 for r in range(NUM_ROWS) for c in range(NUM_COLS)}
    board: dict[tuple[int, int], str] = {(r, c): "empty" for r in range(NUM_ROWS) for c in range(NUM_COLS)}

    print("Agent starts in 3 seconds...")
    for sec in [3, 2, 1]:
        print(f"{sec}...", flush=True)
        time.sleep(1.0)

    if show:
        cv2.namedWindow("PvZ Agent Debug (press q to quit)", cv2.WINDOW_NORMAL)

    with mss() as sct:
        last_t = time.time()
        while True:
            now = time.time()
            dt = now - last_t
            last_t = now
            state.tick(dt)
            _decay_cell_ttl(recent_plant_click_ttl)
            if recent_sun_click_ttl > 0:
                recent_sun_click_ttl -= 1
                if recent_sun_click_ttl <= 0:
                    recent_sun_click = None

            frame_bgr = np.array(sct.grab(lawn_region))[:, :, :3]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            zombies_bgr, raw_bgr, plants_bgr, typed_plants_bgr = detect_zombies(yolo, frame_bgr, conf, imgsz)
            zombies_rgb, raw_rgb, plants_rgb, typed_plants_rgb = detect_zombies(yolo, frame_rgb, conf, imgsz)
            if len(zombies_rgb) > len(zombies_bgr):
                zombies = zombies_rgb
                raw_yolo_counts = raw_rgb
                yolo_plant_points = plants_rgb
                yolo_typed_plants = typed_plants_rgb
                yolo_frame_mode = "rgb"
            else:
                zombies = zombies_bgr
                raw_yolo_counts = raw_bgr
                yolo_plant_points = plants_bgr
                yolo_typed_plants = typed_plants_bgr
                yolo_frame_mode = "bgr"

            yolo_occ_board = build_yolo_occupancy_board(yolo_plant_points, cell_boxes)
            yolo_typed_cell_map = build_yolo_typed_cell_map(yolo_typed_plants, cell_boxes)
            cnn_board, raw_non_empty_conf, raw_empty_conf = classify_board(
                frame_bgr, cell_boxes, cnn_model, tf, empty_idx, device
            )
            raw_board = {(r, c): "empty" for r in range(NUM_ROWS) for c in range(NUM_COLS)}
            for rc in raw_board.keys():
                yolo_occ = yolo_occ_board.get(rc, "empty") == "non_empty"
                cnn_occ = cnn_board.get(rc, "empty") == "non_empty"
                cnn_occ_conf = raw_non_empty_conf.get(rc, 0.0)
                cnn_empty_conf = raw_empty_conf.get(rc, 0.0)

                occupied = yolo_occ
                if not yolo_occ and cnn_occ and cnn_occ_conf >= CNN_SOFT_OCCUPY_CONF:
                    occupied = True
                if yolo_occ and (not cnn_occ) and cnn_empty_conf >= CNN_SOFT_EMPTY_VETO_CONF:
                    occupied = False
                raw_board[rc] = "non_empty" if occupied else "empty"

            for rc, raw_label in raw_board.items():
                if raw_label == "non_empty":
                    occ_streak[rc] += 1
                    empty_streak[rc] = 0
                else:
                    empty_streak[rc] += 1
                    occ_streak[rc] = 0

                if occ_streak[rc] >= OCCUPANCY_CONFIRM_FRAMES:
                    board[rc] = "non_empty"
                elif empty_streak[rc] >= OCCUPANCY_CONFIRM_FRAMES:
                    board[rc] = "empty"

            if pending_plant is not None:
                p_name, p_row, p_col, commit_at, timeout_at = pending_plant
                if now >= commit_at:
                    cell_types = yolo_typed_cell_map.get((p_row, p_col), [])
                    is_type_confirmed = p_name in cell_types
                    if is_type_confirmed:
                        pending_confirm_streak += 1
                    else:
                        pending_confirm_streak = 0

                    if pending_confirm_streak >= PLANT_CONFIRM_FRAMES:
                        state.spend_sun(p_name)
                        state.start_cooldown(p_name)
                        placed_plants[(p_row, p_col)] = p_name
                        recent_plant_click_ttl[(p_row, p_col)] = PLANT_CLICK_BLOCK_FRAMES
                        pending_plant = None
                        pending_confirm_streak = 0
                    elif now >= timeout_at:
                        pending_plant = None
                        pending_confirm_streak = 0

            frame_for_sun = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
            sun_detected, collected, sun_clicked_pos = collect_suns(
                frame_for_sun,
                lawn_region,
                recent_click=recent_sun_click,
            )
            if sun_clicked_pos is not None:
                recent_sun_click = sun_clicked_pos
                recent_sun_click_ttl = SUN_RECENT_BLOCK_FRAMES
            if collected > 0:
                sun_credit_buffer += collected * SUN_CREDIT_RATIO
                credited = int(sun_credit_buffer)
                if credited > 0:
                    state.add_sun(credited)
                    sun_credit_buffer -= credited

            for (r, c) in list(placed_plants.keys()):
                if board.get((r, c), "empty") == "empty":
                    placed_plants.pop((r, c), None)

            plants_for_scoring = [
                PlantObs(row=r, col=c, plant_type=plant_type)
                for (r, c), plant_type in placed_plants.items()
            ]

            threat, defense, gaps = compute_lane_gap(zombies, plants_for_scoring, num_rows=NUM_ROWS)
            row = pick_priority_row(zombies, gaps)
            if len(zombies) > 0:
                presence_row = pick_zombie_presence_row(zombies, threat)
                if (
                    row is None
                    and presence_row is not None
                    and gaps[presence_row] >= COMBAT_FALLBACK_MIN_GAP
                ):
                    row = presence_row
                if row is not None and gaps[row] >= COMBAT_FALLBACK_MIN_GAP:
                    zombie_hold_row = row
                    zombie_hold_ttl = ZOMBIE_PRESENCE_HOLD_FRAMES
            elif row is None and zombie_hold_ttl > 0 and zombie_hold_row is not None:
                if gaps[zombie_hold_row] >= COMBAT_FALLBACK_MIN_GAP:
                    row = zombie_hold_row
                    zombie_hold_ttl -= 1
                else:
                    zombie_hold_row = None
                    zombie_hold_ttl = 0
            elif zombie_hold_ttl > 0:
                zombie_hold_ttl -= 1
                if zombie_hold_ttl <= 0:
                    zombie_hold_row = None
            zombie_counts = [0] * NUM_ROWS
            for z in zombies:
                if 0 <= z.row < NUM_ROWS:
                    zombie_counts[z.row] += 1

            # Decision occupancy is based on confirmed local state to avoid false CNN blocks.
            decision_board = {(r, c): "empty" for r in range(NUM_ROWS) for c in range(NUM_COLS)}
            for (r, c), _p in placed_plants.items():
                decision_board[(r, c)] = "non_empty"
            for (r, c) in recent_plant_click_ttl.keys():
                decision_board[(r, c)] = "non_empty"
            if pending_plant is not None:
                _pp, pr, pc, _pt, _to = pending_plant
                decision_board[(pr, pc)] = "non_empty"

            sunflower_count = sum(1 for p in placed_plants.values() if p == "sun_flower")
            farm_col1 = [board.get((r, 0), "?") for r in range(NUM_ROWS)]
            farm_col2 = [board.get((r, 1), "?") for r in range(NUM_ROWS)]
            farm_col5 = [board.get((r, 4), "?") for r in range(NUM_ROWS)]
            farm_col4 = [board.get((r, 3), "?") for r in range(NUM_ROWS)]
            dec_col1 = [decision_board.get((r, 0), "?") for r in range(NUM_ROWS)]
            dec_col2 = [decision_board.get((r, 1), "?") for r in range(NUM_ROWS)]
            dec_col5 = [decision_board.get((r, 4), "?") for r in range(NUM_ROWS)]
            decision = choose_plant_decision(
                priority_row=row,
                gaps=gaps,
                board=decision_board,
                state=state,
                sunflower_count=sunflower_count,
                num_rows=NUM_ROWS,
            )

            if decision is not None and pending_plant is None:
                plant, target_row, col = decision
                if (target_row, col) in recent_plant_click_ttl:
                    decision = None
                else:
                    actions.plant(plant, target_row, col)
                    pending_plant = (
                        plant,
                        target_row,
                        col,
                        now + PLANT_COMMIT_DELAY_SEC,
                        now + PLANT_CONFIRM_TIMEOUT_SEC,
                    )
                    pending_confirm_streak = 0
                    time.sleep(0.1)

            print(
                f"frame={state.frame_index} sun={state.sun} zombies={len(zombies)} "
                f"sunflower_count={sunflower_count} cd_sunflower={state.cd_remaining['sun_flower']:.2f} "
                f"sun_detected={sun_detected} sun_clicked={collected} "
                f"zombie_counts={zombie_counts} threat={threat} "
                f"raw_yolo={raw_yolo_counts} yolo_mode={yolo_frame_mode} "
                f"yolo_plants={len(yolo_plant_points)} "
                f"col1={farm_col1} col2={farm_col2} col4={farm_col4} col5={farm_col5} "
                f"dec_col1={dec_col1} dec_col2={dec_col2} dec_col5={dec_col5} "
                f"gaps={gaps} target_row={row} decision={decision} pending={pending_plant} "
                f"placed={len(placed_plants)} "
                f"plant_blocked={len(recent_plant_click_ttl)} "
                f"sun_credit_buf={sun_credit_buffer:.2f} sun_credit_ratio={SUN_CREDIT_RATIO} "
                f"sun_recent={recent_sun_click} sun_recent_ttl={recent_sun_click_ttl} "
                f"z_hold_row={zombie_hold_row} z_hold_ttl={zombie_hold_ttl} "
                f"fallback_min_gap={COMBAT_FALLBACK_MIN_GAP} "
                f"typed_plants={len(yolo_typed_plants)} pend_ok={pending_confirm_streak}/{PLANT_CONFIRM_FRAMES} "
                f"target_row_occ={([board.get((row, c), '?') for c in SHOOTER_COLS] if row is not None else [])} "
                f"target_row_dec={([decision_board.get((row, c), '?') for c in SHOOTER_COLS] if row is not None else [])} "
                f"target_row_raw={([raw_board.get((row, c), '?') for c in SHOOTER_COLS] if row is not None else [])} "
                f"target_row_conf={([round(raw_non_empty_conf.get((row, c), 0.0), 2) for c in SHOOTER_COLS] if row is not None else [])} "
                f"guard_conf=off"
            , flush=True)

            if show:
                vis = frame_bgr.copy()
                row_h = vis.shape[0] / NUM_ROWS
                for i in range(1, NUM_ROWS):
                    y = int(i * row_h)
                    cv2.line(vis, (0, y), (vis.shape[1], y), (0, 255, 255), 1)

                for z in zombies:
                    zx = int(z.x)
                    zy = int((z.row + 0.5) * row_h)
                    cv2.circle(vis, (zx, zy), 6, (0, 0, 255), -1)

                cv2.putText(
                    vis,
                    f"sun={state.sun} zombies={len(zombies)} row={row} decision={decision}",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("PvZ Agent Debug", vis)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    if show:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="PvZ game loop: YOLO zombie threat + CNN cell occupancy + planting policy."
    )
    parser.add_argument("--yolo", default=DEFAULT_YOLO, help="Path to YOLO .pt/.onnx model")
    parser.add_argument("--cnn", default=DEFAULT_CNN, help="Path to CNN checkpoint")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--imgsz", type=int, default=416, help="YOLO inference image size")
    parser.add_argument("--show", action="store_true", help="Show live debug preview")
    args = parser.parse_args()

    run(yolo_path=args.yolo, cnn_path=args.cnn, conf=args.conf, imgsz=args.imgsz, show=args.show)


if __name__ == "__main__":
    main()