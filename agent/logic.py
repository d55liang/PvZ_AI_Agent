from agent.state import GameState, PlantObs, ZombieObs

NUM_ROWS = 5

ZOMBIE_THREAT = {"ordinary" : 1.0, "flag" : 1.0, "conehead" : 2.0, "pole_vaulting" : 2.0, "buckethead" : 3.5}
PLANT_DEFENSE = {"sun_flower" : 0, "pea_shooter" : 1.25, "snow_pea" : 2.0, "wall_nut" : 3.0}
PLANT_THRESHOLD = 1.0

# Placement columns (0-indexed)
# Human column mapping:
# - sunflower primary: col 1 -> index 0
# - sunflower secondary: col 5 -> index 4
# - wall-nut: col 6 -> index 5
SUNFLOWER_COL = 0
PEA_COL = 1
WALLNUT_COL = 5
SUNFLOWER_FARM_PRIMARY_COL = 0
SUNFLOWER_FARM_SECONDARY_COL = 4
SHOOTER_COLS = [1, 2, 3]  # human columns 2, 3, 4
MIN_SUNFLOWERS_FOR_SNOW = 5
MIN_SUN_FOR_SNOW = 200


def is_cell_empty(board: dict[tuple[int, int], str], row: int, col: int) -> bool:
    return board.get((row, col), "empty") == "empty"

def compute_row_threat(zombies: list[ZombieObs], num_rows: int = NUM_ROWS):
    threat = [0.0] * num_rows
    for zombie in zombies:
        if 0 <= zombie.row < num_rows:
            threat[zombie.row] += ZOMBIE_THREAT.get(zombie.zombie_type, 0.0)
    return threat

def compute_row_defense(plants: list[PlantObs], num_rows: int = NUM_ROWS):
    defense = [0.0] * num_rows
    for plant in plants:
        if 0 <= plant.row < num_rows:
            defense[plant.row] += PLANT_DEFENSE.get(plant.plant_type, 0.0)
    return defense

def compute_lane_gap(zombies: list[ZombieObs], plants: list[PlantObs], num_rows: int = NUM_ROWS):
    threat = compute_row_threat(zombies, num_rows)
    defense = compute_row_defense(plants, num_rows)
    gap = []
    for row in range(num_rows):
        gap.append(threat[row] - defense[row])
    return threat, defense, gap

def pick_priority_row(zombies: list[ZombieObs], gaps: list[float]):
    candidates = []
    for row, gap in enumerate(gaps):
        if gap >= PLANT_THRESHOLD:
            candidates.append(row)
    if not candidates:
        return None
    
    max_gap = max(gaps[row] for row in candidates)
    tied = []
    for row, gap in enumerate(gaps):
        if gap == max_gap:
            tied.append(row)
    if len(tied) == 1:
        return tied[0]

    best_row = None
    best_dist = float("inf")
    for row in tied:
        zombie_x = []
        for zombie in zombies:
            if zombie.row == row:
                zombie_x.append(zombie.x)
        if not zombie_x:
            continue
        # Tie-break by nearest zombie to the left edge (x = 0).
        dist = min(zombie_x)
        if dist < best_dist:
            best_row = row
            best_dist = dist
    
    if best_row is not None:
        return best_row
    else:
        return tied[0]

def pick_zombie_presence_row(zombies: list[ZombieObs], threat: list[float]):
    # Fallback when no row passes gap threshold: if zombies exist, force a combat row.
    candidate_rows = [r for r, t in enumerate(threat) if t > 0.0]
    if not candidate_rows:
        return None

    max_threat = max(threat[r] for r in candidate_rows)
    tied = [r for r in candidate_rows if threat[r] == max_threat]
    if len(tied) == 1:
        return tied[0]

    best_row = None
    best_dist = float("inf")
    for row in tied:
        xs = [z.x for z in zombies if z.row == row]
        if not xs:
            continue
        dist = min(xs)
        if dist < best_dist:
            best_dist = dist
            best_row = row
    return best_row if best_row is not None else tied[0]


def choose_shooter_for_row(
    row: int,
    board: dict[tuple[int, int], str],
    state: GameState,
    plant_type: str,
) -> tuple[str, int] | None:
    if not (state.can_afford(plant_type) and state.is_ready(plant_type)):
        return None
    for col in SHOOTER_COLS:
        if is_cell_empty(board, row, col):
            return (plant_type, col)
    return None


def choose_sunflower_farm_decision(
    board: dict[tuple[int, int], str],
    state: GameState,
    num_rows: int = NUM_ROWS):
    if not (state.can_afford("sun_flower") and state.is_ready("sun_flower")):
        return None

    for row in range(num_rows):
        if is_cell_empty(board, row, SUNFLOWER_FARM_PRIMARY_COL):
            return ("sun_flower", row, SUNFLOWER_FARM_PRIMARY_COL)

    for row in range(num_rows):
        if is_cell_empty(board, row, SUNFLOWER_FARM_SECONDARY_COL):
            return ("sun_flower", row, SUNFLOWER_FARM_SECONDARY_COL)

    return None


def choose_plant_decision(
    priority_row: int | None,
    gaps: list[float],
    board: dict[tuple[int, int], str],
    state: GameState,
    sunflower_count: int,
    num_rows: int = NUM_ROWS):
    # Build sunflower economy only while there is no unbalanced threat.
    if priority_row is None:
        return choose_sunflower_farm_decision(board, state, num_rows)

    # If more than one row is high-threat, use pea_shooter; otherwise use snow_pea.
    high_threat_rows = [r for r, g in enumerate(gaps) if g >= PLANT_THRESHOLD]
    preferred_plant = "pea_shooter" if len(high_threat_rows) > 1 else "snow_pea"
    if preferred_plant == "snow_pea" and sunflower_count < MIN_SUNFLOWERS_FOR_SNOW:
        preferred_plant = "pea_shooter"
    if preferred_plant == "snow_pea" and state.sun <= MIN_SUN_FOR_SNOW:
        preferred_plant = "pea_shooter"

    # Occupancy-only mode: if column 2 is fully occupied, allow wall-nut in threat row.
    col2_filled = all(not is_cell_empty(board, r, PEA_COL) for r in range(num_rows))
    if (
        priority_row is not None
        and col2_filled
        and state.can_afford("wall_nut")
        and state.is_ready("wall_nut")
    ):
        if is_cell_empty(board, priority_row, WALLNUT_COL):
            return ("wall_nut", priority_row, WALLNUT_COL)

    # Threat mode: place only in the selected priority row.
    choice = choose_shooter_for_row(priority_row, board, state, preferred_plant)
    if choice is not None:
        plant, col = choice
        return (plant, priority_row, col)

    return None
