from dataclasses import dataclass, field

PLANT_COST = {"sun_flower" : 50, "pea_shooter" : 100, "snow_pea" : 175, "wall_nut" : 50}
PLANT_CD = {"sun_flower" : 7.5, "pea_shooter" : 7.5, "snow_pea" : 7.5, "wall_nut" : 30.0}
CONFIRM_FRAMES = 2

@dataclass(frozen=True)
class ZombieObs:
    row: int
    x: float
    zombie_type: str 

@dataclass(frozen=True)
class PlantObs:
    row: int
    col: int
    plant_type: str

@dataclass
class GameState():
    sun: int = 50
    frame_index: int = 0
    cd_remaining: dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in PLANT_CD})

    def new_game(self):
        self.sun = 50
        self.frame_index = 0
        for key in self.cd_remaining:
            self.cd_remaining[key] = 0.0
    
    def tick(self, dt: float):
        if dt <= 0:
            return
        self.frame_index += 1
        for key in self.cd_remaining:
            self.cd_remaining[key] = max(0.0, self.cd_remaining[key] - dt)
    
    def can_afford(self, plant: str):
        return (self.sun >= PLANT_COST[plant])
    
    def is_ready(self, plant: str):
        return (self.cd_remaining[plant] == 0.0)

    def spend_sun(self, plant: str):
        self.sun -= PLANT_COST[plant]
    
    def start_cooldown(self, plant: str):
        self.cd_remaining[plant] = PLANT_CD[plant]
    
    def add_sun(self, count: int = 1):
        self.sun += count * 50
