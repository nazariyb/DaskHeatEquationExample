import sys
import numpy as np
import dask.array as da

from ursina import *
from ursina.prefabs.editor_camera import EditorCamera
from dask.distributed import Client


# Startup
client = Client("tcp://192.168.0.162:8786")
print(client)

worker_to_remove = []
client.retire_workers(workers=worker_to_remove, close_workers=True)

app = Ursina(borderless=False, development_mode=True, fullscreen=False)
window.windowed_size = Vec2(1280, 720)
window.fullscreen_size = Vec2(1280, 720)
window.update_aspect_ratio()

editor_camera = EditorCamera()

current_temps = None
pending_temps = None

grid_size = 8 * 4
vis_grid_size = grid_size - 2


class TempCube(Entity):
    def __init__(self, **kwargs):
        super().__init__(model='cube', collider='box', **kwargs)

    def input(self, key):
        global pending_temps
        if self.hovered and key == 'left mouse down' and pending_temps is not None:
            pending_temps[int(self.x) + 1, int(self.y) + 1, int(self.z) + 1] = 100

# Visual entities
cubes = [[[None for _ in range(vis_grid_size)] for _ in range(vis_grid_size)] for _ in range(vis_grid_size)]
for x in range(vis_grid_size):
    for z in range(vis_grid_size):
        for y in range(vis_grid_size):
            cubes[x][y][z] = TempCube(color=color.blue, position=(x,y,z)) if (x == 0 or x == vis_grid_size - 1 or y == 0 or y == vis_grid_size - 1 or z == 0 or z == vis_grid_size - 1) else None


def lerp_color(start_color, end_color, t):
    return color.rgb(
        start_color.r + (end_color.r - start_color.r) * t,
        start_color.g + (end_color.g - start_color.g) * t,
        start_color.b + (end_color.b - start_color.b) * t
    )


class ThermalFeatures:
    def __init__(self, rho, capacity, coef):
        '''
        rho: density in kg/m^3
        capacity: specific heat capacity in J/(kg*K)
        coef: thermal conductivity in W/(m*K)
        '''
        self.rho = rho # kg/m^3
        self.capacity = capacity # J/(kg*K)
        self.coef = coef * 100 # W/(m*K)
    
    def calculate_a(self) -> float:
        return self.coef / (self.rho * self.capacity)

features = {
    "cuprum": ThermalFeatures(rho=8960, capacity=385, coef=401),
    "aluminium": ThermalFeatures(rho=2700, capacity=900, coef=237),
    "iron": ThermalFeatures(rho=7870, capacity=450, coef=80),
    "glass": ThermalFeatures(rho=2500, capacity=840, coef=1.05),
    "water": ThermalFeatures(rho=1000, capacity=4186, coef=0.6),
    "air": ThermalFeatures(rho=1.225, capacity=1005, coef=0.025),
}


class Params:
    def __init__(self, features : ThermalFeatures, nodes: int = 10, length: float = 50):
        self.a = features.calculate_a()
        self.nodes = nodes
        self.u = np.zeros((self.nodes, self.nodes, self.nodes))
        
        self.dx = length / self.nodes
        self.dy = length / self.nodes
        self.dz = length / self.nodes

        self.u[0, :, :] = 100
        self.u[1, :, :] = 100
        self.u[-2, :, :] = 100

params = Params(features["air"], nodes=grid_size, length=50)


def update_block(block, params : Params, delta_t : float):
    import numpy as np
    MAX_TIME_DELTA = 10.0

    while delta_t > 0:
        current_delta_t = min(MAX_TIME_DELTA, delta_t)
        delta_t -= current_delta_t
        new_block = block.copy()
        inner = np.s_[1:-1, 1:-1, 1:-1]
        dd_ux = (block[2:, 1:-1, 1:-1] - 2 * block[1:-1, 1:-1, 1:-1] + block[:-2, 1:-1, 1:-1]) / params.dx**2
        dd_uy = (block[1:-1, 2:, 1:-1] - 2 * block[1:-1, 1:-1, 1:-1] + block[1:-1, :-2, 1:-1]) / params.dy**2
        dd_uz = (block[1:-1, 1:-1, 2:] - 2 * block[1:-1, 1:-1, 1:-1] + block[1:-1, 1:-1, :-2]) / params.dz**2
        derivative = dd_ux + dd_uy + dd_uz
        new_block[inner] = block[inner] + params.a * current_delta_t * derivative
    return new_block

future = None
def update():
    global current_temps, pending_temps, future, params

    base_data = current_temps if current_temps is not None else params.u

    if future is None:
        if pending_temps is not None:
            mask = pending_temps != 0
            base_data[mask] = pending_temps[mask]
            pending_temps[mask] = 0

        u_dask = da.from_array(base_data, chunks=(8, grid_size, grid_size))
        u_dask_new = u_dask.map_overlap(
            update_block, params=params, delta_t=time.dt*1000, depth=(1,1,1), boundary='nearest', trim=True)
        future = client.compute(u_dask_new)
    elif future.done():
        current_temps = future.result()
        future = None

    # print(f"Avg temp: {current_temps.mean()}, Max temp: {current_temps.max()}, Min temp: {current_temps.min()}")

    if current_temps is not None:
        if pending_temps is None:
            pending_temps = np.zeros(current_temps.shape)

        for x in range(vis_grid_size):
            for y in range(vis_grid_size):
                for z in range(vis_grid_size):
                    if cubes[x][y][z] is not None:
                        cubes[x][y][z].color = lerp_color(color.blue, color.red, current_temps[x + 1, y + 1, z + 1] / 100)

app.run()
