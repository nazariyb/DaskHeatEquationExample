from ursina import *
from ursina.prefabs.editor_camera import EditorCamera
import heat_equation as he
import numpy as np

app = Ursina()

editor_camera = EditorCamera()

grid_size = 15
cubes = [[[None for _ in range(grid_size)] for _ in range(grid_size)] for _ in range(grid_size)]
for x in range(grid_size):
    for z in range(grid_size):
        for y in range(grid_size):
            cubes[x][y][z] = Entity(model='cube', color=color.blue, position=(x,y,z))

heat_calculator = he.HeatEquation(0.5, 10, 100, nodes=grid_size+2)

def lerp_color(start_color, end_color, t):
    return color.rgb(
        start_color.r + (end_color.r - start_color.r) * t,
        start_color.g + (end_color.g - start_color.g) * t,
        start_color.b + (end_color.b - start_color.b) * t
    )

def update():
    new_temps = heat_calculator.calculate(time.dt / 10)
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                cubes[x][y][z].color = lerp_color(color.blue, color.red, new_temps[x + 1, y + 1, z + 1] / 200)

app.run()
