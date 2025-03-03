from ursina import *
from ursina.prefabs.editor_camera import EditorCamera

app = Ursina()

editor_camera = EditorCamera()

grid_size = 10
for x in range(grid_size):
    for z in range(grid_size):
        for y in range(grid_size):
            Entity(model='cube', color=color.azure, position=(x,y,z))

app.run()
