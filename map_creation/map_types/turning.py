import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import time
if __name__ == "__main__":
    from base_map_gen import BaseMapGenerator
else:
    from .base_map_gen import BaseMapGenerator

class MapGenerator(BaseMapGenerator):
    def __init__(self, width=300, height=300, safe_radius=2.5 * 1.41 / 15,
                 start_pos=(0, 0), stop_pos=(1, 0),
                 wall_thickness=3/300,
                 divider_gap_range=(40/300, 60/300),
                 easy_gap_range=(55/300, 70/300),
                 easy_spacing_range=(35/300, 45/300),
                 hard_gap_range=(40/300, 50/300),
                 hard_spacing_range=(15/300, 20/300)):

        super().__init__(width, height, safe_radius, start_pos, stop_pos)
        
        self.wall_thickness = int(wall_thickness * width)
        self.mid_y = height // 2
        
        self.divider_gap = self.random_scaled_h(*divider_gap_range)
        self.easy_gap_range = easy_gap_range
        self.easy_spacing_range = easy_spacing_range
        self.hard_gap_range = hard_gap_range
        self.hard_spacing_range = hard_spacing_range
        
        # Random divider gap
        self.divider_gap = self.random_scaled_h(*divider_gap_range)


    def _draw_chicane(self, grid, n, y, gap,spacing):
        mid_x = self.width // 2
        grid[ int(y[0]): int(y[1]) - self.random_scaled_h(*gap),mid_x - self.wall_thickness // 2 : mid_x + self.wall_thickness // 2] = 0
        last_offset, current_offset = 0,0 
        for i in range(n - 1):
            current_offset,last_offset =int((i % 2 - 0.5) * 2  * (abs(last_offset) +  self.random_scaled_w(*spacing))),current_offset
            up = i  // 2 % 2
            # print ( y[0]  + self.random_scaled_h(*gap) * (1-up):y [1]  - self.random_scaled_h(*gap) * up, mid_x - self.wall_thickness // 2 + current_offset : mid_x + self.wall_thickness // 2 + current_offset )
            grid[ int(y[0]  + self.random_scaled_h(*gap) * (1-up)) :int(y[1]  - self.random_scaled_h(*gap) * up), mid_x - self.wall_thickness // 2 + current_offset : mid_x + self.wall_thickness // 2 + current_offset] = 0

    def generate(self):
        grid = np.ones((self.height, self.width), dtype=np.uint8) 

        grid[self.mid_y - self.divider_gap // 2 : self.mid_y + self.divider_gap // 2 , self.divider_gap : -self.divider_gap] = 0

        self._draw_chicane(grid, 5 , (0, self.height//2 - self.divider_gap // 2)  , self.easy_gap_range, self.easy_spacing_range)
        self._draw_chicane(grid, 3 , (self.height // 2 + self.divider_gap // 2 , self.height), self.hard_gap_range, self.hard_spacing_range)

        return grid * 255 

if __name__ == "__main__":

    gen = MapGenerator(1300,1300)

    print("Previewing turn radius map...")

    map_data = gen.generate()

    plt.figure(figsize=(6, 6))
    plt.imshow(map_data, cmap='gray')
    plt.title("Turn radius map")
    plt.show()