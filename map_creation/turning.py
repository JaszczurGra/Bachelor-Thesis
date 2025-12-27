import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import time

class MapGenerator:
    def __init__(self, size=300, wall_thickness=10):
        self.size = size
        self.wall_thickness = wall_thickness
        self.mid_y = size // 2
        
        self.divider_gap = 40

    def _draw_chicane(self, grid, row_start, row_end, difficulty):
        mid_x = self.size // 2
        
        if difficulty == 'hard':
            # Hard gap
            gap_vertical = random.randint(40, 50)
            spacing_x = random.randint(35, 45)
        else:
            # Easy gap
            gap_vertical = random.randint(55, 70)
            spacing_x = random.randint(60, 75) 

        # Jitter
        jitter = random.randint(-15, 15)
        center_x = mid_x + jitter

        grid[row_start : row_end - gap_vertical, 
             center_x - spacing_x : center_x - spacing_x + self.wall_thickness] = 0
             
        grid[row_start + gap_vertical : row_end, 
             center_x : center_x + self.wall_thickness] = 0
             
        grid[row_start : row_end - gap_vertical, 
             center_x + spacing_x : center_x + spacing_x + self.wall_thickness] = 0

    def generate_map(self):
        grid = np.ones((self.size, self.size), dtype=np.uint8)

        grid[0:5, :] = 0
        grid[-5:, :] = 0
        
        div_start_y = self.mid_y - (self.wall_thickness // 2)
        div_end_y   = self.mid_y + (self.wall_thickness // 2)
        
        grid[div_start_y:div_end_y, self.divider_gap : -self.divider_gap] = 0

        self._draw_chicane(grid, row_start=0, row_end=div_start_y, difficulty='easy')
        
        self._draw_chicane(grid, row_start=div_end_y, row_end=self.size, difficulty='hard')

        return grid

    def save_map(self, map_data, folder, filename):
        if not os.path.exists(folder):
            os.makedirs(folder)     
        img_data = map_data * 255
        img = Image.fromarray(img_data.astype(np.uint8), mode='L')
        img.save(os.path.join(folder, filename))
        print(f"Saved {filename}")

if __name__ == "__main__":
    OUTPUT_FOLDER = "dataset_turn_rad"
    NUM_MAPS = 5

    gen = MapGenerator(size=300)

    for i in range(NUM_MAPS):
        map_data = gen.generate_map()
        
        if i == 0:
            plt.figure(figsize=(6, 6))
            plt.imshow(map_data, cmap='gray')
            plt.title("Turn radius map")
            plt.show()

        fname = f"map_{i}_{int(time.time())}.png"
        gen.save_map(map_data, OUTPUT_FOLDER, fname)