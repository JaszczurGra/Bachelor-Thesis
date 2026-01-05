import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import random
import time


class MapGenerator:
    def __init__(self, w=300, h=300, wall_thickness=8/300):
        self.width = w
        self.height = h
        self.wall_thickness =  int ( wall_thickness * w)
        self.mid_y = h // 2
        
        self.divider_gap = 40

    def random_scaled_h(self, l, h ):
        return random.randint(int(l * self.height), int(h * self.height))
    def random_scaled_w(self, l, h ):
        return random.randint(int(l * self.width), int(h * self.width))


    def _draw_chicane(self, grid, row_start, row_end, difficulty):
        mid_x = self.width // 2
        
        if difficulty == 'hard':
            # Hard gap
            gap_vertical = self.random_scaled_h(40/300, 50/300)
            spacing_x = self.random_scaled_w(35/300, 45/300)
        else:
            # Easy gap
            gap_vertical = self.random_scaled_h(55/300, 70/300)
            spacing_x = self.random_scaled_w(60/300, 75/300) 

        # Jitter
        jitter = self.random_scaled_w(-15/300, 15/300)
        center_x = mid_x + jitter

        grid[row_start : row_end - gap_vertical, 
             center_x - spacing_x : center_x - spacing_x + self.wall_thickness] = 0
             
        grid[row_start + gap_vertical : row_end, 
             center_x : center_x + self.wall_thickness] = 0
             
        grid[row_start : row_end - gap_vertical, 
             center_x + spacing_x : center_x + spacing_x + self.wall_thickness] = 0

    def generate_map(self):
        grid = np.ones((self.width, self.height), dtype=np.uint8)


        div_start_y = self.mid_y - (self.wall_thickness // 2)
        div_end_y   = self.mid_y + (self.wall_thickness // 2)
        
        grid[div_start_y:div_end_y, self.divider_gap : -self.divider_gap] = 0

        self._draw_chicane(grid, row_start=0, row_end=div_start_y, difficulty='easy')
        
        self._draw_chicane(grid, row_start=div_end_y, row_end=self.height, difficulty='hard')

        return grid

    def save_map(self, map_data, folder, filename):
        if not os.path.exists(folder):
            os.makedirs(folder)     
        img_data = map_data * 255
        img = Image.fromarray(img_data.astype(np.uint8), mode='L')
        img.save(os.path.join(folder, filename))
        print(f"Saved {filename}")

if __name__ == "__main__":
    gen = MapGenerator(1300,1300)

    print("Previewing turn radius map...")

    map_data = gen.generate_map()

    plt.figure(figsize=(6, 6))
    plt.imshow(map_data, cmap='gray')
    plt.title("Turn radius map")
    plt.show()