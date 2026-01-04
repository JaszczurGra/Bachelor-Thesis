import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageDraw
import os
import random
import time
import math



class MapGenerator:
    def __init__(self, w=300, h=300):
        self.width = w
        self.height = h
        
        self.start_pos = (0, self.height - 1)
        self.stop_pos = (self.width - 1, self.height - 1)

    def _dist_point_to_rect(self, px, py, rx, ry, rw, rh):
        closest_x = max(rx, min(px, rx + rw))
        
        closest_y = max(ry, min(py, ry + rh))
        
        dx = px - closest_x
        dy = py - closest_y
        
        return math.hypot(dx, dy)

    # Change parameters here
    def generate_map(self, 
                     coverage_percent=0.2, 
                     min_rect_size=40, 
                     max_rect_size=60,
                     safe_radius=60):
        
        self.safe_radius = safe_radius
        img = Image.new('L', (self.width, self.height), 255)
        draw = ImageDraw.Draw(img)

        grid_mask = np.zeros((self.height, self.width), dtype=bool)
        
        # Triangle
        tri_base_width = 60
        tri_height = 120
        mid_x = self.width // 2
        
        triangle_points = [
            (mid_x - tri_base_width, self.height),
            (mid_x + tri_base_width, self.height),
            (mid_x, self.height - tri_height)
        ]
        draw.polygon(triangle_points, fill=0)

        # Rectangles
        total_pixels = self.width * self.height
        target_obstacle_pixels = total_pixels * coverage_percent
        current_obstacle_pixels = 0
        
        attempts = 0
        max_attempts = 5000 
        
        while current_obstacle_pixels < target_obstacle_pixels and attempts < max_attempts:
            attempts += 1
            
            w = random.randint(min_rect_size, max_rect_size)
            h = random.randint(min_rect_size, max_rect_size)
            x = random.randint(0, self.width - w)
            y = random.randint(0, self.height - h)
            
            d_start = self._dist_point_to_rect(self.start_pos[0], self.start_pos[1], x, y, w, h)
            
            d_stop = self._dist_point_to_rect(self.stop_pos[0], self.stop_pos[1], x, y, w, h)
            
            if d_start < safe_radius or d_stop < safe_radius:
                continue 

            draw.rectangle([x, y, x+w, y+h], fill=0)
            grid_mask[y:y+h, x:x+w] = True
            current_obstacle_pixels = np.count_nonzero(grid_mask)
        return img

    def save_map(self, img, folder, filename):
        if not os.path.exists(folder):
            os.makedirs(folder)
        img.save(os.path.join(folder, filename))
        print(f"Saved {filename}")

if __name__ == "__main__":
    gen = MapGenerator(300, 300)
    
    print(f"Previewing random rectangle maps...")
    
    map_img = gen.generate_map()
        
    plt.figure(figsize=(6,6))
    plt.imshow(np.array(map_img), cmap='gray')
    plt.title(f"Strict Safe Zones")
    start_circle = Circle(gen.start_pos, gen.safe_radius, color='r', fill=False, linewidth=2)
    stop_circle = Circle(gen.stop_pos, gen.safe_radius, color='r', fill=False, linewidth=2)
    plt.gca().add_patch(start_circle)
    plt.gca().add_patch(stop_circle)
    plt.show()