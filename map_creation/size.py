import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import random
import time

class MapGenerator:
    def __init__(self, w=300, h=300):
        self.width = w
        self.height = h
        
        self.cx = self.width / 2
        self.cy = self.height - 1
        
        self.rx_base = self.width / 2.5

    def _create_arc_layer(self, ry, thickness):
        layer = Image.new('L', (self.width, self.height), 0)
        draw = ImageDraw.Draw(layer)
        
        rx_outer = self.rx_base + thickness / 2
        ry_outer = ry + thickness / 2
        
        bbox_outer = [
            self.cx - rx_outer, self.cy - ry_outer,
            self.cx + rx_outer, self.cy + ry_outer
        ]
        draw.ellipse(bbox_outer, fill=255)
        
        rx_inner = self.rx_base - thickness / 2
        ry_inner = ry - thickness / 2
        
        bbox_inner = [
            self.cx - rx_inner, self.cy - ry_inner,
            self.cx + rx_inner, self.cy + ry_inner
        ]
        draw.ellipse(bbox_inner, fill=0)
        
        return np.array(layer)

    def generate_map(self):
        final_grid = np.zeros((self.height, self.width), dtype=np.uint8)

        # Bottom line
        line_h = random.randint(5, 10)
        final_grid[self.height - line_h : self.height, :] = 255

        # Arc 1
        h1 = random.randint(45, 65)
        w1 = random.randint(10, 15)
        layer1 = self._create_arc_layer(ry=h1, thickness=w1)
        
        # Arc 2
        h2 = random.randint(130, 160)
        w2 = random.randint(15, 25)
        layer2 = self._create_arc_layer(ry=h2, thickness=w2)
        
        # Arc 3
        h3 = random.randint(230, 260)
        w3 = random.randint(30, 50)
        layer3 = self._create_arc_layer(ry=h3, thickness=w3)

        final_grid = np.maximum(final_grid, layer1)
        final_grid = np.maximum(final_grid, layer2)
        final_grid = np.maximum(final_grid, layer3)

        return final_grid

    def save_map(self, map_data, folder, filename):
        if not os.path.exists(folder):
            os.makedirs(folder)
        img = Image.fromarray(map_data.astype(np.uint8), mode='L')
        img.save(os.path.join(folder, filename))
        print(f"Saved {filename}")

if __name__ == "__main__":
    gen = MapGenerator(300,300)
    
    print(f"Previewing car width maps...")
    
    map_data = gen.generate_map()
        
    plt.figure(figsize=(6,6))
    plt.imshow(map_data, cmap='gray')
    plt.title("Car width map")
    plt.show()