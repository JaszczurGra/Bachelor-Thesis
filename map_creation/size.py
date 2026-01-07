import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import random
import time


#TODO bigger r 
R = 2.5 * 1.41 / 15
VERTICAL_OFFSET = 1.5 / 15 
class MapGenerator:
    def __init__(self, w=300, h=300):
        self.width = w
        self.height = h
        
        self.cx = self.width / 2
        self.cy = self.height - 1
        
        self.rx_base = self.width / 2

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

    def _create_save_zone_layers(self, radius):
        layer = Image.new('L', (self.width, self.height), 0)
        draw = ImageDraw.Draw(layer)
        
        draw.ellipse([-radius, self.height - radius, radius, self.height+radius], fill=255)
        draw.ellipse([self.width - radius, self.height - radius, self.width+radius, self.height+radius], fill=255)
        
        return np.array(layer)

    def random_scaled_h(self, l, h ):
        return random.randint(int(l * self.height), int(h * self.height))

    def generate_map(self):

        offset =  int(VERTICAL_OFFSET * self.height)
        final_grid = np.zeros((self.height, self.width), dtype=np.uint8)

        # Bottom line
        line_h = self.random_scaled_h(5/300,8/300)
        final_grid[self.height - line_h : self.height, :] = 255

        # Arc 1
        h1 = self.random_scaled_h(45/300, 65/300)
        w1 = self.random_scaled_h(10/300, 15/300)
        layer1 = self._create_arc_layer(ry=h1, thickness=w1)
        
        # Arc 2
        h2 = self.random_scaled_h(130/300, 160/300)
        w2 = self.random_scaled_h(15/300, 25/300)
        layer2 = self._create_arc_layer(ry=h2, thickness=w2)
        
        # Arc 3
        h3 = self.random_scaled_h(230/300, 260/300)
        w3 = self.random_scaled_h(30/300, 50/300)
        layer3 = self._create_arc_layer(ry=h3, thickness=w3)
        
        save_zones = self._create_save_zone_layers(radius=R*self.width)

        final_grid = np.maximum(final_grid, layer1)
        final_grid = np.maximum(final_grid, layer2)
        final_grid = np.maximum(final_grid, layer3)

        final_grid = np.concatenate([final_grid[offset:, :], np.zeros((offset, self.width), dtype=np.uint8)], axis=0)

        final_grid = np.maximum(final_grid, save_zones)

        return final_grid

    def save_map(self, map_data, folder, filename):
        if not os.path.exists(folder):
            os.makedirs(folder)
        img = Image.fromarray(map_data.astype(np.uint8), mode='L')
        img.save(os.path.join(folder, filename))
        print(f"Saved {filename}")

if __name__ == "__main__":
    gen = MapGenerator(2300,2300)
    
    print(f"Previewing car width maps...")
    
    map_data = gen.generate_map()
        
    plt.figure(figsize=(6,6))
    plt.imshow(map_data, cmap='gray')
    plt.title("Car width map")
    plt.show()