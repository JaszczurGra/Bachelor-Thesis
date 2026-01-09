import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
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
                 vertical_offset=0.05,
                 arc_configs=None):

        super().__init__(width, height, safe_radius, start_pos, stop_pos)
        
        self.vertical_offset = vertical_offset
        
        # Default arc configurations: [(h_min, h_max, w_min, w_max), ...]
        self.arc_configs = arc_configs if arc_configs is not None else [
            (45/300, 65/300, 10/300, 12/300),   # Arc 1
            (130/300, 160/300, 12/300, 20/300), # Arc 2
            (230/300, 250/300, 25/300, 50/300) ] # Arc 3

        # Center and radius base for arcs
        self.cx = self.width / 2
        self.cy = self.height - 1
        self.rx_base = self.width / 2

    def _create_arc_layer(self, ry, thickness):
        """Create a single arc layer."""
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

    def generate(self):
        offset = int(self.vertical_offset * self.height)
        final_grid = np.zeros((self.height, self.width), dtype=np.uint8)

        # Bottom line
        line_h = self.random_scaled_h(5/300, 8/300)
        final_grid[self.height - line_h : self.height, :] = 255

        # Generate arcs based on configuration
        for h_min, h_max, w_min, w_max in self.arc_configs:
            arc_height = self.random_scaled_h(h_min, h_max)
            arc_width = self.random_scaled_h(w_min, w_max)
            arc_layer = self._create_arc_layer(ry=arc_height, thickness=arc_width)
            final_grid = np.maximum(final_grid, arc_layer)
        
        # Apply vertical offset
        final_grid = np.concatenate([
            final_grid[offset:, :], 
            np.zeros((offset, self.width), dtype=np.uint8)
        ], axis=0)

        # final_grid = np.maximum(final_grid, self.generate_save_radi())
        final_grid[:,0:self.safe_radius] = 255
        final_grid[:,-self.safe_radius:] = 255

        return final_grid

if __name__ == "__main__":
    gen = ArcMapGenerator(2300,2300)
    
    print(f"Previewing car width maps...")
    
    map_data = gen.generate()
        
    plt.figure(figsize=(6,6))
    plt.imshow(map_data, cmap='gray')
    plt.title("Car width map")
    plt.show()