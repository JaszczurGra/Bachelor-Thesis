import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageDraw
import random
import math
if __name__ == "__main__":
    from base_map_gen import BaseMapGenerator
else:
    from .base_map_gen import BaseMapGenerator




class MapGenerator(BaseMapGenerator):
    def __init__(self, width=300, height=300, safe_radius=0.2, start_pos=(0.1, 0.1), stop_pos=(.9,.1),coverage_percent=0.15, 
                     rect_size=(15/300, 40/300),circle_size=(15/300, 40/300) ):
        super().__init__(width, height, safe_radius, start_pos, stop_pos)
        self.coverage_percent = coverage_percent
        self.rect_size = rect_size
        self.circle_size = circle_size


    def _dist_point_to_rect(self, px, py, rx, ry, rw, rh):
        closest_x = max(rx, min(px, rx + rw))
        closest_y = max(ry, min(py, ry + rh))
        dx = px - closest_x
        dy = py - closest_y
        return math.hypot(dx, dy)


    def generate(self):
        img = Image.new('L', (self.width, self.height), 255)
        draw = ImageDraw.Draw(img)
        grid_mask = np.zeros((self.height, self.width), dtype=bool)


        w = min(random.uniform(.1*self.width,0*self.width - self.safe_radius *2 - self.start_pos[0] + self.stop_pos[0]), 0.2*self.width)
        x = random.uniform(self.safe_radius + self.start_pos[0] , self.stop_pos[0] - w - self.safe_radius) 
        h = self.random_scaled_h(0.3,0.4)
        draw.rectangle([x, self.height - h, x+w, self.height], fill=0)
        grid_mask[self.height - h:self.height, int(x):int(x+w)] = True


        target_obstacle_pixels = self.width * self.height * self.coverage_percent
        current_obstacle_pixels = 0
        
        attempts = 0
        while current_obstacle_pixels < target_obstacle_pixels and attempts < 3000:
            attempts += 1
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            if attempts % 2: 
                w = self.random_scaled_w(self.rect_size[0], self.rect_size[1])
                h = self.random_scaled_h(self.rect_size[0], self.rect_size[1])
                d_start = self._dist_point_to_rect(self.start_pos[0], self.start_pos[1], x, y, w, h)
                d_stop = self._dist_point_to_rect(self.stop_pos[0], self.stop_pos[1], x, y, w, h)
                if d_start < self.safe_radius or d_stop < self.safe_radius:
                    continue 

                draw.rectangle([x, y, x+w, y+h], fill=0)
                grid_mask[y:y+h, x:x+w] = True
            else:
                r = self.random_scaled_w(self.circle_size[0], self.circle_size[1]) // 2
                if (x-self.start_pos[0])**2 + (y-self.start_pos[1])**2 < (self.safe_radius+r)**2 or (x-self.stop_pos[0])**2 + (y-self.stop_pos[1])**2 < (self.safe_radius+r)**2:
                    continue
                draw.ellipse([x - r, y - r, x + r, y + r], fill=0)
                yy, xx = np.ogrid[:self.height, :self.width]
                circle_mask = (xx - x)**2 + (yy - y)**2 <= r**2
                grid_mask |= circle_mask
                current_obstacle_pixels = np.count_nonzero(grid_mask)


            current_obstacle_pixels = np.count_nonzero(grid_mask)

        return np.array(img, dtype=np.uint8)



if __name__ == "__main__":
    gen = NoiseMapGenerator(1000, 1000)
    
    print(f"Previewing random rectangle maps...")
    
    map_img = gen.generate()
        
    plt.figure(figsize=(6,6))
    plt.imshow(np.array(map_img), cmap='gray')
    plt.title(f"Strict Safe Zones")
    start_circle = Circle(gen.start_pos, gen.safe_radius, color='r', fill=False, linewidth=2)
    stop_circle = Circle(gen.stop_pos, gen.safe_radius, color='r', fill=False, linewidth=2)
    plt.gca().add_patch(start_circle)
    plt.gca().add_patch(stop_circle)
    print(gen.start_pos, gen.stop_pos, gen.safe_radius)
    plt.show()