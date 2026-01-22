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
    def __init__(self, width=300, height=300, safe_radius=0.1, start_pos=(0.1, 0.1), stop_pos=(.9,.1) ):
        super().__init__(width, height, safe_radius, start_pos, stop_pos)


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


        #top 

        

        bot_lip = self.random_scaled_h(0.4,0.5)
        top_lip = bot_lip + self.random_scaled_h(0.1,0.15)

        l = self.start_pos[0] + self.safe_radius
        r = self.stop_pos[0] - self.safe_radius


        draw.rectangle([0, top_lip, l, self.height], fill=0)
        draw.rectangle([r, top_lip, self.width, self.height], fill=0)

        w = self.random_scaled_w(0.03,0.06)
        draw.rectangle([l, 0, l+w, bot_lip], fill=0)
        draw.rectangle([l, top_lip, l+w, self.height], fill=0)
        draw.rectangle([r-w, 0, r, bot_lip], fill=0)
        draw.rectangle([r-w, top_lip, r, self.height], fill=0)

        draw.rectangle([l, top_lip + (self.height - top_lip)//2, r, self.height], fill=0)
        draw.rectangle([l, 0, r, bot_lip // 2], fill=0)

        min_y = self.height - self.start_pos[1] + self.safe_radius
        draw.rectangle([self.safe_radius, min_y, l, bot_lip], fill=0)
        draw.rectangle([r, min_y, self.width - self.safe_radius, bot_lip], fill=0)
        

        draw.circle([(l+r)// 2, (bot_lip + top_lip) // 2], ((r - l) * random.uniform(0.3, 0.6))  // 2 , fill=0)


        # w = min(random.uniform(.1*self.width,0*self.width - self.safe_radius *2 - self.start_pos[0] + self.stop_pos[0]), 0.2*self.width)
        # x = random.uniform(self.safe_radius + self.start_pos[0] , self.stop_pos[0] - w - self.safe_radius) 
        # h = self.random_scaled_h(0.3,0.4)
        # draw.rectangle([x, self.height - h, x+w, self.height], fill=0)
        # grid_mask[self.height - h:self.height, int(x):int(x+w)] = True


        # target_obstacle_pixels = self.width * self.height * self.coverage_percent
        # current_obstacle_pixels = 0
        
    

        return np.array(img, dtype=np.uint8)[::-1,:]



if __name__ == "__main__":
    gen = MapGenerator(1000, 1000)
    
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