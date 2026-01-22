import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image, ImageDraw
import os
import random
import time
import math


class BaseMapGenerator:
    def __init__(self, width=300, height=300, safe_radius=2.5 * 1.41 / 15, start_pos=(0, 0),stop_pos=(1,0)):
        self.width = width
        self.height = height
        self.start_pos = start_pos
        self.stop_pos = stop_pos
        #TODO scale to both axes? 
        self.safe_radius = int(safe_radius * self.width)
        self.start_pos = (int(self.start_pos[0] * self.width), int((1-self.start_pos[1]) * self.height))
        self.stop_pos = (int(self.stop_pos[0] * self.width), int((1-self.stop_pos[1]) * self.height))

    def random_scaled_h(self, l, h ):
        return random.randint(int(l * self.height), int(h * self.height))
    def random_scaled_w(self, l, h ):
        return random.randint(int(l * self.width), int(h * self.width))

    def scale_h(self, val):
        return int(val * self.height)
    def scale_w(self, val):
        return int(val * self.width)


    def save_map(self, img, folder, filename):
        if not os.path.exists(folder):
            os.makedirs(folder)
        img = Image.fromarray(img.astype(np.uint8))
        img.save(os.path.join(folder, filename))
        print(f"Saved {filename}")

    def generate_save_radi(self):
        print('adfkladsflkjasdlfkj',end='\n\n\n\n')
        layer = Image.new('L', (self.width, self.height), 0)
        draw = ImageDraw.Draw(layer)
        draw.ellipse([
            self.start_pos[0] - self.safe_radius, self.start_pos[1] - self.safe_radius,
            self.start_pos[0] + self.safe_radius, self.start_pos[1] + self.safe_radius
        ], fill=255)
        
        draw.ellipse([
            self.stop_pos[0] - self.safe_radius, self.stop_pos[1] - self.safe_radius,
            self.stop_pos[0] + self.safe_radius, self.stop_pos[1] + self.safe_radius
        ], fill=255)
        return np.array(layer)
    
    def generate_and_save(self, filename, folder):
        self.save_map(self.generate(), folder, filename)
 


    def generate(self):
        return 

