import os
import random
from PIL import Image, ImageDraw    


#Map of viable spaces 

class MapGenerator:
    def __init__(self, num_rect_obstacles=(5, 9), num_circle_obstacles=(0, 0), 
                 image_size=(500, 500)):
        self.num_rect_obstacles = num_rect_obstacles
        self.num_circle_obstacles = num_circle_obstacles
        self.obstacles = []

        self.image_size = image_size
        
    
    def generate_map(self):
        img = Image.new('1', self.image_size, color=1)  # 'L' = grayscale, 255 = white
        draw = ImageDraw.Draw(img)
        
        # Generate and draw random rectangles
        num_rects = random.randint(*self.num_rect_obstacles)
        for _ in range(num_rects):
            width = random.uniform(0.05, 0.1) * self.image_size[0]
            height = random.uniform(0.05, 0.1) * self.image_size[1]
            x = random.uniform(0, self.image_size[0] - width)
            y = random.uniform(0, self.image_size[1] - height)

            draw.rectangle([x, y, x + width, y + height], fill=0, outline=0)  # 0 = black
        
        # Generate and draw random circles
        num_circles = random.randint(*self.num_circle_obstacles)
        for _ in range(num_circles):
            radius = random.uniform(0.03, 0.1) * self.image_size[0] # Scale radius to image size

            x = random.uniform(0, self.image_size[0] - 2 * radius)
            y = random.uniform(0, self.image_size[1] - 2 * radius)
            
            draw.ellipse([x, y, x + 2 * radius, y + 2 * radius],
                        fill=0, outline=0)  # 0 = black
        
        return img




    def generate_maps(self, num_maps=1, output_dir=None):
        """Generate multiple maps."""
        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(base_dir, 'maps')

            os.makedirs(output_dir, exist_ok=True)
        

        adjectives = ['happy', 'sleepy', 'grumpy', 'bouncy', 'fuzzy', 'shiny', 'crazy', 'lazy', 'dizzy', 'quirky']
        nouns = ['panda', 'penguin', 'dragon', 'unicorn', 'robot', 'ninja', 'wizard', 'tiger', 'falcon', 'phoenix']
        
        # Generate unique folder name
        adjective = random.choice(adjectives)
        noun = random.choice(nouns)
        base_name = f"{adjective}_{noun}"
        
        # Count existing folders with same base name
        existing_folders = [d for d in os.listdir(output_dir) 
                        if os.path.isdir(os.path.join(output_dir, d)) and d.startswith(base_name)]
        
        if existing_folders:
            # Extract numbers from existing folders like "happy_panda_001"
            numbers = []
            for folder in existing_folders:
                try:
                    num = int(folder.split('_')[-1])
                    numbers.append(num)
                except ValueError:
                    pass
            next_num = max(numbers) + 1 if numbers else 1
        else:
            next_num = 1
        
        # Create folder name with counter
        output_dir = os.path.join(output_dir, f"{base_name}_{next_num:03d}")
        os.makedirs(output_dir, exist_ok=True)

        
        for i in range(num_maps):

            #TODO :03d be log10(num_maps) + 1 
            filename = f'map_{i+1:03d}'
            img = self.generate_map()
            filepath = os.path.join(output_dir, f'{filename}.png')
            img.save(filepath)
            

#
        # return maps_data
        
        
    

if __name__ == "__main__":
    generator = MapGenerator(num_rect_obstacles=(5, 10), num_circle_obstacles=(1, 3))
    generator.generate_maps(num_maps=10)
