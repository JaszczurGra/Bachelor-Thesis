
import argparse
import time
import os

from map_types.size import MapGenerator as SizeMapGen
from map_types.turning import MapGenerator as TurnMapGen
from map_types.noise import MapGenerator as NoiseMapGen

parser = argparse.ArgumentParser(description="Parallel OMPL Car Planners")

parser.add_argument('-s','--size', type=int, default=5, help='Number of size maps')
parser.add_argument('-t','--turning', type=int, default=5, help='Number of turning radius maps')
parser.add_argument('-r','--random', type=int, default=5, help='Number of random rectangle maps')
parser.add_argument('-o','--output', type=str, default='results', help='Output folder for maps')
parser.add_argument('--res', type=int, default=(300,300), nargs=2, help='Map dimensions (not used)')

#TODO change map geneatros to base on one class and have them generate save zones based on (start,end, Radius ) * self.width
#TODO don't overwrite maps if they exist just add more 
if __name__ == "__main__":
    args = parser.parse_args()


    

    output_folder = os.path.join('maps',args.output)
    os.makedirs(output_folder,exist_ok=True)

    R = 1.5/15 
    start_pos = (.1,.1)
    stop_pos = (.9,.1)

    generators = {
        'size': SizeMapGen(args.res[0], args.res[1], R, start_pos, stop_pos),
        'turning': TurnMapGen(args.res[0], args.res[1], R, start_pos, stop_pos),
        'random': NoiseMapGen(args.res[0], args.res[1], R, start_pos, stop_pos)
    }

    for key, value in vars(args).items():
        if key in generators:
            for i in range(value):
                last = max([int(f.split('_')[-1].strip('.png')) for f in os.listdir(output_folder) if f.startswith(f'{key}_map_')]+[0])
                filename = f"{key}_map_{last + 1}.png"
                generators[key].generate_and_save(filename, output_folder)
