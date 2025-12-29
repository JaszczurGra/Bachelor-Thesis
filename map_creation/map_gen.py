
import argparse
import time

parser = argparse.ArgumentParser(description="Parallel OMPL Car Planners")

parser.add_argument('-s', type=int, default=5, help='Number of size maps')
parser.add_argument('-t', type=int, default=5,  help='Number of turning radius maps')
parser.add_argument('-o', type=str, default='results', help='Output folder for maps')
parser.add_argument('--size', type=int, default=(300,300), nargs=2, help='Map dimensions (not used)')


if __name__ == "__main__":
    args = parser.parse_args()

    from size import MapGenerator as SizeMapGen
    from turning import MapGenerator as TurnMapGen
    import os

    if not os.path.exists('maps'):
        os.mkdir('maps')
    output_folder = os.path.join('maps',args.o)

    size_gen = SizeMapGen(args.size[0],args.size[1])
    turn_gen = TurnMapGen(args.size[0],args.size[1])

    print(f"Generating {args.s} size maps...")
    for i in range(args.s):
        map_data = size_gen.generate_map()
        filename = f"size_map_{i}.png"
        size_gen.save_map(map_data, output_folder, filename)

    print(f"Generating {args.t} turning radius maps...")
    for i in range(args.t):
        map_data = turn_gen.generate_map()
        filename = f"turn_map_{i}.png"
        turn_gen.save_map(map_data, output_folder, filename)