import argparse
import os 
import datetime 
import random
import shutil

parser = argparse.ArgumentParser(description="Combines parallel runs")
parser.add_argument('-s','--save', type=str, default='', help='Folder names from parallel_generation to combine, comma separtated')
parser.add_argument('-rm' ,'--remove_originals', action='store_true', help='Remove original folders after combining')
args = parser.parse_args()




if __name__ == "__main__":

    if args.save == '':
        print("Please provide output folder name using -s")
        exit(1)
    

    #TODO ? Check if number of maps is correct 
    


    # Copy map_0 completely


    folders = []
    for output_name in args.save.split(','):
        folders.extend([x for x in os.listdir('data') if x.startswith(output_name)])

    paths = {}
    maps = {}

    for folder in folders:
        for sub in os.listdir(os.path.join('data', folder)):
            for file in os.listdir(os.path.join('data', folder, sub)):
                if file.endswith('.json'):
                    if sub not in paths:
                        paths[sub] = []
                    paths[sub].append(os.path.join('data', folder, sub, file))
                if sub not in maps and file.endswith(('png','jpg')):
                    maps[sub] = os.path.join('data', folder, sub, file)
      

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H:%M')
    


    adjectives = ["dizzy", "zippy", "bouncy", "quirky", "snappy", "jazzy", "peppy", "zesty", "spicy", "witty"]
    animals = ["falcon", "mongoose", "penguin", "otter", "badger", "porcupine", "lemur", "ferret", "quail", "lynx"]

    funny_name = f"{random.choice(adjectives)}_{random.choice(animals)}"
    output_name = f"{funny_name}_{timestamp}"


    output_folder = os.path.join('data', output_name)
    os.makedirs(output_folder, exist_ok=True)


    #maps instead of paths if we want empty maps too 
    for sub in maps:
        output_subfolder = os.path.join(output_folder, sub)
        os.makedirs(output_subfolder, exist_ok=True)

      
        shutil.copy2(maps[sub], os.path.join(output_subfolder, os.path.basename(maps[sub])))


        if sub in paths:
            for j, path in enumerate(paths[sub]):
                shutil.copy2(path, os.path.join(output_subfolder, f'path_{j}.json'))


    if args.remove_originals:
        for folder in folders:
            shutil.rmtree(os.path.join('data', folder))