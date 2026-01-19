import argparse
import os 
import datetime 
import random
import shutil
from PIL import Image
import hashlib


#TODO don't copy empty folders 
parser = argparse.ArgumentParser(description="Combines parallel runs")
parser.add_argument('-s','--save', type=str, default='', help='Folder names from parallel_generation to combine, comma separtated')
parser.add_argument('-rm' ,'--remove_originals', action='store_true', help='Remove original folders after combining')

#if there isn't the same structure map0, map1 coresponding to map0,map1 etc.
parser.add_argument('-cp', '--completly_diffrent',action='store_true', help='If the maps in the folders are completly diffrent skip checking for same maps')

args = parser.parse_args()


def merge_same_photos(output_folder,folders):
    maps = {}
    for folder in folders:
        for sub in os.listdir(os.path.join('data', folder)):
            file = [file for file in os.listdir(os.path.join('data', folder, sub)) if file.endswith(('png','jpg'))]
            if len(file) > 0:
                file = file[0]
                map = Image.open(os.path.join('data', folder, sub, file)).convert('L')
                map_hash = hashlib.md5(map.tobytes()).hexdigest()
                if map_hash not in maps:
                    maps[map_hash] = [os.path.join('data', folder, sub, file)]
                for file in os.listdir(os.path.join('data', folder, sub)):
                    if file.endswith(('json')):
                        maps[map_hash].append(os.path.join('data', folder, sub, file))

    #TODO skip empty maps 
    for i,map_hash in enumerate(maps.keys()):
        output_subfolder = os.path.join(output_folder, f'map_{i+1}')
        os.makedirs(output_subfolder, exist_ok=True)

        shutil.copy2(maps[map_hash][0], os.path.join(output_subfolder, 'map.png'))

        for j, path in enumerate(maps[map_hash][1:]):
            shutil.copy2(path, os.path.join(output_subfolder, f'path_{j}.json'))


#Just copies whole folders 
def completly_diffrent(output_folder,folders):
    i = 0 
    for folder in folders:
        for sub in os.listdir(os.path.join('data', folder)):
            shutil.copytree(os.path.join('data', folder, sub), os.path.join(output_folder, f'map_{i}'), dirs_exist_ok=True)
            i += 1

#TODO add  parameter and implement checking for the same maps 

if __name__ == "__main__":

    if args.save == '':
        print("Please provide output folder name using -s")
        exit(1)
    

    #TODO ? Check if number of maps is correct 
    



    folders = []
    for output_name in args.save.split(','):
        folders.extend([x for x in os.listdir('data') if x.startswith(output_name)])


        
  
    output_folder = os.path.join('data', f"{args.save.split(',')[0]}_{datetime.datetime.now().strftime('%d-%m-%Y_%H:%M')}")
    os.makedirs(output_folder, exist_ok=True)

    print('Merging folders: ',folders,' into ',output_folder)


    if args.completly_diffrent:
        completly_diffrent(output_folder, folders)
    else:
        merge_same_photos(output_folder, folders)

    if args.remove_originals:
        for folder in folders:
            shutil.rmtree(os.path.join('data', folder))