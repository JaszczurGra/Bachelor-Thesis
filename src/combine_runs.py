import argparse
import os 
import datetime 
import random
import shutil

parser = argparse.ArgumentParser(description="Combines parallel runs")
parser.add_argument('-s','--save', type=str, default='', help='Folder names from parallel_generation to combine, comma separtated')
parser.add_argument('-rm' ,'--remove_originals', action='store_true', help='Remove original folders after combining')

#if there isn't the same structure map0, map1 coresponding to map0,map1 etc.
parser.add_argument('-cp', '--completly_diffrent',action='store_true', help='If the maps in the folders are completly diffrent skip checking for same maps')

args = parser.parse_args()

def same_structure(output_folder,folders):
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
      
    for sub in maps:
            output_subfolder = os.path.join(output_folder, sub)
            os.makedirs(output_subfolder, exist_ok=True)

        
            shutil.copy2(maps[sub], os.path.join(output_subfolder, os.path.basename(maps[sub])))


            if sub in paths:
                for j, path in enumerate(paths[sub]):
                    shutil.copy2(path, os.path.join(output_subfolder, f'path_{j}.json'))
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


    if args.completly_diffrent:
        completly_diffrent(output_folder, folders)
    else:
        same_structure(output_folder, folders)

    if args.remove_originals:
        for folder in folders:
            shutil.rmtree(os.path.join('data', folder))