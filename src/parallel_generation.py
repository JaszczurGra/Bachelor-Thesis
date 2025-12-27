import io
import os
from RRT_acceleration import CarOMPL_acceleration
from STRRT_acceleration import SSTCarOMPL_acceleration
from RRT import RRT_Planer
from STRRT import STRRT_Planer
from  Dubins import Dubins_pathfinding
from Pacejka import Pacejka_pathfinding,PacejkaRectangleRobot   
import time 
from multiprocessing import Pool, cpu_count,Manager
from base_pathfind_classes import Robot,RectangleRobot
import argparse
import random
from datetime import datetime
import json

from visualizer import Visualizer

from PIL import Image
import numpy as np
import math

#TODO reverse the map color 0 free 1 occupied now is inverse ?
#TODO should the structure be map and paths or link to map 

#TODO instead of time as name of saving name like for maps 
#TODO robot setmap could be optimazed if the same map is used, no it couldnt as the w,h changes 

#TODO bounds automatic by scale of img 
#TODO should the velocity weight be 1 or smaller 
parser = argparse.ArgumentParser(description="Parallel OMPL Car Planners")
parser.add_argument('-n', '--num_threads', type=int, default=4, help='Number of parallel planner threads')
parser.add_argument('-r', '--runs_per_planner', type=int, default=5, help='Number of runs per planner')
parser.add_argument('-t', '--max_runtime', type=float, default=3.0, help='Maximum runtime per planner run (seconds)')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
parser.add_argument('--vis', action='store_true', help='Enable visualization of planning process')
parser.add_argument('--save', action='store_true', help='Save solved paths to file')
parser.add_argument('--map', type=str, default=None, help='Path to load map from file')

args = parser.parse_args()


def run_planner_continuous(planner_id, max_runtime, result_list, stop_event, runs_per_planner, save_dir=None, maps=None, map_indexes=None):
    """Run planner continuously and store results at specific slot."""
    run_count = 0
    
    while run_count < runs_per_planner and not stop_event.is_set():

        start_time = time.time()
        

        if maps is not None and map_indexes is not None:
            map_data = maps[map_indexes[planner_id][run_count]]
        else:
            map_data = np.ones((50,50))
        # obstacles = [RectangleObstacle(random.uniform(0,10), random.uniform(0,10), random.uniform(0.5,2), random.uniform(0.5,2)) for i in range(random.randint(5,9))]
 
        # robot = RectangleRobot( )
        # robot.width = random.uniform(0.01,0.5)
        # robot.lenght = random.uniform(0.01,0.5) 

        # robot.width = 1
        # robot.lenght = 1


        # robot.radius = random.uniform(0.2,0.6)
        # robot.wheelbase = robot.lenght * 0.5
        # robot.max_velocity = 5.0 + random.uniform(-1.0,1.0)
        # robot.acceleration = 4
        # robot.max_steering_at_zero_v = random.uniform(math.pi / 10.0, math.pi / 6.0)
        # robot.max_steering_at_max_v = random.uniform(math.pi / 20.0, math.pi / 12.0)
        # robot.max_steering_at_zero_v = 0
        # robot.max_steering_at_max_v = 0
        
        robot=RectangleRobot(0.5,1.0,collision_check_angle_res=8)

        # car_planner = SSTCarOMPL_acceleration(robot=robot,map=map_data,start=(2.0,2.0,1),goal=(8.0,2.0,0),pos_treshold=0.5,max_runtime=max_runtime, vel_threshold=1, velocity_weight=0.15)
        # car_planner = Dubins_pathfinding(robot=robot,map=map_data,start=(2.0,2.0,1),goal=(8.0,2.0,0),max_runtime=max_runtime,bounds=(10,10))

        car_planner = Pacejka_pathfinding(max_runtime=max_runtime, map=map_data,robot =PacejkaRectangleRobot(0.5,1.0,max_velocity=200),vel_threshold=0.5,velocity_weight=0.0,start=(2.0,2.0,0.0),goal=(8.0,2.0,0.0), bounds=(10,10))

        if args.verbose:
            print(f"[Planner {planner_id}] Starting run #{run_count + 1}")
        # car_planner = CarOMPL_acceleration(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5,max_runtime=max_runtime)
        # car_planner = SSTCarOMPL_acceleration(robot=robot,map=map_data,start=(1.0,1.0),goal=(9.0,9.0),pos_treshold=0.5,max_runtime=max_runtime, vel_threshold=1, velocity_weight=0.1)
        # car_planner = STRRT_Planer(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5,max_runtime=max_runtime, selection_radius= 1.5, pruning_radius=0.1)
            print(f"Car planner params: {car_planner.print_info()}")
        # car_planner = RRT_Planer(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5,max_runtime=max_runtime)
        solved = car_planner.solve()

        if args.verbose:
            print(f"[Planner {planner_id}] Run #{run_count+1} finished in {time.time() - start_time:.2f}s. Success: {'Exact' if solved else 'Approximate' if solved is not None else 'No solution'}")
        # print(f"[Planner {planner_id}] Run #{run_count} finished in {elapsed:.2f}s. Success: {'Exact' if solved else 'Approximate' if solved is not None else 'No solution'}")
        
        if solved is not None: 
        # Store result at this planner's slot
            result_list[planner_id] = {
                'planner': car_planner,
                'timestamp': time.time(),
                'solved': solved,
                'run': run_count, 
            }
        if solved and save_dir is not None and map_indexes is not None:
            save_to_file(car_planner, save_dir,planner_id,run_count,map_indexes[planner_id][run_count])

        run_count += 1        
        
        
    if args.verbose:
        print(f"[Planner {planner_id}] Completed all {runs_per_planner} runs")



def save_to_file(planner, save_dir,thread,run, map_index):
    #TODO checking if dir exists? 
    if args.verbose:
        print(f"Saving path for Thread {thread} Run {run} Map {map_index} to file.")

    planner_data =  planner.print_info() 
    robot = planner_data.pop('robot', None)
    path_data = planner_data.pop('solved_path', None)
    data = {
        'robot': robot,
        'planner': planner_data,
        'path': path_data,
    }

    save_dir = os.path.join(save_dir,f"map_{map_index}")
    filepath = os.path.join(save_dir,  f'path_{thread}_{run}.json')
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)




def generate_map_indexes_and_folders(num_threads, runs_per_planner, maps_png):
    n_maps = len(maps_png)
    n_runs = num_threads * runs_per_planner
    base_runs_per_map = n_runs // n_maps
    map_indexes = [[] for _ in range(num_threads)]    
    current_map =  0 
    for i in range(n_runs):
        map_indexes[i %num_threads] += [current_map]
        if (i+1) % base_runs_per_map == 0: 
            current_map += 1 
        if current_map >= n_maps:
            current_map = 0

    if args.save:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M')
        save_dir = os.path.join(data_dir, timestamp)
        for map_idx in range(n_maps):
            map_folder = os.path.join(save_dir, f"map_{map_idx}")
            os.makedirs(map_folder, exist_ok=True)
            maps_png[map_idx].save(os.path.join(map_folder, 'map.png'))


    print(f'Each map will be used approximately {base_runs_per_map} times.')
    return map_indexes,save_dir if args.save else None


def run_parallel(num_threads=4, runs_per_planner=5, max_runtime=3):
    num_threads = min(num_threads, cpu_count() - 1) 
    
    if args.vis:
        vis = Visualizer(num_threads)  
    


    manager = Manager()
    result_list = manager.list([None] * num_threads)  # Fixed-size list
    stop_event = manager.Event()
    

    maps = []
    if args.map is not None:
        print(f"Loading map from: {args.map}")

        maps_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder_path = os.path.join(maps_dir, args.map)
        if not os.path.isdir(folder_path):
            print("Provided map path is not a valid folder.")
            return
        png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]


        for png in png_files:
            img_path = os.path.join(folder_path, png)
            img = Image.open(img_path) # Convert to grayscale
            maps.append(img)

    else:
        print("No map loading path provided, using random maps not implemented yet")
        return






    print(f"Running {num_threads} planners ({runs_per_planner} runs each)...\n")

    map_indexes,save_dir = generate_map_indexes_and_folders(num_threads, runs_per_planner, maps)


    maps = [np.array(m.convert('1'))[::-1] for m in maps]

    start_time = time.time()

    with Pool(processes=num_threads) as pool:
        async_results = [
            pool.apply_async(run_planner_continuous, 
                           (i, max_runtime, result_list, stop_event, runs_per_planner,save_dir,maps, map_indexes))
            for i in range(num_threads)
        ]
        

        while not all(r.ready() for r in async_results):
            if args.vis:
                vis.update(result_list)
            print(f"Elapsed Time: {time.time() - start_time:.2f}/{max_runtime*runs_per_planner}s", end='\r')
            time.sleep(0.05)

    print('\n\nAll planners completed all runs.')


if __name__ == "__main__":
    run_parallel(args.num_threads, args.runs_per_planner, args.max_runtime)

