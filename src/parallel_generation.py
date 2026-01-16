from datetime import datetime
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
import json



from PIL import Image
import numpy as np
import math


#TODO should the structure be map and paths or link to map 
#TODO chenge the verbose add some nice tuchouces


parser = argparse.ArgumentParser(description="Parallel OMPL Car Planners")
parser.add_argument('-n', '--num_threads', type=int, default=4, help='Number of parallel planner threads')
parser.add_argument('-r', '--runs_per_planner', type=int, default=5, help='Number of runs per planner')
parser.add_argument('-t', '--max_runtime', type=float, default=3.0, help='Maximum runtime per planner run (seconds)')
parser.add_argument('--save', type=str, default=None, help='Saving paths to files')
parser.add_argument('--map', type=str, default=None, help='Path to load map from file')
parser.add_argument('--run_id', type=int, default=None, help='Identifier to append to saved run folders')
#run id not given run folder gen and parallel 
#run id >= 0 just run 
#run id = -1 generate folders only 
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
parser.add_argument('--vis', action='store_true', help='Enable visualization of planning process')


args = parser.parse_args()
def run_planner_continuous(planner_id, max_runtime, result_list, stop_event, runs_per_planner, save_dir=None, maps=None, map_indexes=None):
    run_count = 0
    
    if maps is None or map_indexes is None or len(map_indexes) <= planner_id:
        print(f"[Planner {planner_id}] No maps provided, exiting.")
        return

    while run_count < runs_per_planner and not stop_event.is_set():

        start_time = time.time()
        
        if map_indexes[planner_id][run_count] < len(maps):
            map_data = maps[map_indexes[planner_id][run_count]]
        else:
            continue

        robot=RectangleRobot(random.uniform(0.1,1),random.uniform(0.1,1),collision_check_angle_res=180)
        robot.wheelbase = robot.length * random.uniform(0.4, 1.0)
        robot.max_velocity = random.uniform(5.0,20.0)
        robot.acceleration = random.uniform(2.0,10.0)
        robot.mu_static = random.uniform(0.05,2.5)
        robot.max_steering_at_zero_v = random.uniform(math.pi / 8.0, math.pi / 3.0)

        car_planner = SSTCarOMPL_acceleration(robot=robot,map=map_data,start=(1.5,1.5,random.uniform(0, math.pi/2)),goal=(13.5,1.5,0),pos_treshold=0.5,max_runtime=max_runtime, vel_threshold=1,bounds=(15,15))
        
        
        
        # car_planner = Dubins_pathfinding(robot=robot,map=map_data,start=(1.5,1.5,0),goal=(13.5,1.5,-math.pi),max_runtime=max_runtime,bounds=(15,15))
        # car_planner = Pacejka_pathfinding(max_runtime=max_runtime, map=map_data,robot =PacejkaRectangleRobot(random.uniform(0.1,0.5),random.uniform(0.3,1.0),max_velocity=15),vel_threshold=2,velocity_weight=0,start=(1.5,3.0,0.0),goal=(9.0,7.0,0.0), bounds=(10,10))
        # car_planner = CarOMPL_acceleration(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5,max_runtime=max_runtime)
        # car_planner = SSTCarOMPL_acceleration(robot=robot,map=map_data,start=(1.0,1.0),goal=(9.0,9.0),pos_treshold=0.5,max_runtime=max_runtime, vel_threshold=1, velocity_weight=0.1)
        # car_planner = STRRT_Planer(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5,max_runtime=max_runtime, selection_radius= 1.5, pruning_radius=0.1)
        # car_planner = RRT_Planer(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5,max_runtime=max_runtime)

        if args.verbose:
            print(f"[Planner {planner_id}] Starting run #{run_count + 1}")
            print(f"Car planner params: {car_planner.print_info()}")
        solved = car_planner.solve()

        if args.verbose:
            print(f"[Planner {planner_id}] Run #{run_count+1} finished in {time.time() - start_time:.2f}s. Success: {'Exact' if solved else 'Approximate' if solved is not None else 'No solution'}")
        
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


def verbose(message):
    if args.verbose:
        print(message)

def save_to_file(planner, save_dir,thread,run, map_index):
    # n = (args.run_id * args.runs_per_planner * args.num_threads if args.run_id else 0) + thread * args.runs_per_planner + run 
    n = "_".join(([args.run_id] if args.run_id is not None else []) +  [thread,run])
    filepath = os.path.join(save_dir, f"map_{map_index}" , f'path_{n}.json')
    if not os.path.exists(filepath):
        print('Map folders not generated, skiping saving')
        return 
   
    verbose(f"Saving path for Thread {thread} Run {run} Map {map_index} to file.")

    planner_data =  planner.print_info() 
    robot = planner_data.pop('robot', None)
    path_data = planner_data.pop('solved_path', None)
    data = {
        'robot': robot,
        'planner': planner_data,
        'path': path_data,
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def generate_map_foldrers(maps_png):
    if not args.save:
        return None 

    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', f"{args.save}")

    if args.run_id is None or args.run_id == -1:
        for map_idx in range(len(maps_png)):
            map_folder = os.path.join(save_dir, f"map_{map_idx}")
            os.makedirs(map_folder, exist_ok=True)
            maps_png[map_idx].save(os.path.join(map_folder, 'map.png'))

    return save_dir


def generate_map_indexes (num_threads, runs_per_planner, n_maps):
    n_runs = num_threads * runs_per_planner
    map_indexes = [[] for _ in range(num_threads)]    
    if args.run_id is not None or  n_runs <= n_maps :
        offset = (args.run_id if args.run_id else 0) * n_runs 
        for i in range(n_runs):
            map_indexes[i %num_threads] += [(offset + i) % n_maps]
    else:
        base_runs_per_map = n_runs // n_maps
        current_map =  0 
        for i in range(n_runs):
            map_indexes[i %num_threads] += [current_map]
            if (i+1) % base_runs_per_map == 0: 
                current_map += 1 
            if current_map >= n_maps:
                current_map = 0
        print(f'Each map will be used approximately {base_runs_per_map} times.')
        
    return map_indexes


def run_parallel(num_threads=4, runs_per_planner=5, max_runtime=3):
    num_threads = min(num_threads, cpu_count() - 1) 
    
    if args.vis:
        from visualizer import Visualizer
        vis = Visualizer(num_threads)  

    manager = Manager()
    result_list = manager.list([None] * num_threads)  # Fixed-size list
    stop_event = manager.Event()
    
    folder_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.map)
    if args.map is None or not os.path.isdir(folder_path):
        print("No map loading path provided, or invalid path")
        return

    print(f"Loading map from: {args.map}")

    pngs = [Image.open(os.path.join(folder_path, f))  for f in os.listdir(folder_path) if f.endswith('.png')]
    maps = [np.array(f.convert('1'))[::-1]  for f in pngs]

    print(f"Running {num_threads} planners ({runs_per_planner} runs each)...\n")



    map_indexes = generate_map_indexes(num_threads, runs_per_planner, len(maps))
    save_dir = generate_map_foldrers(pngs) 

    if args.run_id == -1:
        print("Map folders generated, exiting as per run_id = -1.")
        return


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
            if args.run_id is None:
                print(f"Elapsed Time: {time.time() - start_time:.2f}/{max_runtime*runs_per_planner}s", end='\r')
            
            time.sleep(0.05)

    print('\n\nAll planners completed all runs.')

if __name__ == "__main__":
    run_parallel(args.num_threads, args.runs_per_planner, args.max_runtime)

