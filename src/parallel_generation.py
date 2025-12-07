import os
from RRT_acceleration import CarOMPL_acceleration
from STRRT_acceleration import SSTCarOMPL_acceleration
from RRT import RRT_Planer
from STRRT import STRRT_Planer
from  Dubins import Dubins_pathfinding
import time 
from multiprocessing import Pool, cpu_count,Manager
from base_pathfind_classes import Robot
import argparse
import random
from datetime import datetime


from visualizer import Visualizer

from PIL import Image
import numpy as np


#TODO reverse the map color 0 free 1 occupied now is inverse 

parser = argparse.ArgumentParser(description="Parallel OMPL Car Planners")
parser.add_argument('-n', '--num_threads', type=int, default=4, help='Number of parallel planner threads')
parser.add_argument('-r', '--runs_per_planner', type=int, default=5, help='Number of runs per planner')
parser.add_argument('-t', '--max_runtime', type=float, default=3.0, help='Maximum runtime per planner run (seconds)')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
parser.add_argument('--vis', action='store_true', help='Enable visualization of planning process')
parser.add_argument('--save', action='store_true', help='Save solved paths to file')
parser.add_argument('--map', type=str, default=None, help='Path to load map from file')
args = parser.parse_args()


def run_planner_continuous(planner_id, max_runtime, result_list, stop_event, runs_per_planner, save_dir=None, map_data=None):
    """Run planner continuously and store results at specific slot."""
    run_count = 0
    
    while run_count < runs_per_planner and not stop_event.is_set():
        if args.verbose:
            print(f"[Planner {planner_id}] Starting run #{run_count + 1}")
        start_time = time.time()
        
        robot = Robot()

        robot.radius = random.uniform(0.2,0.4)
        robot.wheelbase = random.uniform(0.3,1.2) 
        robot.max_velocity = 14
        robot.wheelbase = 0.3
        robot.acceleration = 5



        # obstacles = [RectangleObstacle(random.uniform(0,10), random.uniform(0,10), random.uniform(0.5,2), random.uniform(0.5,2)) for i in range(random.randint(5,9))]
        # obstacles += [CircleObstacle(random.uniform(0,10), random.uniform(0,10), random.uniform(0.3,1.0)) for i in range(random.randint(3,10))]
      
      
        # car_planner = CarOMPL_acceleration(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5,max_runtime=max_runtime)
        # car_planner = SSTCarOMPL_acceleration(robot=robot,map=map_data,start=(1.0,1.0),goal=(9.0,9.0),pos_treshold=0.5,max_runtime=max_runtime)
        car_planner = Dubins_pathfinding(robot=robot,map=map_data,start=(1.0,1.0),goal=(9.0,9.0),max_runtime=max_runtime)
        # car_planner = STRRT_Planer(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5,max_runtime=max_runtime, selection_radius= 1.5, pruning_radius=0.1)
       
        # car_planner = RRT_Planer(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5,max_runtime=max_runtime)
       
        solved = car_planner.solve()
        run_count += 1
        

        print(f"[Planner {planner_id}] Run #{run_count} finished in {time.time() - start_time:.2f}s. Success: {'Exact' if solved else 'Approximate' if solved is not None else 'No solution'}")
        # print(f"[Planner {planner_id}] Run #{run_count} finished in {elapsed:.2f}s. Success: {'Exact' if solved else 'Approximate' if solved is not None else 'No solution'}")
        
        if solved is not None: 
        # Store result at this planner's slot
            result_list[planner_id] = {
                'planner': car_planner,
                'timestamp': time.time(),
                'solved': solved,

                'run': run_count, 
                'randomized_params': {
                    'robot_radius': robot.radius,
                    'wheelbase': robot.wheelbase,
                    'max_velocity': robot.max_velocity,
                    # 'pruning_radius': car_planner.pruning_radius,
                    # 'selection_radius': car_planner.selection_radius,
                }
            }
        if solved and save_dir is not None:
            save_to_file(car_planner, save_dir,planner_id,run_count)
        
    if args.verbose:
        print(f"[Planner {planner_id}] Completed all {runs_per_planner} runs")



def save_to_file(planner, save_dir,thread,run):
    filepath = os.path.join(save_dir,  f'planner_{thread}_run_{run}_path.txt')
    with open(filepath, 'w') as f:
        f.write(planner.solved_path)






def run_parallel(num_threads=4, runs_per_planner=5, max_runtime=3):
    num_threads = min(num_threads, cpu_count() - 1) 
    
    if args.vis:
        vis = Visualizer(num_threads)  
    

    save_dir = None
    if args.save:
        # Go back one directory and create data folder
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Create timestamped folder (YYYY-MM-DD_HH-MM)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M')
        save_dir = os.path.join(data_dir, timestamp)
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving results to: {save_dir}")
        

    manager = Manager()
    result_list = manager.list([None] * num_threads)  # Fixed-size list
    stop_event = manager.Event()
    

    maps = []
    if args.map is not None:
        print(f"Loading map from: {args.map}")

        maps_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'maps')
        folder_path = os.path.join(maps_dir, args.map)
        if not os.path.isdir(folder_path):
            print("Provided map path is not a valid folder.")
            return
        png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]


        for png in png_files:
            img_path = os.path.join(folder_path, png)
            img = Image.open(img_path).convert('1')  # Convert to grayscale
            map_array = np.array(img)
            # occupancy_grid = np.where(map_array < 128, 1, 0)  # Thresholding
            maps.append(map_array)



    else:
        print("No map loading path provided, using random maps not implemented yet")
        return


 


    print(f"Running {num_threads} planners ({runs_per_planner} runs each)...\n")
    
    start_time = time.time()

    with Pool(processes=num_threads) as pool:
        async_results = [
            pool.apply_async(run_planner_continuous, 
                           (i, max_runtime, result_list, stop_event, runs_per_planner,save_dir,maps[0]))
            for i in range(num_threads)
        ]
        

        while not all(r.ready() for r in async_results):
            if args.vis:
                vis.update(result_list)
            print(f"Elapsed Time: {time.time() - start_time:.2f}/{max_runtime*runs_per_planner}s", end='\r')
            time.sleep(0.1)

    print('\n\nAll planners completed all runs.')


if __name__ == "__main__":
    run_parallel(args.num_threads, args.runs_per_planner, args.max_runtime)

