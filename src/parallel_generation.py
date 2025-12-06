from RRT_acceleration import CarOMPL_acceleration
from STRRT_acceleration import SSTCarOMPL_acceleration
from RRT import RRT_Planer
from STRRT import STRRT_Planer
from  Dubins import Dubins_pathfinding
import time 
from multiprocessing import Pool, cpu_count,Manager
from base_pathfind_classes import CircleObstacle, RectangleObstacle, Robot,Obstacle

import matplotlib.pyplot as plt
import math

import argparse
import random

from ompl import base as ob 

#TODO diferentiat visualization with pooling no vis which just runs and saves to file   


parser = argparse.ArgumentParser(description="Parallel OMPL Car Planners")
parser.add_argument('-n', '--num_threads', type=int, default=4, help='Number of parallel planner threads')
parser.add_argument('-r', '--runs_per_planner', type=int, default=5, help='Number of runs per planner')
parser.add_argument('-t', '--max_runtime', type=float, default=3.0, help='Maximum runtime per planner run (seconds)')
parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
parser.add_argument('--vis', action='store_true', help='Enable visualization of planning process')
args = parser.parse_args()


def run_planner_continuous(planner_id, max_runtime, result_list, stop_event, runs_per_planner):
    """Run planner continuously and store results at specific slot."""
    run_count = 0
    
    while run_count < runs_per_planner and not stop_event.is_set():
        if args.verbose:
            print(f"[Planner {planner_id}] Starting run #{run_count + 1}")
        start_time = time.time()
        
        robot = Robot()

        robot.radius = random.uniform(0.2,0.4)
        robot.wheelbase = random.uniform(0.3,1.2) 
        robot.max_velocity = random.uniform(10,20)
        robot.wheelbase = 0.3
        robot.acceleration = 2



        obstacles = [RectangleObstacle(random.uniform(0,10), random.uniform(0,10), random.uniform(0.5,2), random.uniform(0.5,2)) for i in range(random.randint(5,9))]
        # obstacles += [CircleObstacle(random.uniform(0,10), random.uniform(0,10), random.uniform(0.3,1.0)) for i in range(random.randint(3,10))]
      
      
        # car_planner = CarOMPL_acceleration(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5,max_runtime=max_runtime)
        
        car_planner = SSTCarOMPL_acceleration(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),pos_treshold=0.5,max_runtime=max_runtime)
        # car_planner = Dubins_pathfinding(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),max_runtime=max_runtime)
        # car_planner = STRRT_Planer(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5,max_runtime=max_runtime, selection_radius= 1.5, pruning_radius=0.1)
       
        # car_planner = RRT_Planer(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5,max_runtime=max_runtime)
       
        solved = car_planner.solve()
        run_count += 1
        
        # print(f"[Planner {planner_id}] Run #{run_count} finished in {elapsed:.2f}s. Success: {'Exact' if solved else 'Approximate' if solved is not None else 'No solution'}")
        
        if solved is not None: 
        # Store result at this planner's slot
            result_list[planner_id] = {
                'planner': car_planner,
                'timestamp': time.time(),
                'run': run_count, 
                'solved': solved,
                'randomized_params': {
                    'robot_radius': robot.radius,
                    'wheelbase': robot.wheelbase,
                    'max_velocity': robot.max_velocity,
                    # 'pruning_radius': car_planner.pruning_radius,
                    # 'selection_radius': car_planner.selection_radius,
                }
                #TODO count exact solutions 
                # 'exact': solved is ob.PlannerStatus.EXACT_SOLUTION,
            }
        
    if args.verbose:
        print(f"[Planner {planner_id}] Completed all {runs_per_planner} runs")




def run_parallel(num_threads=4, runs_per_planner=5, max_runtime=3):
    num_threads = min(num_threads, cpu_count() - 1) 

    if args.vis:
        plt.ion()
        n_plots = num_threads
        n_cols = math.ceil(math.sqrt(n_plots))
        n_rows = math.ceil(n_plots / n_cols)
        
        fig, axs = plt.subplots(n_rows, n_cols)#, figsize=(5 * n_cols, 5 * n_rows))
        fig.suptitle(f'OMPL Car Planning - Continuous ({runs_per_planner} runs each)', fontsize=16)
        # fig.set_facecolor('#2e2e2e')
        # Flatten axes
        if n_plots == 1:
            axs_flat = [axs]
        elif n_rows == 1 or n_cols == 1:
            axs_flat = axs.flatten() if hasattr(axs, 'flatten') else list(axs)
        else:
            axs_flat = axs.flatten()
        
        # Initialize plots
        for idx, ax in enumerate(axs_flat):
            ax.set_title(f'Planner {idx} - waiting...')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            # ax.set_facecolor('#aaaaaa')
            ax.grid(True, alpha=0.3)
            if idx >= num_threads:
                ax.set_visible(False)

        plt.tight_layout()
    plt.pause(0.5)
    
    # Create shared resources


    manager = Manager()
    result_list = manager.list([None] * num_threads)  # Fixed-size list
    stop_event = manager.Event()
    
    num_cores = min(num_threads, cpu_count())
    print(f"Running {num_threads} planners on {num_cores} cores ({runs_per_planner} runs each)...")
    
    # Track what we've drawn
    last_timestamps = [0.0] * num_threads
    total_expected = num_threads * runs_per_planner
    total_drawn = 0
    
    exact_solutions = 0 

    with Pool(processes=num_cores) as pool:
        # Start all continuous planners
        async_results = [
            pool.apply_async(run_planner_continuous, 
                           (i, max_runtime, result_list, stop_event, runs_per_planner))
            for i in range(num_threads)
        ]
        
        # Poll and draw as results arrive
        while total_drawn < total_expected:
            for i in range(num_threads):
                result = result_list[i]
                
                # Check if this slot has a new result
                if result is not None and result['timestamp'] > last_timestamps[i]:
                    last_timestamps[i] = result['timestamp']
                    total_drawn += 1


                    if result['solved']:
                        exact_solutions += 1
                        print(f'FOUND EXACT SOLUTION NR. {exact_solutions}')
                        #TODO save to file, map and path 


                    if args.vis:
                        ax = axs_flat[i]
                        
                        ax.clear()
                        ax.set_xlim(0, 10)
                        ax.set_ylim(0, 10)
                        ax.grid(True, alpha=0.3)
                        
                        result['planner'].visualize(ax)
                        ax.set_title(f'Planner {i} - Run {result["run"]} - Solved:  {"Exact" if result["solved"] else "Approximate" if result["solved"] is not None else "No solution"}')
                        

                        handles, labels = ax.get_legend_handles_labels()
                        unique_labels = dict(zip(labels, handles))
                        fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper left')

                        
                        legend_text = "\n".join(f"{key}: {value:.2f}" for key, value in result['randomized_params'].items())
                        ax.text(0.02, -0.1, legend_text, transform=ax.transAxes, 
                            verticalalignment='top', fontsize=9,
                            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))


                        fig.canvas.draw()
                        fig.canvas.flush_events()
            if args.vis:
                plt.pause(0.01)
            time.sleep(0.05)  # Polling interval
        
        # Signal stop (shouldn't be needed since each planner stops after N runs)
        stop_event.set()
        
        # Wait for all workers
        for async_result in async_results:
            async_result.wait()
    
    print('\n\nAll planners completed all runs.')


if __name__ == "__main__":
    run_parallel(args.num_threads, args.runs_per_planner, args.max_runtime)

