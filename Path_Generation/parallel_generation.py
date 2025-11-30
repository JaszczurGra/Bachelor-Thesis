from RRT_ompl_car_acceleration_class import CarOMPL_acceleration
import time 
from multiprocessing import Pool, cpu_count,Manager
from base_pathfind_classes import Robot,Obstacle

import matplotlib.pyplot as plt
import math




def run_planner_continuous(planner_id, max_runtime, result_list, stop_event, runs_per_planner):
    """Run planner continuously and store results at specific slot."""
    run_count = 0
    
    while run_count < runs_per_planner and not stop_event.is_set():
        print(f"[Planner {planner_id}] Starting run #{run_count}...")
        start_time = time.time()
        
        robot = Robot()

        obstacles = []
        for obs in [(4.0, 6.0, 4.0, 6.0), (2.0, 3.0, 7.0, 8.0), (7.0, 8.0, 2.0, 3.0),(6.5, 7.0, 2.0, 5.0)]:
            obstacles.append(Obstacle(*obs))


        car_planner = CarOMPL_acceleration(robot=robot,Obstacles=obstacles,start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5,max_runtime=max_runtime)
        solved = car_planner.solve(max_runtime)
        
        elapsed = time.time() - start_time
        
        print(f"[Planner {planner_id}] Run #{run_count} finished in {elapsed:.2f}s. Success: {solved}")
        
        # Store result at this planner's slot
        result_list[planner_id] = {
            'planner': car_planner,
            'timestamp': time.time(),
            'run': run_count
        }
        
        run_count += 1
    
    print(f"[Planner {planner_id}] Completed all {runs_per_planner} runs")




def run_parallel(num_threads=4, runs_per_planner=5, max_runtime=3):


    plt.ion()
    n_plots = num_threads
    n_cols = math.ceil(math.sqrt(n_plots))
    n_rows = math.ceil(n_plots / n_cols)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    fig.suptitle(f'OMPL Car Planning - Continuous ({runs_per_planner} runs each)', fontsize=16)
    
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
                    
                    ax = axs_flat[i]
                    
                    ax.clear()
                    ax.set_xlim(0, 10)
                    ax.set_ylim(0, 10)
                    ax.grid(True, alpha=0.3)
                    
                    result['planner'].visualize(ax)
                    ax.set_title(f'Planner {i} - Run {result["run"]}')
                    
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.01)
                    
            
            time.sleep(0.1)  # Polling interval
        
        # Signal stop (shouldn't be needed since each planner stops after N runs)
        stop_event.set()
        
        # Wait for all workers
        for async_result in async_results:
            async_result.wait()
    
    print('\n\nAll planners completed all runs.')


if __name__ == "__main__":
    run_parallel(9,10,50)

