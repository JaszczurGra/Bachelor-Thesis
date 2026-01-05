from datetime import datetime
import matplotlib.pyplot as plt
import math
import argparse
import os
import numpy as np
from PIL import Image
import json
from base_pathfind_classes import BasePathfinding, get_robot

#TODO change title of the vis

      # Use faster backend
import matplotlib
#TODO uncomment 
# matplotlib.use('TkAgg') 


#TODO check if tkinter exists if not run with base backend and warn user and chane draw idle etc 
#  LineCollection for drawingpaths 
class Visualizer:
    def __init__(self, n_plots):

        plt.ion()
        self.n_plots = n_plots
        n_cols = math.ceil(math.sqrt(self.n_plots))

        self.font_size  = max(4, 12 -  n_cols) 
        n_rows = math.ceil(self.n_plots / n_cols)
        
        self.fig, axs = plt.subplots(n_rows, n_cols)#, figsize=(5 * n_cols, 5 * n_rows))
        self.fig.suptitle(f'OMPL Car Planning - Continuous', fontsize=16)
        # fig.set_facecolor('#2e2e2e')
        # Flatten axes
        if self.n_plots == 1:
            self.axs_flat = [axs]
        elif n_rows == 1 or n_cols == 1:
            self.axs_flat = axs.flatten() if hasattr(axs, 'flatten') else list(axs)
        else:
            self.axs_flat = axs.flatten()
        
        # Initialize plots
        for idx, ax in enumerate(self.axs_flat):
            ax.set_title(f'Planner {idx} - waiting...')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            # ax.set_facecolor('#aaaaaa')
            ax.grid(True, alpha=0.3)
            if idx >= n_plots:
                ax.set_visible(False)
            ax.set_autoscale_on(False)

        # plt.tight_layout()
        plt.pause(0.5)

        self.last_timestamps = [0.0] * self.n_plots

        self.set_labels = False

    def update(self, result_list,show_params=True):
        draw = False
        for i in range(self.n_plots):
                result = result_list[i]
                ax = self.axs_flat[i]
                if result is None:
                    ax.set_visible(False)
                    continue
                if result['timestamp'] > self.last_timestamps[i]:
                    self.last_timestamps[i] = result['timestamp']
                    ax.set_visible(True)
 

                    ax.cla()
                    ax.set_xlim(0, 10)
                    ax.set_ylim(0, 10)
                    ax.grid(True, alpha=0.3)
                    
                    max_vel = result['planner'].visualize(ax)
                    # ax.set_title(f'Planner {i} - Run {result["run"]} - Solved:  {"Exact" if result["solved"] else "Approximate" if result["solved"] is not None else "No solution"}')
                    # ax.set_title(f'P:{i}R:{result["run"]}-{"Exact" if result["solved"] else "Approximate" if result["solved"] is not None else "No solution"} V:{max_vel:.2f}/{result["planner"].robot.max_velocity:.2f}')
                    ax.set_title(f'V:{max_vel:.2f}/{result["planner"].robot.max_velocity:.2f} ' + (f'T:{result["planner"].solved_time:.2f}s' if result['planner'].solved_time is not None else ''))
                    if not self.set_labels:
                        handles, labels = ax.get_legend_handles_labels()
                        unique_labels = dict(zip(labels, handles))
                        self.fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper left')
                        self.set_labels = True


                    if show_params:
                        legend_text = "\n".join(f"{key}: {value:.2f}" for key, value in result['planner'].robot.print_info().items() if isinstance(value, (int, float))) 
                        ax.text(1,0, legend_text, transform=ax.transAxes, 
                            verticalalignment='bottom',fontsize = self.font_size , horizontalalignment='right',
                            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.5))
                    draw = True

        if draw:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()



    def close(self):
        pass
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualization of OMPL Car Planning Results")
    parser.add_argument('-n', '--num_plots', type=int, default=16, help='Number of plots')
    parser.add_argument('-d', '--data', type=str, default=None, help='Path to load data from file')
    parser.add_argument('--params', action='store_true', help='Do not show robot parameters on plots')
    args = parser.parse_args()

    if args.data is None:
        print("Please provide a data file path using the -d or --data argument.")
        exit(1)
    if not os.path.isdir(args.data):
        print(f"The provided data path '{args.data}' is not a valid directory.")
        exit(1)


    map_folders = [d for d in os.listdir(args.data) if d.startswith('map_')]

    all_results = []
    for map_folder in map_folders:
        map_path = os.path.join(args.data, map_folder)
        
        # Load map image
        map_files = [f for f in os.listdir(map_path) if f == 'map.png']

        map_array =  np.array(Image.open(os.path.join(map_path, map_files[0])).convert('1'))[::-1] if map_files else np.zeros((50,50)) 


        
        # Load all path JSON files
        path_files = [f for f in os.listdir(map_path) if f.startswith('path_') and f.endswith('.json')]
        
        for i, path_file in enumerate(path_files):
            with open(os.path.join(map_path, path_file), 'r') as f:
                data = json.load(f)
            

            planner_data = data['planner']
            robot = get_robot(data['robot'])

            if robot and hasattr(robot, 'collision_check_angle_res'):
                collision_check = robot.collision_check_angle_res 
                #Disabling the multiple dilated map generation
                robot.collision_check_angle_res = 0
            planner_data['robot'] = robot
            #TODO add diffrent planners data implementatin finding class 

            filtered_params = {k: v for k, v in planner_data.items() if k in BasePathfinding.__init__.__code__.co_varnames}
            planner = BasePathfinding(map=map_array, **filtered_params)
            planner.solved_path = data['path']

            if robot and hasattr(robot, 'collision_check_angle_res'):
                #Disabling the multiple dilated map generation
                robot.collision_check_angle_res = collision_check

            all_results.append({
                'planner': planner,
                'solved': True,
                'timestamp': 0.0,
                'run': path_file,
            })
            print(f"Loaded path {i + 1}/{len(path_files)} from {map_folder}")
    
    print(f"Loaded {len(all_results)} paths from {len(map_folders)} maps")
    


    visualizer = Visualizer(n_plots=args.num_plots)
    current_page = 0 
    total_pages = math.ceil(len(all_results) / args.num_plots)

    def update_display():
        visualizer.fig.suptitle(f'OMPL Car Planning - Page {current_page + 1}/{total_pages} (Use ← → arrows)', fontsize=16)

        start_idx = current_page * args.num_plots
        end_idx = min(start_idx + args.num_plots, len(all_results))
        
        result_list = [None] * args.num_plots
        
        for plot_idx in range(args.num_plots):
            result_idx = start_idx + plot_idx
            if result_idx < end_idx:
                all_results[result_idx]['timestamp'] = datetime.now().timestamp()
                all_results[result_idx]['run'] = result_idx + 1
                result_list[plot_idx] = all_results[result_idx]
        
        visualizer.update(result_list, args.params)
        # print(f"Showing page {current_page + 1}/{total_pages} (results {start_idx + 1}-{end_idx} of {len(all_results)})")
    

    def on_key(event):
        """Handle keyboard input."""
        global current_page
        if event.key == 'right':
            if current_page < total_pages - 1:
                current_page += 1
                update_display()
        elif event.key == 'left':
            if current_page > 0:
                current_page -= 1
                update_display()

    visualizer.fig.canvas.mpl_connect('key_press_event', on_key)
    update_display()

    plt.ioff()
    plt.show()
