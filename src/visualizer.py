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
    def __init__(self, n_plots,show=True):
        
        if show: plt.ion()
        self.n_plots = n_plots
        n_cols = math.ceil(math.sqrt(self.n_plots))

        self.font_size  = max(4, 12 -  n_cols) 
        n_rows = math.ceil(self.n_plots / n_cols)
        
        self.fig, axs = plt.subplots(n_rows, n_cols,figsize=(5 * n_cols, 5 * n_rows))#, figsize=(5 * n_cols, 5 * n_rows))
        #TODO ADD the thtile back
        # self.fig.suptitle(f'OMPL Car Planning - Continuous', fontsize=16)
        # fig.set_facecolor('#2e2e2e')
        # Flatten axes
        self.fig.tight_layout()
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
        # plt.pause(0.5)

        self.last_timestamps = [0.0] * self.n_plots

        self.set_labels = False

    def update(self, result_list,show_params=True,show_labels=True,custom_title=None):
        #custom title : list of strings with length n_plots
        for i in range(self.n_plots):
                result = result_list[i]
                ax = self.axs_flat[i]
                if result is None:
                    ax.set_visible(False)
                    continue
                if result['timestamp'] > self.last_timestamps[i]:
                    self.last_timestamps[i] = result['timestamp']
                    ax.set_visible(True)
                    self.draw_one(result, ax, show_params, show_labels, custom_tile=custom_title[i] if custom_title is not None else None)

                    ax.cla()
                    ax.set_xlim(0, 10)
                    ax.set_ylim(0, 10)
                    ax.grid(True, alpha=0.3)
                    
                    max_vel = result['planner'].visualize(ax,point_iteration = 50)
                    # ax.set_title(f'Planner {i} - Run {result["run"]} - Solved:  {"Exact" if result["solved"] else "Approximate" if result["solved"] is not None else "No solution"}')
                    # ax.set_title(f'P:{i}R:{result["run"]}-{"Exact" if result["solved"] else "Approximate" if result["solved"] is not None else "No solution"} V:{max_vel:.2f}/{result["planner"].robot.max_velocity:.2f}')
                    ax.set_title((f'V:{max_vel:.2f}/{result["planner"].robot.max_velocity:.2f} ' + (f'T:{result["planner"].solved_time:.2f}s' if result['planner'].solved_time is not None else '')) if custom_tile is None else custom_tile[i])
                    if not self.set_labels:
                        handles, labels = ax.get_legend_handles_labels()
                        unique_labels = dict(zip(labels, handles))
                        # self.fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper left')
                        # self.set_labels = True


            if show_params:
                legend_text = "\n".join(f"{key}: {value:.2f}" for key, value in result['planner'].robot.print_info().items() if isinstance(value, (int, float))) 
                ax.text(1,0, legend_text, transform=ax.transAxes, 
                    verticalalignment='bottom',fontsize = self.font_size , horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.5))



    def close(self):
        return




def parse_path(map_path, path_files,map_array):
    all_paths = []
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
            print(f"Loaded path {i + 1}/{len(path_files)} from {map_folder}")
            all_paths.append({   
                'planner': planner,
                'solved': True,
                'timestamp': 0.0,
                'run': path_file,
            })
    return all_paths

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Visualization of OMPL Car Planning Results")
    parser.add_argument('-n', '--num_plots', type=int, default=16, help='Number of plots')
    parser.add_argument('-d', '--data', type=str, default=None, help='Path to load data from file')
    parser.add_argument('--params', action='store_true', help='Do not show robot parameters on plots')
    parser.add_argument('--save', type=str, default=None, help='Path to save the visualization as an image')
    args = parser.parse_args()

    if args.data is None:
        print("Please provide a data file path using the -d or --data argument.")
        exit(1)
    if not os.path.isdir(args.data):
        print(f"The provided data path '{args.data}' is not a valid directory.")
        exit(1)


    map_folders = sorted(
    [d for d in os.listdir(args.data) if d.startswith('map_')],
    key=lambda x: int(x.split('_')[1])
)


    all_results = []
    map_idx = []
    for map_folder in map_folders:
        map_path = os.path.join(args.data, map_folder)
        map_files = [f for f in os.listdir(map_path) if f == 'map.png']
        map_array =  np.array(Image.open(os.path.join(map_path, map_files[0])).convert('1'))[::-1] if map_files else np.zeros((50,50)) 
        path_files = [f for f in os.listdir(map_path) if f.startswith('path_') and f.endswith('.json')]
        all_results += parse_path(map_path, path_files,map_array)
        map_idx += [f"{map_folder}_path_{i}" for i in range(len(path_files))]
    total_pages = math.ceil(len(all_results) / args.num_plots)

    if args.save is not None:
        os.makedirs(os.path.join('path_visualizations', args.save), exist_ok=True)

        visualizer = Visualizer(n_plots=args.num_plots,show=False)
        for i in range(total_pages):

            start_idx = i * args.num_plots
            end_idx = min(start_idx + args.num_plots, len(all_results))
            
            result_list = [None] * args.num_plots
            
            for plot_idx in range(args.num_plots):
                result_idx = start_idx + plot_idx
                if result_idx < end_idx:
                    all_results[result_idx]['timestamp'] = datetime.now().timestamp() 
                    result_list[plot_idx] = all_results[result_idx]
            visualizer.update(result_list, args.params,show_labels=False, custom_title=['']*args.num_plots)
            visualizer.fig.savefig(os.path.join('path_visualizations', args.save, f'visualization_{"-".join([map_idx[start_idx]] + ( [map_idx[end_idx-1]] if end_idx - 1 != start_idx else []) )}.pdf'))
        exit(0)

    visualizer = Visualizer(n_plots=args.num_plots,show= args.save is not None)
    current_page = 0 

    def update_display(show_labels=True,custom_title=None):
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
        
        visualizer.update(result_list, args.params, show_labels, custom_title=custom_title)
        if args.save:
            visualizer.fig.savefig(os.path.join('path_visualizations', args.save, f'visualization_{"-".join([map_idx[start_idx]] + ( [map_idx[end_idx-1]] if end_idx - 1 != start_idx else []) )}.pdf'))

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

    if args.save is None:
        visualizer.fig.canvas.mpl_connect('key_press_event', on_key)
        update_display()
    else:
        os.makedirs(os.path.join('path_visualizations', args.save), exist_ok=True)
        for i in range(total_pages):
            current_page = i
            update_display(show_labels=False, custom_title=['']*args.num_plots)
        exit(0)


    plt.ioff()
    plt.show()
