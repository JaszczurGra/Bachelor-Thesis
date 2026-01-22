import math
import os
import io
from pyexpat import model
import numpy as np
import torch
import wandb
import argparse
import matplotlib

from train import DiffusionManager, PathDataset
from model import DiffusionDenoiser
import cv2 #opencv-python

#TODO can be easily swaped to saave per path 
#for coping model scp -r jaszczur@eagle:/mnt/storage_3/home/jaszczur/pl0467-01/scratch/jborowiecki/Bachelor-Thesis/models/crisp-sweep-26 models/

parser = argparse.ArgumentParser()
parser.add_argument('--run_url', type=str, required=True, help='WandB run URL to load the model from')
parser.add_argument('-m', '--max_dataset_length', type=int, default=None, help='Maximum number of samples to load from the dataset for debuging')
parser.add_argument('--custom_dataset', type=str, default=None, help='Path to a custom dataset to visualize')

parser.add_argument('--n_viz', type=int, default=300, help='Number of visualizations imgs to generate ')

parser.add_argument('-n', '--num_plots', type=int, default=1, help='Number of plots in the path')
#TODO make this a number to save n number of singular plots 
parser.add_argument('--save', action='store_true', help='Whether to save the plots as images instead of displaying them')

parser.add_argument('--batch_size', type=int, default=512, help='Batch size for processing paths')
args = parser.parse_args()

matplotlib.use('Agg' if args.save else 'TkAgg') 

import matplotlib.pyplot as plt
import re
from matplotlib.patches import Polygon

#TODO vis goes over efvery map and path in order so to have diffrent ones it doesnt work 


def parse_run_url(url):
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/(?:sweeps/[^/]+/)?runs/([^/?]+)", url)    if 'runs' in url else re.search(r"wandb\.ai/([^/]+)/([^/]+)/sweeps/([^/?#]+)", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    else:
        raise ValueError("Invalid wandb run URL format.")




#TODO do this as tensor 
def renormalize_robot(robot_params, robot_normalization, robot_variables):
    norm_ranges = np.array([robot_normalization[var][1] - robot_normalization[var][0] for var in robot_variables])
    norm_mins = np.array([robot_normalization[var][0] for var in robot_variables])
    renormalized = 0.5 * (robot_params + 1) * norm_ranges + norm_mins
    return {var: renormalized[:, i] for i, var in enumerate(robot_variables)}


def calculate_path_length(resampled_paths):
    # resampled_path: [T, path_dim]
    #returns length in normalized cords to denormilize * 15 ^ 2
    lengths = []
    for path in resampled_paths:
        path = np.array(path)  # shape [N, 2]
        length = 0.0
        for i in range(1, path.shape[0]):
            segment = path[i, :2] - path[i - 1, :2]
            length += np.linalg.norm(segment)
        lengths.append(length)
    return lengths


def calculate_validity_collisions(resampled_paths, map_tensors, robot_params_normalized):
    # resampled_paths: [N, 2, L] L not the same for the same 
    # map_tensors: [N, 496, 496]
    # robot_params_normalized: list of dicts with 'width' and 'length'
    # returns collision count in [0,1] for each path
    collision_counts = []
    for i in range(len(resampled_paths)):
        path = np.array(resampled_paths[i])  # shape [N,2]
        map_img = map_tensors[i][0]   # shape [496, 496]
        width = robot_params_normalized['width'][i]
        # length = robot_params_normalized[i]['length']


        img = np.zeros((map_img.shape[0], map_img.shape[1]), dtype=np.uint8)
    
        #TODO should be extended by length 
        pts = np.round((path[:, :2] * 0.5 + 0.5 ) * map_img.shape[0]).astype(int)

        #TODO rounding to int is not accuarate
        #Needs [2,N]?
        thickness = max(1, int(width *  map_img.shape[0] / 15))
        cv2.polylines(img, [pts.reshape((-1, 1, 2))], isClosed=False, color=1, thickness=thickness)

        collisions = np.sum((map_img == 0) & (img == 1))
      
        collision_counts.append(collisions / map_img.shape[0] / map_img.shape[1])  
    # print('Collisions:', collision_counts)
    return collision_counts

def calculate_turning_radius(resampled_paths):
    #TODO not finished yet

    avg_bending_energies = [] 
    avg_curvatures = [] 
    for path in resampled_paths:
        path = np.array(path)  # shape [N, 2]
        # Initialize arrays with 'inf' for radius and 0 for curvature
        bending_energy_sum = 0 
        curvature_sum = 0 
        for i in range(1, len(path) - 1):
            p1 = path[i - 1]
            p2 = path[i]
            p3 = path[i + 1]

            # 1. Calculate the lengths of the sides of the triangle
            a = np.linalg.norm(p2 - p1)
            b = np.linalg.norm(p3 - p2)
            c = np.linalg.norm(p3 - p1)

            # 2. Calculate the area of the triangle using the cross product
            # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
            area = 0.5 * abs(p1[0]*(p2[1] - p3[1]) + 
                            p2[0]*(p3[1] - p1[1]) + 
                            p3[0]*(p1[1] - p2[1]))


                # r = 1 / k

 
            if area > 1e-9:  
                k = (4 * area) / (a * b * c)
                bending_energy_sum += k  *  k
                curvature_sum += k

        avg_bending_energies.append(bending_energy_sum / (len(path) - 2))
        avg_curvatures.append(curvature_sum / (len(path) - 2))

    return avg_curvatures, avg_bending_energies
visualized = 0
def visualize_results(maps, robot_params_renormalized, dynamic_paths, gt_paths, model_resampled_paths, model_path_points,axes,save_dir):


    
    # N, 2 
    def draw_robot(paths,index, i):
        path = np.array(paths[index])
        robot_width = robot_params_renormalized['width'][i]
        robot_length = robot_params_renormalized['length'][i]
        corners = np.array([
            [-robot_length/2, -robot_width/2],
            [robot_length/2, -robot_width/2],
            [robot_length/2, robot_width/2],
            [-robot_length/2, robot_width/2],
            [-robot_length/2, -robot_width/2]
        ])

        #show work coretly no mater if normalized or not 
        theta = math.atan2(path[1][1] - path[0][1],path[1][0] - path[0][0])

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_corners = corners @ rotation_matrix.T
        rotated_corners[:, 0] += (path[0][0] + 1) * 7.5
        rotated_corners[:, 1] += (path[0][1] + 1) * 7.5

        rect = Polygon(rotated_corners, closed=True, facecolor='green', edgecolor='b', alpha=0.5, label='Robot')
        axes[i].add_patch(rect)

    def plot_path(paths, index, i, style, label):
        if paths is not None:
            path = np.array(paths[index])


            x = (path[:, 0] + 1) * 7.5
            y = (path[:, 1] + 1) * 7.5
            # print(min(x),max(x), min(y), max(y))
            axes[i].plot(x, y, style, linewidth=2, label=label)

    def plot_points(paths, index, i, label):
        if paths is not None:
            path = np.array(paths[index])

            x = (path[:, 0] + 1) * 7.5
            y = (path[:, 1] + 1) * 7.5

            axes[i].scatter(x, y, color='r', marker='o', label=label)

    global visualized
    n = min(len(maps), args.n_viz - visualized)
    for index in range(n):
        #TODO this will only loop through 4 and then repeast the same maps and not go forward 
        i = index % args.num_plots
        axes[i].clear()
        axes[i].imshow(maps[i, 0], cmap='gray', extent=[0,15,0,15], origin='lower')
        # axes[i].imshow(maps[i, 0], cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
        # # Set ticks/labels to show 0-15 scale, but data stays in [-1, 1]
        # axes[i].set_xticks(np.linspace(-1, 1, 6))
        # axes[i].set_xticklabels([f"{x:.1f}" for x in np.linspace(0, 15, 6)])
        # axes[i].set_yticks(np.linspace(-1, 1, 6))
        # axes[i].set_yticklabels([f"{y:.1f}" for y in np.linspace(0, 15, 6)])
        # axes[i].set_xlabel("X [m]")
        # axes[i].set_ylabel("Y [m]")
    

        #TODO switch from [i] to passing whole array 
        
        
        plot_points(model_path_points,index ,i, 'Generated points')
        plot_path(model_resampled_paths, index,i,'b-', 'Resampled')
        plot_path(dynamic_paths, index,i,'m-', 'Dynamic Simulated')

        plot_path(gt_paths,index ,i,'g--', 'GT')
        draw_robot(gt_paths,index,i)
        

        
        axes[i].legend()
        axes[i].set_title('')
        visualized += 1

        if i == args.num_plots - 1 or visualized == args.n_viz:
            if args.save:
                plt.savefig(f"{save_dir}/visualization_{visualized}.pdf")
            else:
                plt.ion()
                plt.show(block=False)
                plt.pause(0.1)



def simulate_path_cuda(path,robot_params, dataset):
    # states = torch.zeros((N, T, 4), device=device)      # [x, y, theta, v]
    # controls = torch.zeros((N, T, 2), device=device)    # [acceleration, steering_angle]
    device = path.device  # Ensure all tensors are on the same device
    # Make sure these tensors are on the same device as path
    norm_ranges = torch.tensor(
        [[dataset.path_normalization[var][1] - dataset.path_normalization[var][0] for var in dataset.path_variables]],
        device=device
    ).unsqueeze(-1)
    norm_mins = torch.tensor(
        [[dataset.path_normalization[var][0] for var in dataset.path_variables]],
        device=device
    ).unsqueeze(-1)

    path = path.clone()
    path = 0.5 * (path + 1) * norm_ranges + norm_mins
# Your robot_params should be tensors or broadcastable arrays on the same device
    states = path[:, :4, :]      # [N, T, 4]  [x, y, theta, v]
    controls = path[:, 4:6, :]    # [N, T, 2]  [acceleration, steering_angle]
    robot_param_dict = {}
    for i, key in enumerate(dataset.robot_variables + ['dt']):
        robot_param_dict[key] = robot_params[:, i]
    robot_params = robot_param_dict
    
    robot_params = { k: (v + 1) *  (dataset.robot_normalization[k][1] - dataset.robot_normalization[k][0]) / 2 + dataset.robot_normalization[k][0] for k, v in robot_params.items()}

    _lateral_force_min_v = torch.sqrt(robot_params["wheelbase"] * robot_params["mu_static"] * 9.81 / torch.tan(robot_params["max_steering_at_zero_v"]) ) 
   
    dt = robot_params["dt"]  # (N, 1)

    for t in range(1, dataset.path_length):
        prev_state = states[:, :, t-1]  # (N, 4)
        control = controls[:, :, t]     # (N, 2)
        # All math below should use torch functions!
        # Example:
        angle = torch.where(
            prev_state[:, 3] >= _lateral_force_min_v,
            robot_params["wheelbase"] * robot_params["mu_static"] * 9.81 / prev_state[:, 3]**2 * torch.sign(control[:,1]),
            torch.tan(torch.clamp(control[:, 1], -robot_params["max_steering_at_zero_v"], robot_params["max_steering_at_zero_v"]))
        )
        dx = prev_state[:, 3] * torch.cos(prev_state[:, 2])
        dy = prev_state[:, 3] * torch.sin(prev_state[:, 2])
        dtheta = (prev_state[:, 3] / robot_params["wheelbase"]) * angle
        dv = control[:, 0]
        # Update state
        states[:, 0, t] = prev_state[:, 0] + dx * dt
        states[:, 1, t] = prev_state[:, 1] + dy * dt
        states[:, 2, t] = prev_state[:, 2] + dtheta * dt
        states[:, 3, t] = prev_state[:, 3] + dv * dt

    path = (torch.cat([states, controls], dim=1) - norm_mins) / norm_ranges * 2 - 1

    return path.cpu().numpy()
    


from scipy.interpolate import CubicSpline
from train import BSpline
def resample_paths(paths, path_type, original_paths):
    #only resample cubic and bspline 

    path_type, path_len = path_type.split(':')[0]  , int(path_type.split(':')[1] if len(path_type.split(':')) >1 else  0)


    resampled = []

    if path_type == 'cubic':
        for path, original_path in zip(paths, original_paths):
            cs = CubicSpline(np.linspace(0, 1, path_len), path, axis=0)
            new_points = cs(np.linspace(0, 1, len(original_path)))
            resampled.append(new_points)
    elif path_type == 'bspline':
        for path, original_path in zip(paths, original_paths):
            bspline = BSpline(path_len,6, len(original_path))
            new_points =  bspline.N[0] @ path 
            resampled.append(new_points)
    elif path_type == 'extend':
        #cut to original length
        for path, original_path in zip(paths, original_paths):
            resampled.append(path[:original_path.shape[0], :])
    else :
        #don't do anything can't reverse downsampling for linear 
        return paths 

    #all have diffrent lenghts excpet the linear as it's not reversible 

    return resampled

    



import argparse
from matplotlib.path import Path as MplPath

CPU_COUNT =  os.cpu_count() - 1 if os.cpu_count() is not None else 2  #TODO set it up better 
from multiprocessing import Pool

def resample_single_path(path, path_type, original_path):
    return resample_paths([path], path_type, [original_path])[0]

def visualize_model(run, sweep_name=None):
    if run.state == 'failed':
        print(f"Skipping failed run: {run.name}")
        return

    print(f"Visualizing results for run: {run.name}")

    config = run.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dynamic = config.get('dynamic', False)

    n_maps = config["n_maps"] if args.max_dataset_length is None else min(config["n_maps"], args.max_dataset_length)
    
    dataset = PathDataset(args.custom_dataset if args.custom_dataset is not None else config["dataset_path"], n_maps, config["map_resolution"], config.get('path_type', None),dynamic)

    start_time = datetime.datetime.now()

    diff = DiffusionManager(
        timesteps=config["timesteps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        device=device
    )

    best_model_path = os.path.join('models', run.name, 'best_model_checkpoint.pth')
    vis_model = DiffusionDenoiser(
        state_dim=dataset.path_dim,
        robot_param_dim=dataset.robot_dim,
        map_size=dataset.maps.shape[2],
        map_feat_dim=config["model"]["map_feat_dim"],
        robot_feat_dim=config["model"]["robot_feat_dim"],
        time_feat_dim=config["model"]["time_feat_dim"],
        num_internal_layers=config["model"]["num_internal_layers"],
        base_layer_dim=config["model"]["base_layer_dim"],
        verbose=False
    ).to(device)
    if os.path.exists(best_model_path):
        print('DIDN"T LOAD MODEL IT"S NOT LOCALY SAVED')
        vis_model.load_state_dict(torch.load(best_model_path))

    print('Dynamic:',dynamic, end='\n\n')
    n =  args.num_plots
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    # set n of subplots to n 
    #TODO move this into the visualize to open new one 
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten() if n > 1 else [axes]
    plt.tight_layout()


    save_dir = '/'.join(["visualizations"] + ([sweep_name] if sweep_name else []) + [run.name])
    if args.save:
        os.makedirs(save_dir, exist_ok=True)

    collisons_original = []
    collisons_model = []
    curvatures_original = []
    bending_energies_original = []
    curvatures_model = []
    bending_energies_model = []
    path_lengths_original = []
    path_lengths_model = []

    global visualized
    visualized = 0

    print('Loaded dataset')

    #TODO parallize calculations 
    for b in range(0, len(dataset), args.batch_size):
        batch_idxs = list(range(b, min(b + args.batch_size, len(dataset))))
        map_tensors, robots_params, sampled_paths  = [torch.stack(tensors) for tensors in zip(*[dataset[i] for i in batch_idxs])]
        original_paths = dataset.original_paths[batch_idxs]  # [B, path_len, path_dim]
        # Normalize each path: (x / 7.5) - 1 for all path_dim
        normalized_original_paths = []
        for path in original_paths:
            norm_path = (np.array(path) / 7.5) - 1
            normalized_original_paths.append(norm_path)
        original_paths = normalized_original_paths

        generated_path = diff.sample(vis_model, map_tensors.to(device),robots_params.to(device), sampled_paths.shape[2], sampled_paths.shape[1]).squeeze(0) 
        generated_path = generated_path if n > 1 else generated_path

        #TODO do cpu paralelization 
        robot_params_renormalized = renormalize_robot(robots_params, dataset.robot_normalization, dataset.robot_variables)
        map_tensors = map_tensors.cpu().numpy()
        sampled_paths = np.transpose(sampled_paths.cpu().numpy(), (0, 2, 1))
        generated_path = np.transpose(generated_path.cpu().numpy(), (0, 2, 1))
        
        #TODO paralize this 


        #TODO not sure if this doesnt need .T 
        dynamic_paths = simulate_path_cuda(generated_path,robots_params, dataset) if dynamic else None 





        # if CPU_COUNT == 1:
        #     resampled_paths = resample_paths(generated_path, config.get('path_type', ''), original_paths)
        #     collisons_original.extend(calculate_validity_collisions(original_paths, map_tensors, robot_params_renormalized))
        #     collisons_model.extend(calculate_validity_collisions(resampled_paths, map_tensors, robot_params_renormalized))
        #     curvatures, bending_energies = calculate_turning_radius(original_paths)
        #     curvatures_original.extend(curvatures)
        #     bending_energies_original.extend(bending_energies)
        #     curvatures, bending_energies = calculate_turning_radius(resampled_paths)
        #     curvatures_model.extend(curvatures)
        #     bending_energies_model.extend(bending_energies)
        #     path_lengths_original.extend(calculate_path_length(original_paths))
        #     path_lengths_model.extend(calculate_path_length(resampled_paths))
        #     visualize_results(map_tensors, robot_params_renormalized, dynamic_paths, original_paths, model_resampled_paths=resampled_paths, model_path_points=generated_path,axes=axes, save_dir=save_dir)
        # else:
        #     #rebatching 
        with Pool(CPU_COUNT) as pool:
            # Resample paths in parallel for each path if needed
            resampled_paths = pool.starmap(
                resample_single_path,
                [(generated_path[i], config.get('path_type', ''), original_paths[i]) for i in range(len(generated_path))]
            )

            for result in pool.starmap(calculate_path_length, [( [original_paths[i]], ) for i in range(len(original_paths))]):
                path_lengths_original.extend(result)
            for result in pool.starmap(calculate_path_length, [( [resampled_paths[i]], ) for i in range(len(resampled_paths))]):
                path_lengths_model.extend(result)

            for result in pool.starmap(calculate_validity_collisions, [( [original_paths[i]], map_tensors[i:i+1], {k: np.array([v[i]]) for k, v in robot_params_renormalized.items()} ) for i in range(len(original_paths))]):
                collisons_original.extend(result)
            for result in pool.starmap(calculate_validity_collisions, [( [resampled_paths[i]], map_tensors[i:i+1], {k: np.array([v[i]]) for k, v in robot_params_renormalized.items()} ) for i in range(len(resampled_paths))]):
                collisons_model.extend(result)
            for curvatures, bending_energies in pool.starmap(calculate_turning_radius, [( [original_paths[i]], ) for i in range(len(original_paths))]):
                curvatures_original.extend(curvatures)
                bending_energies_original.extend(bending_energies)
            for curvatures, bending_energies in pool.starmap(calculate_turning_radius, [( [resampled_paths[i]], ) for i in range(len(resampled_paths))]):
                curvatures_model.extend(curvatures)
                bending_energies_model.extend(bending_energies)

            # collisons_original.extend(pool.starmap(calculate_validity_collisions, [( [original_paths[i]], map_tensors[i:i+1], {k: np.array([v[i]]) for k, v in robot_params_renormalized.items()} ) for i in range(len(original_paths))]))
            # collisons_model.extend(pool.starmap(calculate_validity_collisions, [( [resampled_paths[i]], map_tensors[i:i+1], {k: np.array([v[i]]) for k, v in robot_params_renormalized.items()} ) for i in range(len(resampled_paths))]))
            # curvatures_bending = pool.starmap(calculate_turning_radius, [( [original_paths[i]], ) for i in range(len(original_paths))])
            # for curvatures, bending_energies in curvatures_bending:
            #     curvatures_original.extend(curvatures)
            #     bending_energies_original.extend(bending_energies)
            # curvatures_bending = pool.starmap(calculate_turning_radius, [( [resampled_paths[i]], ) for i in range(len(resampled_paths))])
            # for curvatures, bending_energies in curvatures_bending:
            #     curvatures_model.extend(curvatures)
            #     bending_energies_model.extend(bending_energies)
            # path_lengths_original.extend(pool.starmap(calculate_path_length, [( [original_paths[i]], ) for i in range(len(original_paths))]))
            # path_lengths_model.extend(pool.starmap(calculate_path_length, [( [resampled_paths[i]], ) for i in range(len(resampled_paths))]))
            visualize_results(map_tensors, robot_params_renormalized, dynamic_paths, original_paths, model_resampled_paths=resampled_paths, model_path_points=generated_path,axes=axes, save_dir=save_dir)
            # print(path_lengths_model,collisons_model, curvatures_model, bending_energies_model)
    dt = datetime.datetime.now() - start_time
    print(f"Visualization completed in {dt}")







    if args.save:
        with open(f"{save_dir}/metrics.txt", 'w') as f:
            f.write('MODEL: \n')
            f.write(f"Average Path Length: {np.mean(path_lengths_model):.8f}\n")
            f.write(f"Collision Rate: {np.mean(collisons_model):.8f}\n")
            f.write(f"Average Curvature: {np.mean(curvatures_model):.8f}\n")
            f.write(f"Average Bending Energy: {np.mean(bending_energies_model):.8f}\n\n")
            f.write('ORIGINAL: \n')
            f.write(f"Average Path Length: {np.mean(path_lengths_original):.8f}\n")
            f.write(f"Collision Rate: {np.mean(collisons_original):.8f}\n")
            f.write(f"Average Curvature: {np.mean(curvatures_original):.8f}\n")
            f.write(f"Average Bending Energy: {np.mean(bending_energies_original):.8f}\n")

        np.savetxt(f"{save_dir}/metrics_model.csv", np.array([path_lengths_model, collisons_model, curvatures_model, bending_energies_model]).T, delimiter=",", header="path_length,collision,curvature,bending_energy", comments='')
        np.savetxt(f"{save_dir}/metrics_original.csv", np.array([path_lengths_original, collisons_original, curvatures_original, bending_energies_original]).T, delimiter=",", header="path_length,collision,curvature,bending_energy", comments='')






    pass 
import datetime

if __name__ == "__main__":






    pared_url = parse_run_url(args.run_url)

   
    api = wandb.Api()



    if 'runs' in args.run_url:
        visualize_model(api.run(parse_run_url(args.run_url)) )
    else:
        sweep = api.sweep(parse_run_url(args.run_url))
        sweep_name = sweep.name  
        print(f"Sweep name: {sweep_name}")
        for run in sweep.runs:
            visualize_model(run, sweep_name)


    