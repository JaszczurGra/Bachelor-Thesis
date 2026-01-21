import math
import os
import io
import numpy as np
import torch
import wandb
import argparse
import matplotlib

from train import DiffusionManager, PathDataset
from model import DiffusionDenoiser

#for coping model scp -r jaszczur@eagle:/mnt/storage_3/home/jaszczur/pl0467-01/scratch/jborowiecki/Bachelor-Thesis/models/crisp-sweep-26 models/

parser = argparse.ArgumentParser()
parser.add_argument('--run_url', type=str, required=True, help='WandB run URL to load the model from')
parser.add_argument('-m', '--max_dataset_length', type=int, default=None, help='Maximum number of samples to load from the dataset')
parser.add_argument('-n', '--num_plots', type=int, default=4, help='Number of plots in the path')
#TODO make this a number to save n number of singular plots 
parser.add_argument('--save_plots', action='store_true', help='Whether to save the plots as images instead of displaying them')
parser.add_argument('--custom_dataset', type=str, default=None, help='Path to a custom dataset to visualize')
args = parser.parse_args()

matplotlib.use('Agg' if args.save_plots else 'TkAgg') 

import matplotlib.pyplot as plt
import re
from matplotlib.patches import Polygon

def parse_run_url(url):
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/(?:sweeps/[^/]+/)?runs/([^/?]+)", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    else:
        raise ValueError("Invalid wandb run URL format.")


def renormalize_robot(robot_params, robot_normalization, robot_variables):
    norm_ranges = np.array([robot_normalization[var][1] - robot_normalization[var][0] for var in robot_variables])
    norm_mins = np.array([robot_normalization[var][0] for var in robot_variables])
    renormalized = 0.5 * (robot_params + 1) * norm_ranges + norm_mins
    return {key: value for key, value in zip(robot_variables, renormalized)}

def check_validity(resampled_path, map_tensor, robot_params):
    # Check if any point in the path collides with obstacles in the map
    # resampled_path: [T, path_dim]
    # map_tensor: [1, H, W]


    # Compute the robot rectangle corners in normalized coordinates
    robot_x, robot_y = resampled_path.T[0, 0], resampled_path.T[1, 0]
    robot_theta = resampled_path.T[2, 0]
    robot_width = ((robot_params[6] + 1) / 2 * (1 - 0.1) + 0.1) / 15
    robot_length = ((robot_params[7] + 1) / 2 * (1 - 0.1) + 0.1) / 15

    corners = np.array([
        [-robot_length/2, -robot_width/2],
        [robot_length/2, -robot_width/2],
        [robot_length/2, robot_width/2],
        [-robot_length/2, robot_width/2]
    ])

    cos_theta = np.cos(robot_theta)
    sin_theta = np.sin(robot_theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    rotated_corners = corners @ rotation_matrix.T
    rotated_corners[:, 0] += robot_x
    rotated_corners[:, 1] += robot_y

    width, height = robot_width, robot_length

    # Check if any pixel inside the rotated rectangle is an obstacle
    H, W = map_tensor.shape[1], map_tensor.shape[2]
    rect_path = MplPath(rotated_corners)
    # Generate a grid of points covering the rectangle bounding box
    min_x, max_x = rotated_corners[:, 0].min(), rotated_corners[:, 0].max()
    min_y, max_y = rotated_corners[:, 1].min(), rotated_corners[:, 1].max()
    x_grid = np.linspace(min_x, max_x, num=int(np.ceil((max_x-min_x)*W/2)))
    y_grid = np.linspace(min_y, max_y, num=int(np.ceil((max_y-min_y)*H/2)))
    xx, yy = np.meshgrid(x_grid, y_grid)
    points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    inside = rect_path.contains_points(points)
    inside_points = points[inside]
    if inside_points.shape[0] == 0:
        return width, height  # fallback

    x_normalized = (inside_points[:, 0] + 1) / 2
    y_normalized = (inside_points[:, 1] + 1) / 2
    x_indices = np.clip((x_normalized * (W - 1)).astype(int), 0, W - 1)
    y_indices = np.clip((y_normalized * (H - 1)).astype(int), 0, H - 1)
    for xi, yi in zip(x_indices, y_indices):
        if map_tensor[0, yi, xi] < 0.5:
            return width, height

    path = resampled_path.T  # shape: [T, path_dim], assume x=0, y=1
    H, W = map_tensor.shape[1], map_tensor.shape[2]
    x_normalized = (path[:, 0] + 1) / 2  # Normalize to [0, 1]
    y_normalized = (path[:, 1] + 1) / 2  # Normalize to [0, 1]
    x_indices = np.clip((x_normalized * (W - 1)).astype(int), 0, W - 1)
    y_indices = np.clip((y_normalized * (H - 1)).astype(int), 0, H - 1)

    for xi, yi in zip(x_indices, y_indices):
        if map_tensor[0, yi, xi] < 0.5:  # Assuming obstacle pixels are < 0.5
            return False
    return True


def calculate_turning_radius(resampled_path):
    # resampled_path: [T, path_dim]
    # Calculate turning radius between three consecutive points
    # For two points, we need a third to define a circle; so use a sliding window of 3
    # Here, we compute radius for each triplet in the path
    radii = []
    # Transpose to shape [T, path_dim] if needed
    path = resampled_path.T  # shape: [T, path_dim], assume x=0, y=1

    # Vectorized calculation for turning radii (sum only, not individual radii)
    # This avoids explicit Python loops for speed.
    p1 = path[:-2, :2]
    p2 = path[1:-1, :2]
    p3 = path[2:, :2]

    # Calculate determinants for each triplet
    A = np.linalg.det(np.stack([
        np.concatenate([p1, np.ones((p1.shape[0], 1))], axis=1),
        np.concatenate([p2, np.ones((p2.shape[0], 1))], axis=1),
        np.concatenate([p3, np.ones((p3.shape[0], 1))], axis=1)
    ], axis=1))

    # Avoid division by zero
    mask = np.abs(A) >= 1e-8

    B = -np.linalg.det(np.stack([
        np.concatenate([(p1[:, 0]**2 + p1[:, 1]**2)[:, None], p1[:, 1:2], np.ones((p1.shape[0], 1))], axis=1),
        np.concatenate([(p2[:, 0]**2 + p2[:, 1]**2)[:, None], p2[:, 1:2], np.ones((p2.shape[0], 1))], axis=1),
        np.concatenate([(p3[:, 0]**2 + p3[:, 1]**2)[:, None], p3[:, 1:2], np.ones((p3.shape[0], 1))], axis=1)
    ], axis=1))

    C = np.linalg.det(np.stack([
        np.concatenate([(p1[:, 0]**2 + p1[:, 1]**2)[:, None], p1[:, 0:1], np.ones((p1.shape[0], 1))], axis=1),
        np.concatenate([(p2[:, 0]**2 + p2[:, 1]**2)[:, None], p2[:, 0:1], np.ones((p2.shape[0], 1))], axis=1),
        np.concatenate([(p3[:, 0]**2 + p3[:, 1]**2)[:, None], p3[:, 0:1], np.ones((p3.shape[0], 1))], axis=1)
    ], axis=1))

    D = -np.linalg.det(np.stack([
        np.concatenate([(p1[:, 0]**2 + p1[:, 1]**2)[:, None], p1[:, 0:1], p1[:, 1:2]], axis=1),
        np.concatenate([(p2[:, 0]**2 + p2[:, 1]**2)[:, None], p2[:, 0:1], p2[:, 1:2]], axis=1),
        np.concatenate([(p3[:, 0]**2 + p3[:, 1]**2)[:, None], p3[:, 0:1], p3[:, 1:2]], axis=1)
    ], axis=1))

    center_x = np.zeros_like(A)
    center_y = np.zeros_like(A)
    radius = np.full_like(A, np.inf)

    center_x[mask] = -B[mask] / (2 * A[mask])
    center_y[mask] = -C[mask] / (2 * A[mask])
    radius[mask] = np.sqrt(center_x[mask]**2 + center_y[mask]**2 - D[mask] / A[mask])

    # If you only want the sum:
    # / path lenght
    total_radius = np.sum(radius) / path.shape[0]


    return total_radius
    for i in range(1, path.shape[0] - 1):
        p1 = path[i - 1][:2]
        p2 = path[i][:2]
        p3 = path[i + 1][:2]
        # Calculate circle from three points
        A = np.linalg.det([
            [p1[0], p1[1], 1],
            [p2[0], p2[1], 1],
            [p3[0], p3[1], 1]
        ])
        if abs(A) < 1e-8:
            radii.append(np.inf)
            continue
        B = -np.linalg.det([
            [p1[0]**2 + p1[1]**2, p1[1], 1],
            [p2[0]**2 + p2[1]**2, p2[1], 1],
            [p3[0]**2 + p3[1]**2, p3[1], 1]
        ])
        C = np.linalg.det([
            [p1[0]**2 + p1[1]**2, p1[0], 1],
            [p2[0]**2 + p2[1]**2, p2[0], 1],
            [p3[0]**2 + p3[1]**2, p3[0], 1]
        ])
        D = -np.linalg.det([
            [p1[0]**2 + p1[1]**2, p1[0], p1[1]],
            [p2[0]**2 + p2[1]**2, p2[0], p2[1]],
            [p3[0]**2 + p3[1]**2, p3[0], p3[1]]
        ])
        center_x = -B / (2 * A)
        center_y = -C / (2 * A)
        radius = np.sqrt(center_x**2 + center_y**2 - D / A)
        radii.append(radius)
    return sum(np.array(radii)) / path.shape[0]
    pass 

#TODO paralize this for calcuation so only few maps are visualized and then all the others are just calculated 
#TODO make batching 
def visualize_results(model, diff, dataset,device, axes, idxs, dynamic=False):
    n = len(idxs)
    samples = [dataset[i] for i in idxs]
    map_tensor, robot, real_path = [torch.stack(tensors) for tensors in zip(*samples)]
    # map_tensor, robot, real_path = dataset[idx]
    map_tensor, robot, real_path = map_tensor.to(device), robot.to(device),real_path.to(device)
    
    
    #TODO add simulated path from control using propaget function 


    generated_path = diff.sample(model, map_tensor,robot, real_path).squeeze(0) 
    generated_path = generated_path if n > 1 else generated_path.unsqueeze(0)
    gen_path = generated_path.cpu().numpy() 

    real_path = real_path.cpu().numpy()
    map_img = map_tensor.cpu().numpy()


    dynamic_path = None
    if dynamic:
        dynamic_path = simulate_path_cuda(generated_path,robot, dataset)



    for i in range(n):
        ax = axes[i]
        ax.imshow(map_img[i,0], cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
        ax.plot(real_path[i,0,:], real_path[i,1,:], 'g--', linewidth=2, label='GT')

        #TODO add plotting of the robot at start position of gt and generated path 
        robot_x, robot_y = real_path[i, 0, 0], real_path[i, 1, 0]
        robot_theta = real_path[i, 2, 0]
        robot_width = ((robot[i, 6].item() + 1) / 2 * (1 - 0.1) + 0.1) / 15
        robot_length = ((robot[i, 7].item() + 1) / 2 * (1 - 0.1) + 0.1) / 15

        corners = np.array([
            [-robot_length/2, -robot_width/2],
            [robot_length/2, -robot_width/2],
            [robot_length/2, robot_width/2],
            [-robot_length/2, robot_width/2],
            [-robot_length/2, -robot_width/2]
        ])

        cos_theta = np.cos(robot_theta)
        sin_theta = np.sin(robot_theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_corners = corners @ rotation_matrix.T
        rotated_corners[:, 0] += robot_x
        rotated_corners[:, 1] += robot_y

        # Draw the robot as a filled rectangle (patch)
        rect = Polygon(rotated_corners, closed=True, facecolor='cyan', edgecolor='b', alpha=0.5, label='Robot')
        ax.add_patch(rect)

        # ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'b-', linewidth=2, label='Robot')
        ax.plot(gen_path[i,0,:], gen_path[i,1,:], 'r-', linewidth=2, label='Diffusion')


        if dynamic_path is not None :
            ax.plot(dynamic_path[i,0,:], dynamic_path[i,1,:], 'm-', linewidth=2, label='Dynamic Simulated')
        # simulated_path = simulate_path(generated_path[i], robot[i].cpu().numpy(), dt=0.01)
        # # ax.plot(simulated_path[0,:], simulated_path[1,:], 'b-', linewidth=2, label='Simulated from gt')
        
        ax.legend()
        ax.set_title(idxs[i])


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
    

import argparse
from matplotlib.path import Path as MplPath

if __name__ == "__main__":





    pared_url = parse_run_url(args.run_url)
    # print(f"Parsed url: {pared_url}")
   
    api = wandb.Api()
    run = api.run(pared_url) # fill in your actual values
    config = run.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dynamic = config.get('dynamic', False)

    n_maps = config["n_maps"] if args.max_dataset_length is None else min(config["n_maps"], args.max_dataset_length)
    dataset = PathDataset(config["dataset_path"], n_maps, config["map_resolution"], config["path_length"],dynamic)
    #TODO nadpisac mape zeby byl na tych co nie widzial na jakis inny folder

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
        vis_model.load_state_dict(torch.load(best_model_path))
    # # print(run.config)

    # n_maps = config.n_maps if args.max_dataset_length is None else min(config.n_maps, args.max_dataset_length) 
    # dataset = PathDataset(config.dataset_path,n_maps, config.map_resolution,config.path_length)
    # # dynamic=config.get('dynamic', False)
    # diff = DiffusionDenoiser(state_dim=dataset.path_dim,robot_param_dim=dataset.robot_dim,map_size=dataset.maps.shape[2], map_feat_dim=config.model['map_feat_dim'], robot_feat_dim=config.model['robot_feat_dim'], time_feat_dim=config.model['time_feat_dim'], num_internal_layers=config.model['num_internal_layers'], base_layer_dim=config.model['base_layer_dim'], verbose=False).to(device)


    # best_model_path = os.path.join('models', run.name , 'best_model_checkpoint.pth')
    # vis_model = DiffusionDenoiser(state_dim=dataset.path_dim,robot_param_dim=dataset.robot_dim,map_size=dataset.maps.shape[2], map_feat_dim=config.model['map_feat_dim'], robot_feat_dim=config.model['robot_feat_dim'], time_feat_dim=config.model['time_feat_dim'], num_internal_layers=config.model['num_internal_layers'], base_layer_dim=config.model['base_layer_dim'], verbose=False).to(device)
    # if os.path.exists(best_model_path):
    #     vis_model.load_state_dict(torch.load(best_model_path))
    print(f"Visualizing results for run: {run.name}")
    print('Dynamic:',dynamic)
    n =  args.num_plots
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    # set n of subplots to n 
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten() if n > 1 else [axes]
    plt.tight_layout()

    save_dir = f"visualizations/{run.name}"
    if args.save_plots:
        os.makedirs(save_dir, exist_ok=True)


    for i in range(0, len(dataset), n):
        idxs = list(range(i, min(i + n, len(dataset))))
        visualize_results(vis_model, diff, dataset, device, axes, idxs,dynamic)
        if args.save_plots:
            plt.savefig(f"{save_dir}/visualization_{int(i//n)}.png")
        else:
            plt.show(block=False)
            plt.pause(0.1)
        for ax in axes:
            ax.clear()



