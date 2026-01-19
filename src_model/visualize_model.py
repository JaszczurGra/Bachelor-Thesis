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
parser.add_argument('--save_plots', action='store_true', help='Whether to save the plots as images instead of displaying them')

args = parser.parse_args()

matplotlib.use('Agg' if args.save_plots else 'TkAgg') 

import matplotlib.pyplot as plt
import re

def parse_run_url(url):
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/(?:sweeps/[^/]+/)?runs/([^/?]+)", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    else:
        raise ValueError("Invalid wandb run URL format.")


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

        ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'b-', linewidth=2, label='Robot')
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
            prev_state[3, :] >= _lateral_force_min_v,
            robot_params["wheelbase"] * robot_params["mu_static"] * 9.81 / prev_state[3, :]**2 * torch.sign(control[:,1]),
            torch.tan(torch.clamp(control[:, 1], -robot_params["max_steering_at_zero_v"], robot_params["max_steering_at_zero_v"]))
        )
        dx = prev_state[3, :] * torch.cos(prev_state[2, :])
        dy = prev_state[3, :] * torch.sin(prev_state[2, :])
        dtheta = (prev_state[:, 3] / robot_params["wheelbase"]) * angle
        dv = control[:, 0]
        # Update state
        states[:, 0, t] = prev_state[:, 0] + dx * dt
        states[:, 1, t] = prev_state[:, 1] + dy * dt
        states[:, 2, t] = prev_state[:, 2] + dtheta * dt
        states[:, 3, t] = prev_state[:, 3] + dv * dt

    path = (torch.cat([states, controls], dim=1) - norm_mins) / norm_ranges * 2 - 1

    return path.cpu().numpy()
    
def simulate_path(path,robot_params, dataset):
    

    robot_params = {key: val for key, val in zip(dataset.robot_normalization, robot_params)}
    
    _lateral_force_min_v = math.sqrt(robot_params["wheelbase"] * robot_params["mu_static"] * 9.81 / math.tan(robot_params["max_steering_at_zero_v"]) ) 
    result = [path[0]]
    def propagate(state, control, result):
     
        """
        State: [x, y, theta, v]
        Control: [acceleration, steering_angle]
        F_l= mvv/r < mu * g 
        """
        angle = math.copysign(robot_params["wheelbase"] * robot_params["mu_static"] * 9.81 / state[3]**2, control[1])  if state[3] >= _lateral_force_min_v else math.tan(np.clip(control[1], -robot_params["max_steering_at_zero_v"], robot_params["max_steering_at_zero_v"]))
        # self._debug_counter += 1
        # if self._debug_counter % 100 == 0:
        #     print(angle, math.atan(angle) * 180/math.pi)
            # print(self._debug_counter / 1000000 , 'MIL propagation steps')
        result[0] =  state[3] * math.cos(state[2])  
        result[1] = state[3] * math.sin(state[2])  
        # result[2] = (state[3] / self.robot.wheelbase) * math.tan(np.clip(control[1], -MAX_DELTA, MAX_DELTA)) 
        result[2] = (state[3] / robot_params["wheelbase"]) *  angle 
        result[3] = control[0]

    for i in range(1, len(path)-1):
        state = result[-1][:4]  # [x, y, theta, v]
        control = path[i][4:6]  # [acceleration, delta]
        next_state = [0, 0, 0, 0]
        propagate(state, control, next_state)
        new_x = state[0] + next_state[0] * dt
        new_y = state[1] + next_state[1] * dt
        new_theta = state[2] + next_state[2] * dt
        new_v = state[3] + next_state[3] * dt
        result.append([new_x, new_y, new_theta, new_v] + path[i][4:].tolist())

    return result


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



