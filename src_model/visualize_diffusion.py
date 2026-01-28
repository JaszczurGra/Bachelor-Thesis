import math
import os
import io
from pyexpat import model
import numpy as np
import torch
import wandb
import argparse
import matplotlib

from train import PathDataset
from model import DiffusionDenoiser
import cv2 #opencv-python

import datetime
import re 
#/mnt/storage_3/home/jaszczur/pl0467-01/scratch/jaszczur/Bachelor-Thesis/models
from collections import defaultdict


import argparse
from matplotlib.path import Path as MplPath
parser = argparse.ArgumentParser()
parser.add_argument('--run_url', type=str, required=True, help='WandB run URL to load the model from')
parser.add_argument('-m', '--max_dataset_length', type=int, default=None, help='Maximum number of samples to load from the dataset for debuging')

parser.add_argument('-n', '--num_plots', type=int, default=1, help='Number of plots in the path')
#TODO make this a number to save n number of singular plots 
parser.add_argument('--save', action='store_true', help='Whether to save the plots as images instead of displaying them')

parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing paths')
parser.add_argument('--viz_interval', type=int, default=250, help='Interval of diffusion steps to visualize')
parser.add_argument('--skip', type=int, default=25, help='Skip every n paths when visualizing diffusion process')

args = parser.parse_args()
matplotlib.use('Agg' if args.save else 'TkAgg') 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#cubic:64
#python src_model/visualize_diffusion.py --run_url "https://wandb.ai/j-boro-poznan-university-of-technology/Bachelor-Thesis-src_model/sweeps/xrqywefs/runs/jpc89t7y?nw=nwuserjaszczurgra" -m 5 --save
#beysian:15
#python src_model/visualize_diffusion.py --run_url "https://wandb.ai/j-boro-poznan-university-of-technology/Bachelor-Thesis-src_model/sweeps/xrqywefs/runs/szwf88tt?nw=nwuserjaszczurgra" -m 5 --save

#950
INTERVALS = [1,900, 975, 1000]

INTERVALS_A = [1000 - i for i in INTERVALS]
INTERVALS = [1,800, 900,950, 980, 1000]
INTERVALS_B = [1000 - i for i in INTERVALS]
current_drawing  = 0 

def draw(x_batched,  map_tensors,save_dir):
    global current_drawing
    n_batches = len(next(iter(x_batched.values())))
    steps = len(x_batched)
    for i in range(n_batches):
        map = map_tensors[i]

        # fig, axes = plt.subplots(1, steps, figsize=(steps * 5, 1 * 5))

        w = steps // 2 
        h = steps // w 
        fig, axes = plt.subplots(h,w, figsize=(w*5,h*5))



        plt.tight_layout()

        axes = axes.flatten() if steps > 1 else [axes]




        fig.subplots_adjust(wspace=0.5,hspace=0.5)  
        for s in range(steps):
            # Arrange axes in a "snake" order: 0,1,3,2 for a 2x2 grid
            # Arrange axes in a "snake" order for any grid size
            row = s // w
            col = s % w
            idx = row * w + (w - 1 - col) * (row   % 2) + col * ((row + 1)% 2) 
            ax = axes[idx]
            step_key = list(x_batched.keys())[s]
            x = x_batched[step_key][i]
            ax.imshow(map[0], cmap='gray', extent=[-1,1,-1,1], origin='lower')
            ax.plot(x[0,:], x[1, :], marker='o')
            ax.set_title(f'STEP: {step_key}')
        for s in range(steps - 1):
            row_from = s // w
            col_from = s % w if row_from % 2 == 0 else w - 1 - (s % w)
            ax_from = axes[row_from * w + col_from]
            row_to = (s + 1) // w
            col_to = (s + 1) % w if row_to % 2 == 0 else w - 1 - ((s + 1) % w)
            ax_to = axes[row_to * w + col_to]
            bbox_from = ax_from.get_position()
            bbox_to = ax_to.get_position()

            if row_from == row_to :
                if col_to > col_from:
                    # Left to right
                    start = (bbox_from.x1 + 0.01, (bbox_from.y0 + bbox_from.y1) / 2)
                    end = (bbox_to.x0 - 0.045, (bbox_to.y0 + bbox_to.y1) / 2)
                else:
                    # Right to left
                    # bbox_from, bbox_to = bbox_to, bbox_from
                    start = (bbox_from.x0 - 0.045, (bbox_from.y0 + bbox_from.y1) / 2)
                    end = (bbox_to.x1 + 0.01, (bbox_to.y0 + bbox_to.y1) / 2)
            # If moving down to next row
            else: 
                start = ((bbox_from.x0 + bbox_from.x1) / 2, bbox_from.y0 - 0.025)
                end = ((bbox_to.x0 + bbox_to.x1) / 2, bbox_to.y1 + 0.025)


            arrow = mpatches.FancyArrowPatch(
                start, end,
                transform=fig.transFigure,
                arrowstyle='->', color='gray', linewidth=2, mutation_scale=20
            )
            fig.patches.append(arrow)

        # plt.show()
        current_drawing += 1
        if args.save:
            plt.savefig(f"{save_dir}/diffusion_{steps}_{current_drawing}.pdf")  
            plt.close()
        else:
            plt.ion()
            plt.show(block=False)
            plt.pause(2)
    #     for step in x_batched.keys():
    #         x = x_batched[step][i]


    # for x_steps,map in zip(x_batched, map_tensor):
    #     for x, i in zip(x_steps, steps):    
    #     # 2, N 

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



class DiffusionManager:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def add_noise(self, x_start, t):
        noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise

    def sample(self, model, map_cond,robot_params, path_len, path_params_len, save_dir, intervals):
        to_draw = defaultdict(list)
        model.eval()
        with torch.no_grad():
            batch_size = map_cond.shape[0]
            x = torch.randn((batch_size, path_params_len, path_len), device=self.device)
            
            for i in reversed(range(self.timesteps)):
                t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
                predicted_noise = model(x, t, map_cond, robot_params)
                alpha_t = self.alphas[i]
                alpha_cumprod_t = self.alphas_cumprod[i]
                beta_t = self.betas[i]
                coef1 = 1 / torch.sqrt(alpha_t)
                coef2 = beta_t / torch.sqrt(1 - alpha_cumprod_t)
                mean = coef1 * (x - coef2 * predicted_noise)
                if i > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(beta_t)
                    x = mean + sigma * noise
                else:
                    x = mean
                # CRITICAL: Clamp to prevent explosion
                x = torch.clamp(x, -3, 3)  # Allow some wiggle room

                if i in intervals:
                    to_draw[self.timesteps - i] = torch.clamp(x,-1,1).detach().cpu().numpy()
            
            draw(to_draw, map_cond.detach().cpu().numpy(), save_dir)
                        

            x = torch.clamp(x, -1, 1)
            
        return x


def visualize_model(run, sweep_name=None):
    if run.state == 'failed':
        print(f"Skipping failed run: {run.name}")
        return

    print(f"Visualizing results for run: {run.name}")

    config = run.config

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dynamic = config.get('dynamic', False)

    n_maps = config["n_maps"] if args.max_dataset_length is None else min(config["n_maps"], args.max_dataset_length)
    
    dataset = PathDataset(config["dataset_path"], n_maps, config["map_resolution"], config.get('path_type', None),dynamic)

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
        vis_model.load_state_dict(torch.load(best_model_path))
    else:
        print('DIDN"T LOAD MODEL IT"S NOT LOCALY SAVED')



    #TODO custom_dataset only works for sweeps 
    save_dir = '/'.join(["visualizations"] + ([f"sweep_{sweep_name}_{args.custom_dataset.split('/')[-1]}"] if sweep_name else []) + [f"{run.name}_{config.get('path_type', '')}_layers:{config['model']['num_internal_layers']}"])
    print('Save dir:', save_dir)
    print('Dynamic:',dynamic, end='\n\n')
    if args.save:
        os.makedirs(save_dir, exist_ok=True)


    n =  args.num_plots
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    # set n of subplots to n 
    #TODO move this into the visualize to open new one 

    idxs = list(range(0, len(dataset), args.skip))


    #TODO parallize calculations 
    for b in range(0, len(idxs), args.batch_size):
        batch_idxs = idxs[b : b + args.batch_size]
        map_tensors, robots_params, sampled_paths  = [torch.stack(tensors) for tensors in zip(*[dataset[i] for i in batch_idxs])]
        original_paths = dataset.original_paths[batch_idxs]  # [B, path_len, path_dim]
        # Normalize each path: (x / 7.5) - 1 for all path_dim
        # normalized_original_paths = []
        # for path in original_paths:
        #     norm_path = (np.array(path) / 7.5) - 1
        #     normalized_original_paths.append(norm_path)
        # original_paths = normalized_original_paths

        generated_path = diff.sample(vis_model, map_tensors.to(device),robots_params.to(device), sampled_paths.shape[2], sampled_paths.shape[1], save_dir, INTERVALS_A).squeeze(0) 
        generated_path = diff.sample(vis_model, map_tensors.to(device),robots_params.to(device), sampled_paths.shape[2], sampled_paths.shape[1], save_dir , INTERVALS_B).squeeze(0) 

        # generated_path = generated_path if n > 1 else generated_path

        #TODO do cpu paralelization 
        # robot_params_renormalized = renormalize_robot(robots_params, dataset.robot_normalization, dataset.robot_variables)
        # map_tensors = map_tensors.cpu().numpy()
        # sampled_paths = np.transpose(sampled_paths.cpu().numpy(), (0, 2, 1))
        # generated_path = np.transpose(generated_path.cpu().numpy(), (0, 2, 1))`
        

           
    dt = datetime.datetime.now() - start_time
    print(f"Visualization completed in {dt}")







def parse_run_url(url):
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/(?:sweeps/[^/]+/)?runs/([^/?]+)", url)    if 'runs' in url else re.search(r"wandb\.ai/([^/]+)/([^/]+)/sweeps/([^/?#]+)", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    else:
        raise ValueError("Invalid wandb run URL format.")


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
