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

    

#/mnt/storage_3/home/jaszczur/pl0467-01/scratch/jaszczur/Bachelor-Thesis/models



import argparse
from matplotlib.path import Path as MplPath
parser = argparse.ArgumentParser()
parser.add_argument('--run_url', type=str, required=True, help='WandB run URL to load the model from')
parser.add_argument('-m', '--max_dataset_length', type=int, default=None, help='Maximum number of samples to load from the dataset for debuging')
parser.add_argument('--custom_dataset', type=str, default="", help='Path to a custom dataset to visualize')

parser.add_argument('--n_viz', type=int, default=300, help='Number of visualizations imgs to generate ')
parser.add_argument('-n', '--num_plots', type=int, default=1, help='Number of plots in the path')
#TODO make this a number to save n number of singular plots 
parser.add_argument('--save', action='store_true', help='Whether to save the plots as images instead of displaying them')

parser.add_argument('--batch_size', type=int, default=512, help='Batch size for processing paths')
args = parser.parse_args()
matplotlib.use('Agg' if args.save else 'TkAgg') 

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
    
    dataset = PathDataset(args.custom_dataset if args.custom_dataset != "" else config["dataset_path"], n_maps, config["map_resolution"], config.get('path_type', None),dynamic)

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



    #TODO custom_dataset only works for sweeps 
    save_dir = '/'.join(["visualizations"] + ([f"sweep_{sweep_name}_{args.custom_dataset.split('/')[-1]}"] if sweep_name else []) + [f"{run.name}_{config.get('path_type', '')}_layers:{config['model']['num_internal_layers']}"])
    print('Save dir:', save_dir)
    print('Dynamic:',dynamic, end='\n\n')
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




    n =  args.num_plots
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    # set n of subplots to n 
    #TODO move this into the visualize to open new one 
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten() if n > 1 else [axes]
    plt.tight_layout()
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
        

        #TODO not sure if this doesnt need .T 
        dynamic_paths = simulate_path_cuda(generated_path,robots_params, dataset) if dynamic else None 

        with Pool(CPU_COUNT) as pool:
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
           
            visualize_results(map_tensors, robot_params_renormalized, dynamic_paths, original_paths, model_resampled_paths=resampled_paths, model_path_points=generated_path,axes=axes, save_dir=save_dir)
    dt = datetime.datetime.now() - start_time
    print(f"Visualization completed in {dt}")







    return 

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
        merge_metrics(sweep_name)
