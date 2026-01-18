import os
import io
import numpy as np
import torch
import wandb
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--run_url', type=str, required=True, help='WandB run URL to load the model from')
args = parser.parse_args()

import re
def parse_run_url(url):
    if '/' in url and not url.startswith("http"):
        return url
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/runs/([^/?]+)", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    else:
        raise ValueError("Invalid wandb run URL format.")


def visualize_results(model, diff, dataset, epoch,device):
    max_n = 16
    idxs = np.random.choice(len(dataset), size=min(max_n, len(dataset)), replace=False).astype(int)
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
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    # set n of subplots to n 
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten() if n > 1 else [axes]

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

        # simulated_path = simulate_path(generated_path[i], robot[i].cpu().numpy(), dt=0.01)
        # # ax.plot(simulated_path[0,:], simulated_path[1,:], 'b-', linewidth=2, label='Simulated from gt')
        
        ax.legend()
        ax.set_title(f"Epoch {epoch}")
    
    plt.tight_layout()
    #This should return fig so we can log it thogherter with epoch etc so ther nr of logs is equal to the nr of epochs
    wandb.log({"generated_path": wandb.Image(fig)})
    # plt.show(block=False)
    # plt.pause(5)
    plt.close(fig)


if __name__ == "__main__":

    pared_url = parse_run_url(args.run_url)
    # print(f"Parsed url: {pared_url}")
   
    api = wandb.Api()
    run = api.run(pared_url) # fill in your actual values
    # print(run.config)

    best_model_path = os.path.join('models', run.name , 'best_model_checkpoint.pth')
    vis_model = DiffusionDenoiser(state_dim=dataset.path_dim,robot_param_dim=dataset.robot_dim,map_size=dataset.maps.shape[2], map_feat_dim=config.model['map_feat_dim'], robot_feat_dim=config.model['robot_feat_dim'], time_feat_dim=config.model['time_feat_dim'], num_internal_layers=config.model['num_internal_layers'], base_layer_dim=config.model['base_layer_dim'], verbose=False).to(device)
    if os.path.exists(best_model_path):
        vis_model.load_state_dict(torch.load(best_model_path))
