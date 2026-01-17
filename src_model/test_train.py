import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import json
import cv2
import math
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb  # <--- NEW IMPORT

# Import your model
from test_model import ConditionalUnet1D

# --- Configuration ---
CONFIG = {
    "data_dir": "slurm_data/test_dataset",
    "seq_len": 128,
    "map_size": 64,
    "batch_size": 16,
    "epochs": 10000,
    "lr": 1e-4,
    "viz_interval": 500,  # Upload image every N epochs
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

# --- 1. Utilities ---
def smooth_path(path_array, kernel_size=9):
    kernel = np.ones(kernel_size) / kernel_size
    smoothed = path_array.copy()
    smoothed[0, :] = np.convolve(path_array[0, :], kernel, mode='same')
    smoothed[1, :] = np.convolve(path_array[1, :], kernel, mode='same')
    half_k = kernel_size // 2
    smoothed[:, :half_k] = path_array[:, :half_k]
    smoothed[:, -half_k:] = path_array[:, -half_k:]
    return smoothed

def resample_path(path, target_len):
    original_len = path.shape[0]
    idx_old = np.linspace(0, original_len - 1, num=original_len)
    idx_new = np.linspace(0, original_len - 1, num=target_len)
    new_path = np.zeros((target_len, path.shape[1]))
    for dim in range(path.shape[1]):
        new_path[:, dim] = np.interp(idx_new, idx_old, path[:, dim])
    return new_path

# --- 2. Dataset ---
class RobotPathDataset(Dataset):
    def __init__(self, data_dir, map_size=64, seq_len=128):
        self.data_dir = data_dir
        self.seq_len = seq_len
        
        map_path = os.path.join(data_dir, "map.png")
        if not os.path.exists(map_path):
            raise ValueError(f"map.png not found in {data_dir}")
            
        img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError("Failed to decode map.png")
        img = cv2.resize(img, (map_size, map_size))
        self.map_tensor = torch.FloatTensor(img).unsqueeze(0) / 255.0 
        
        self.json_files = sorted(glob.glob(os.path.join(data_dir, "path_*.json")))
        if len(self.json_files) == 0:
            raise ValueError(f"No path_*.json files found in {data_dir}")
        print(f"Found {len(self.json_files)} trajectories.")

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        file_path = self.json_files[idx]
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        robot_vec = torch.tensor([
            data['robot']['wheelbase'],
            data['robot']['acceleration'] / 10.0,
            data['planner']['start'][0] / 15.0,
            data['planner']['start'][1] / 15.0,
            data['planner']['goal'][0] / 15.0,
            data['planner']['goal'][1] / 15.0
        ], dtype=torch.float32)

        raw_path = np.array(data['path']) 
        processed_path = resample_path(raw_path, self.seq_len)
        processed_path = processed_path / 15.0 
        processed_path = (processed_path * 2) - 1 
        path_tensor = torch.FloatTensor(processed_path).transpose(0, 1)

        return path_tensor, self.map_tensor, robot_vec

# --- 3. Diffusion Scheduler ---
class DDPMScheduler:
    def __init__(self, num_timesteps=1000):
        self.num_timesteps = num_timesteps
        steps = torch.arange(num_timesteps + 1, dtype=torch.float32) / num_timesteps
        alpha_bar = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        self.beta = torch.clip(betas, 0.0001, 0.999).to(CONFIG["device"])
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def add_noise(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1. - self.alpha_hat[t])[:, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample(self, model, map_cond, vec_cond, shape):
        model.eval()
        with torch.no_grad():
            x = torch.randn(shape).to(CONFIG["device"])
            # We iterate silently to avoid spamming TQDM logs in WandB console
            for i in reversed(range(self.num_timesteps)):
                t = torch.tensor([i] * shape[0]).to(CONFIG["device"])
                predicted_noise = model(x, t, map_cond, vec_cond)
                
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                
                noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
                
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                x = torch.clamp(x, -1.2, 1.2)
        model.train()
        return x

# --- 4. Main Training Loop ---
def train():
    # --- WandB Init ---
    wandb.init(project="Motion Planning", config=CONFIG)
    
    # 1. Dataset
    full_dataset = RobotPathDataset(CONFIG["data_dir"], map_size=CONFIG["map_size"], seq_len=CONFIG["seq_len"])
    
    # Train/Val Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False)
    
    # 2. Setup
    model = ConditionalUnet1D(input_dim=7, cond_dim=6, channels=64).to(CONFIG["device"])
    scheduler = DDPMScheduler(num_timesteps=1000)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    loss_fn = nn.MSELoss()

    # --- Pre-Select Fixed Samples for Visualization ---
    # We grab 4 fixed samples from validation set to track progress over time
    # This creates a consistent "Timelapse" in WandB
    print("Preparing visualization batch...")
    vis_indices = torch.linspace(0, len(val_ds)-1, steps=4).long()
    vis_batch_maps = []
    vis_batch_conds = []
    vis_gt_paths = []
    
    for idx in vis_indices:
        p, m, c = val_ds[idx]
        vis_batch_maps.append(m)
        vis_batch_conds.append(c)
        vis_gt_paths.append(p.numpy()) # Store as numpy for plotting later

    # Stack for efficient batch inference
    vis_tensor_maps = torch.stack(vis_batch_maps).to(CONFIG["device"])
    vis_tensor_conds = torch.stack(vis_batch_conds).to(CONFIG["device"])
    
    # Load High-Res Map for Plotting background
    high_res_map = cv2.imread(os.path.join(CONFIG["data_dir"], "map.png"), cv2.IMREAD_GRAYSCALE)
    high_res_map = np.flipud(high_res_map)

    print(f"Starting Training on {CONFIG['device']}...")
    
    # 3. Training Loop
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_train_loss = 0
        
        for paths, maps, conds in train_loader:
            paths, maps, conds = paths.to(CONFIG["device"]), maps.to(CONFIG["device"]), conds.to(CONFIG["device"])
            
            t = torch.randint(0, scheduler.num_timesteps, (paths.shape[0],)).to(CONFIG["device"])
            noisy_paths, noise = scheduler.add_noise(paths, t)
            
            noise_pred = model(noisy_paths, t, maps, conds)
            loss = loss_fn(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        # Log Train Metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        wandb.log({
            "train_loss": avg_train_loss, 
            "learning_rate": current_lr,
            "epoch": epoch
        })

        # --- Validation Loop (Every 10 Epochs) ---
        if epoch % 10 == 0:
            model.eval()
            val_loss_acc = 0
            with torch.no_grad():
                for paths, maps, conds in val_loader:
                    paths, maps, conds = paths.to(CONFIG["device"]), maps.to(CONFIG["device"]), conds.to(CONFIG["device"])
                    t = torch.randint(0, scheduler.num_timesteps, (paths.shape[0],)).to(CONFIG["device"])
                    noisy_paths, noise = scheduler.add_noise(paths, t)
                    noise_pred = model(noisy_paths, t, maps, conds)
                    val_loss_acc += loss_fn(noise_pred, noise).item()
            
            avg_val_loss = val_loss_acc / len(val_loader)
            wandb.log({"val_loss": avg_val_loss, "epoch": epoch})
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

        # --- Visualization Loop (Every N Epochs) ---
        if epoch % CONFIG["viz_interval"] == 0:
            print(f"Generating visualization for epoch {epoch}...")
            
            # Run inference on our fixed batch
            gen_batch = scheduler.sample(model, vis_tensor_maps, vis_tensor_conds, (4, 7, CONFIG["seq_len"]))
            
            # Create Plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            axes = axes.flatten()
            
            for i in range(4):
                ax = axes[i]
                
                # Ground Truth
                gt_path = vis_gt_paths[i]
                gt_path = (gt_path + 1) / 2 * 15.0
                
                # Generated
                raw_path = gen_batch[i].squeeze().cpu().numpy()
                raw_path = (raw_path + 1) / 2 * 15.0
                smooth_p = smooth_path(raw_path)
                
                # Plot
                ax.imshow(high_res_map, cmap='gray', extent=[0, 15, 0, 15], origin='lower', alpha=0.5, zorder=0)
                ax.plot(gt_path[0, :], gt_path[1, :], label='Ground Truth', color='blue', linewidth=3, zorder=5)
                ax.plot(raw_path[0, :], raw_path[1, :], color='red', linestyle='--', alpha=0.3, zorder=6)
                ax.plot(smooth_p[0, :], smooth_p[1, :], label='Inference', color='red', linewidth=3, zorder=10)
                
                ax.set_xlim(0, 15); ax.set_ylim(0, 15)
                ax.set_title(f"Val Sample {i}")
                if i == 0: ax.legend()

            plt.suptitle(f"Epoch {epoch}", fontsize=16)
            plt.tight_layout()
            
            # Upload to WandB
            wandb.log({"Inference Samples": wandb.Image(fig), "epoch": epoch})
            
            # Close figure to prevent memory leak
            plt.close(fig)

    # Finish run
    wandb.finish()

if __name__ == "__main__":
    train()