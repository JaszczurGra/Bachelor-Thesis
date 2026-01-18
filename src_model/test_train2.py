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
import wandb

from test_model import ConditionalUnet1D

# --- Configuration ---
CONFIG = {
    "data_dir": "slurm_data/slurm_10_01_12-01-2026_00:07",
    "seq_len": 128,
    "map_size": 64,       # Training size
    "batch_size": 128,    # Crank this up! With RAM loading, you can handle massive batches.
    "epochs": 3000,
    "lr": 1e-4,
    "viz_interval": 100,
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

# --- 2. Dataset (RAM Cached) ---
class RobotPathDataset(Dataset):
    def __init__(self, root_dir, map_size=64, seq_len=128):
        self.samples = [] 
        
        print(f"Pre-loading dataset from {root_dir} into RAM...")
        
        if not os.path.exists(root_dir):
            raise ValueError(f"Directory {root_dir} not found")

        # 1. Find all map folders
        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # 2. Iterate and Load
        # We use a map_cache so we don't reload the same map image 50 times for 50 paths
        map_cache = {} 
        
        for d in tqdm(subdirs, desc="Loading Maps & Paths"):
            folder_path = os.path.join(root_dir, d)
            map_file = os.path.join(folder_path, "map.png")
            
            if not os.path.exists(map_file):
                continue
            
            # -- CACHE THE MAP TENSOR --
            # Load once per folder
            img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
            if img is None: continue
            
            # Store low-res tensor for training
            img_resized = cv2.resize(img, (map_size, map_size))
            map_tensor = torch.FloatTensor(img_resized).unsqueeze(0) / 255.0
            
            # Find paths
            json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
            
            for jf in json_files:
                try:
                    with open(jf, 'r') as f:
                        data = json.load(f)
                    
                    # -- PROCESS ROBOT VECTOR --
                    robot_vec = torch.tensor([
                        data['robot']['wheelbase'],
                        data['robot']['acceleration'] / 10.0,
                        data['planner']['start'][0] / 15.0,
                        data['planner']['start'][1] / 15.0,
                        data['planner']['goal'][0] / 15.0,
                        data['planner']['goal'][1] / 15.0
                    ], dtype=torch.float32)

                    # -- PROCESS PATH --
                    raw_path = np.array(data['path']) 
                    processed_path = resample_path(raw_path, seq_len)
                    processed_path = processed_path / 15.0 
                    processed_path = (processed_path * 2) - 1 
                    path_tensor = torch.FloatTensor(processed_path).transpose(0, 1)

                    # Store everything needed for __getitem__ in RAM
                    # We store map_file path str for visualization lookup later
                    self.samples.append({
                        "path": path_tensor,
                        "map": map_tensor,
                        "cond": robot_vec,
                        "map_path_str": map_file 
                    })
                except Exception as e:
                    print(f"Skipping bad file {jf}: {e}")

        print(f"Done. Loaded {len(self.samples)} trajectories into RAM.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Zero disk I/O here! Instant access.
        sample = self.samples[idx]
        return sample["path"], sample["map"], sample["cond"]

    def get_viz_info(self, idx):
        """Helper to get high-res map path for plotting"""
        return self.samples[idx]["map_path_str"]

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
            for i in tqdm(reversed(range(self.num_timesteps)), desc="Inference", leave=False):
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
    wandb.init(project="Motion Planning", config=CONFIG)
    
    # 1. Dataset
    try:
        full_dataset = RobotPathDataset(CONFIG["data_dir"], map_size=CONFIG["map_size"], seq_len=CONFIG["seq_len"])
    except Exception as e:
        print(f"Error: {e}")
        return

    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    # NOTE: num_workers=0 is usually FASTER for RAM datasets because there's no multiprocessing overhead
    # We set pin_memory=True to speed up transfer to GPU
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
    
    # 2. Setup
    model = ConditionalUnet1D(input_dim=7, cond_dim=6, channels=64).to(CONFIG["device"])
    scheduler = DDPMScheduler(num_timesteps=1000)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    loss_fn = nn.MSELoss()
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    # --- Pre-Select Visualization Batch ---
    print("Preparing visualization batch...")
    num_viz = min(4, val_size)
    vis_indices = torch.linspace(0, val_size-1, steps=num_viz).long()
    
    vis_data_store = [] 

    for idx in vis_indices:
        # Access the sample
        path_T, map_T, cond_T = val_ds[idx]
        
        # Get Original path string for high-res map
        original_idx = val_ds.indices[idx]
        map_path_str = full_dataset.get_viz_info(original_idx)
        
        high_res = cv2.imread(map_path_str, cv2.IMREAD_GRAYSCALE)
        high_res = np.flipud(high_res)
        
        vis_data_store.append({
            "map_T": map_T,
            "cond_T": cond_T,
            "gt_np": (path_T.numpy() + 1) / 2 * 15.0,
            "high_res": high_res
        })

    # 3. Training Loop
    print(f"Starting Training on {CONFIG['device']}...")
    for epoch in range(CONFIG["epochs"]):
        model.train()
        epoch_train_loss = 0
        
        # No tqdm here for speed - it can slow down very fast loops
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
        
        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss_acc = 0
        with torch.no_grad():
            for paths, maps, conds in val_loader:
                paths, maps, conds = paths.to(CONFIG["device"]), maps.to(CONFIG["device"]), conds.to(CONFIG["device"])
                t = torch.randint(0, scheduler.num_timesteps, (paths.shape[0],)).to(CONFIG["device"])
                noisy_paths, noise = scheduler.add_noise(paths, t)
                noise_pred = model(noisy_paths, t, maps, conds)
                val_loss_acc += loss_fn(noise_pred, noise).item()
        
        avg_val_loss = val_loss_acc / max(len(val_loader), 1)
        lr_scheduler.step(avg_val_loss)
        
        wandb.log({
            "train_loss": avg_train_loss, 
            "val_loss": avg_val_loss,
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": epoch
        })

        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

        # --- Visualization ---
        if epoch % CONFIG["viz_interval"] == 0:
            print(f"Generating visualization...")
            batch_maps = torch.stack([d["map_T"] for d in vis_data_store]).to(CONFIG["device"])
            batch_conds = torch.stack([d["cond_T"] for d in vis_data_store]).to(CONFIG["device"])
            
            gen_batch = scheduler.sample(model, batch_maps, batch_conds, (len(vis_data_store), 7, CONFIG["seq_len"]))
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            axes = axes.flatten()
            
            for i, data in enumerate(vis_data_store):
                ax = axes[i]
                raw = gen_batch[i].squeeze().cpu().numpy()
                raw = (raw + 1) / 2 * 15.0
                smooth = smooth_path(raw)
                
                ax.imshow(data["high_res"], cmap='gray', extent=[0, 15, 0, 15], origin='lower', alpha=0.5, zorder=0)
                ax.plot(data["gt_np"][0, :], data["gt_np"][1, :], label='GT', color='blue', linewidth=3, zorder=5)
                ax.plot(raw[0, :], raw[1, :], color='red', linestyle='--', alpha=0.3, zorder=6)
                ax.plot(smooth[0, :], smooth[1, :], label='Pred', color='red', linewidth=3, zorder=10)
                
                ax.set_xlim(0, 15); ax.set_ylim(0, 15)
                ax.set_title(f"Val Sample {i}")
                if i == 0: ax.legend()

            plt.suptitle(f"Epoch {epoch}", fontsize=16)
            plt.tight_layout()
            wandb.log({"Inference": wandb.Image(fig)})
            plt.close(fig)

    wandb.finish()

if __name__ == "__main__":
    train()