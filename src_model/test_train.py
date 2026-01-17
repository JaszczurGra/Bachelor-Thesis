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

# Import your model
from test_model import ConditionalUnet1D

# --- Configuration ---
CONFIG = {
    "data_dir": "slurm_data/test_dataset",  # Root directory containing map_0, map_1, etc.
    "seq_len": 128,
    "map_size": 64,       # Training resolution (efficient)
    "batch_size": 32,     # Increased batch size is safe with small maps
    "epochs": 5000,
    "lr": 1e-4,
    "viz_interval": 100,   # Visualize often to check generalization
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

# --- 2. Dataset (Recursive Multi-Map) ---
class RobotPathDataset(Dataset):
    def __init__(self, root_dir, map_size=64, seq_len=128):
        self.seq_len = seq_len
        self.map_size = map_size
        self.samples = [] # List of tuples: (json_path, map_path)

        # 1. Walk through root directory looking for map folders
        if not os.path.exists(root_dir):
            raise ValueError(f"Data directory {root_dir} does not exist")

        print(f"Scanning {root_dir} for maps and paths...")
        
        # Iterate over map_0, map_1, etc.
        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        for d in subdirs:
            folder_path = os.path.join(root_dir, d)
            map_file = os.path.join(folder_path, "map.png")
            
            # Skip if no map
            if not os.path.exists(map_file):
                continue
                
            # Find all json paths in this folder
            json_files = sorted(glob.glob(os.path.join(folder_path, "*.json")))
            
            # Add valid pairs to our list
            for jf in json_files:
                self.samples.append((jf, map_file))

        if len(self.samples) == 0:
            raise ValueError(f"No valid pairs of (map.png, *.json) found in {root_dir}")
            
        print(f"Found {len(self.samples)} trajectories across {len(subdirs)} map folders.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        json_path, map_path = self.samples[idx]
        
        # A. Load Map (On the fly to save RAM)
        img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        if img is None: raise ValueError(f"Failed to decode {map_path}")
        img = cv2.resize(img, (self.map_size, self.map_size))
        map_tensor = torch.FloatTensor(img).unsqueeze(0) / 255.0 

        # B. Load Path & Physics
        with open(json_path, 'r') as f:
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

        return path_tensor, map_tensor, robot_vec

    # Helper to get raw file paths for visualization
    def get_paths(self, idx):
        return self.samples[idx]

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
            # Iterate silently for visualization
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
    wandb.init(project="Motion Planning", config=CONFIG)
    
    # 1. Dataset
    try:
        full_dataset = RobotPathDataset(CONFIG["data_dir"], map_size=CONFIG["map_size"], seq_len=CONFIG["seq_len"])
    except Exception as e:
        print(f"Error: {e}")
        return

    # Train/Val Split
    train_size = int(0.9 * len(full_dataset)) # 90/10 split
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    
    print(f"Train samples: {train_size}, Val samples: {val_size}")
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True)
    # Drop last to avoid batch-norm errors on tiny leftovers
    val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, drop_last=False)
    
    # 2. Setup
    model = ConditionalUnet1D(input_dim=7, cond_dim=6, channels=64).to(CONFIG["device"])
    scheduler = DDPMScheduler(num_timesteps=1000)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    loss_fn = nn.MSELoss()
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)

    # --- Pre-Select Visualization Batch ---
    # We select 4 fixed validation samples.
    # CRITICAL: We must retrieve the High-Res maps for these specific samples.
    print("Preparing visualization batch...")
    num_viz = min(4, val_size)
    vis_indices = torch.linspace(0, val_size-1, steps=num_viz).long()
    
    vis_data_store = [] # Stores (MapTensor, CondTensor, GT_Numpy, HighResMap_Numpy)

    for idx in vis_indices:
        # Get Tensor Data
        path_T, map_T, cond_T = val_ds[idx]
        
        # Get Original Index to find File Path
        original_idx = val_ds.indices[idx]
        _, map_path_str = full_dataset.get_paths(original_idx)
        
        # Load High Res Map
        high_res = cv2.imread(map_path_str, cv2.IMREAD_GRAYSCALE)
        high_res = np.flipud(high_res) # Plotting convention
        
        vis_data_store.append({
            "map_T": map_T,
            "cond_T": cond_T,
            "gt_np": (path_T.numpy() + 1) / 2 * 15.0, # Denormalize GT
            "high_res": high_res
        })

    # 3. Training Loop
    print(f"Starting Training on {CONFIG['device']}...")
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
        
        avg_train_loss = epoch_train_loss / len(train_loader)

        # Validation (Every epoch for smooth graphs)
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
            
            # Construct Batch
            batch_maps = torch.stack([d["map_T"] for d in vis_data_store]).to(CONFIG["device"])
            batch_conds = torch.stack([d["cond_T"] for d in vis_data_store]).to(CONFIG["device"])
            
            # Inference
            gen_batch = scheduler.sample(model, batch_maps, batch_conds, (len(vis_data_store), 7, CONFIG["seq_len"]))
            
            # Plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            axes = axes.flatten()
            
            for i, data in enumerate(vis_data_store):
                ax = axes[i]
                
                # Get generated path
                raw = gen_batch[i].squeeze().cpu().numpy()
                raw = (raw + 1) / 2 * 15.0
                smooth = smooth_path(raw)
                
                # Plot
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