import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
import numpy as np
import matplotlib.pyplot as plt
from model import ConditionalPathDiffusion
import os 
from PIL import Image
# --- Configuration ---
CONFIG = {
    "epochs": 2500,
    "batch_size": 64,
    "lr": 1e-4,
    "timesteps": 1000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset_path": "data/slurm_10_01_12-01-2026_00:07",
    # snappy_porcupine_2026-01-02_19:13
    "checkpoint_freq": 250,
    "resume_path": None
}

#TODO interpolation of path to same length around 5k rn 
#Is there a split for train validate 
#How to set up the values in model ?? 



#TODO combine with visualizer.py from dataset gen for more accurate reconstruction
# --- Dataset Class with Augmentation ---
class PathDataset(Dataset):
    def __init__(self, path):
        print(f"Loading data from {path}...")
        self.maps = []
        self.paths = []
        self.robot = [] 
        # self.planner = [] 

        self.map_indexes = []

        for i, folder in enumerate(os.listdir(path)):
            map_folder = os.path.join(path, folder)

            map_file = os.path.join(map_folder, 'map.png')
            if not os.path.isfile(map_file):
                continue
            

            map_tensor = np.array(Image.open(map_file).convert('1'))
            
            path_files = [f for f in os.listdir(map_folder) if f.endswith('.json')]
            for path_file in path_files:
                with open(os.path.join(map_folder, path_file), 'r') as f:
                    data = json.load(f)
                
                    path_tensor = data['path']
                    #do we need dt? last element in path 
                    robot = data['robot']
                    planner = data['planner'] 

                    self.robot.append(robot)
                    self.planner.append(planner)
                    self.paths.append(path_tensor)
                    self.map_indexes.append(i) 


            if len(path_files) > 0:
                self.maps.append(map_tensor)

        
        # TODO implement linear or sline interpolation 
        max_path_len = max(len(p) for p in self.paths)
        for i in range(len(self.paths)):
            path = self.paths[i]
            if len(path) < max_path_len:
                last_point = path[-1]
                padding = [last_point] * (max_path_len - len(path))
                self.paths[i] = path + padding


        self.maps = torch.tensor(np.array(self.maps) ,dtype=torch.bool)
        self.paths = torch.tensor(self.paths).float().permute(0,2,1)  # [N, 128, 2] -> [N, 2, 128]


        #TODO do wee need that? 
        # print("Augmenting Data (Flipping & Mirroring)...")
        # for item in data:
        #     m = torch.tensor(item['map'], dtype=torch.float32)      # [3, 64, 64]
        #     p = torch.tensor(item['path'], dtype=torch.float32)     # [128, 2]
            
        #     # 1. Original
        #     # self.maps.append(m)
        #     # self.paths.append(p.transpose(0, 1))

        #     # # 2. Flip Horizontal (Mirror width)
        #     # # Flip map width (axis 2)
        #     # m_flip = torch.flip(m, [2]) 
        #     # p_flip = p.clone()
        #     # # Invert X coordinate (since range is -1 to 1)
        #     # p_flip[:, 0] = -p_flip[:, 0] 
        #     # self.maps.append(m_flip)
        #     # self.paths.append(p_flip.transpose(0, 1))
            
        #     # # 3. Flip Vertical (Mirror height)
        #     # # Flip map height (axis 1)
        #     # m_v = torch.flip(m, [1]) 
        #     # p_v = p.clone()
        #     # # Invert Y coordinate
        #     # p_v[:, 1] = -p_v[:, 1] 
        #     # self.maps.append(m_v)
        #     # self.paths.append(p_v.transpose(0, 1))
            
        print(f"Training on {len(self.maps)} maps with {len(self.paths)} paths.")
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        #HERE WE HAVE 3 ITEMS WERE 2    
        return self.maps[idx], self.robot[idx],self.paths[self.map_indexes[idx]]
    
    # def vis(self,idx):
    #     return self.maps[idx], self.paths[self.map_indexes[idx]], self.planner[idx], self.robot[idx]
    


# --- Diffusion Utilities ---
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

    def sample(self, model, map_cond):
        model.eval()
        with torch.no_grad():
            batch_size = map_cond.shape[0]
            x = torch.randn((batch_size, 2, 128)).to(self.device)
            
            for i in reversed(range(self.timesteps)):
                t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
                predicted_noise = model(x, t, map_cond)
                
                alpha = self.alphas[i]
                alpha_cumprod = self.alphas_cumprod[i]
                beta = self.betas[i]
                
                if i > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise) + torch.sqrt(beta) * noise
                
        return x

# --- Training Loop ---
def train():
    # wandb.init(project="sst-path-diffusion", config=CONFIG)
    
    try:
        dataset = PathDataset(CONFIG['dataset_path'])
    except FileNotFoundError:
        print(f"ERROR: Could not find {CONFIG['dataset_path']}")
        return

    #set procentage for checking validation 
    loader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)

    # Setup
    diff = DiffusionManager(timesteps=CONFIG['timesteps'], device=CONFIG['device'])
    model = ConditionalPathDiffusion().to(CONFIG['device'])
    
    if CONFIG['resume_path'] is not None:
        print(f"Loading weights from {CONFIG['resume_path']}...")
        weights = torch.load(CONFIG['resume_path'], map_location=CONFIG['device'])
        model.load_state_dict(weights)
        print("Resuming training from checkpoint!")

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = torch.nn.MSELoss()
    # TODO torch.nn.RMSNorm?
    
    print(f"Starting Training on {CONFIG['device']}...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        for maps, paths in loader:
            maps, paths = maps.to(CONFIG['device']), paths.to(CONFIG['device'])
            t = torch.randint(0, CONFIG['timesteps'], (maps.size(0),), device=CONFIG['device'])
            noisy_paths, noise = diff.add_noise(paths, t)
            noise_pred = model(noisy_paths, t, maps)
            loss = criterion(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(loader)
        wandb.log({"loss": avg_loss, "epoch": epoch})
        print(f"Epoch {epoch} Loss: {avg_loss:.4f}")
        
        # if epoch % 100 == 0:
        #     visualize_results(model, diff, dataset, epoch)
            
        if (epoch + 1) % CONFIG['checkpoint_freq'] == 0:
             filename = f"model_checkpoint_{epoch+1}.pth"
             torch.save(model.state_dict(), filename)
             print(f"Saved checkpoint: {filename}")

def visualize_results(model, diff, dataset, epoch):
    
    #ineficient but good enoough rebuild the json and pass it to visualizer 
    #how to make this diffrent robots into accuaont 
    #TODO sample should also take robot params 

    idx = np.random.randint(0, len(dataset))
    map_tensor, real_path = dataset[idx]
    



    map_batch = map_tensor.unsqueeze(0).to(CONFIG['device'])
    generated_path = diff.sample(model, map_batch)
    
    gen_path = generated_path.squeeze().cpu().numpy()
    real_path = real_path.cpu().numpy()
    map_img = map_tensor.cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(map_img[0], cmap='gray_r', extent=[-1, 1, -1, 1], origin='lower')
    ax.plot(real_path[0], real_path[1], 'g--', linewidth=2, label='GT')
    ax.plot(gen_path[0], gen_path[1], 'r-', linewidth=2, label='Diffusion')
    ax.legend()
    ax.set_title(f"Epoch {epoch}")
    
    wandb.log({"generated_path": wandb.Image(fig)})
    plt.close(fig)

if __name__ == "__main__":
    train()