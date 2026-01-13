import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
import numpy as np
import matplotlib.pyplot as plt
# from model import ConditionalPathDiffusion
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

#TODO beysian serach impementation in slurm 


#TODO combine with visualizer.py from dataset gen for more accurate reconstruction
# --- Dataset Class with Augmentation ---
class PathDataset(Dataset):
    def __init__(self, path):
        print(f"Loading data from {path}...")

        
        #should be slightly higher to later allow robots wiht diffrent params such as higher acceleration or max_vel 
        path_variables = ['x','y','theta','v','accel','delta']
        path_normalization = {
            'x': (0,15),
            'y': (0,15),
            'theta': (-np.pi, np.pi),
            'v': (0,20),
            'delta': (-np.pi/2, np.pi/2),
            'accel': (-10, 10)
        }
        robot_variables = [   "wheelbase", "max_velocity","max_steering_at_zero_v","max_steering_at_max_v","acceleration","mu_static","width","length"]
        
        robot_normalization = {
        "wheelbase": (0.04,1),
        "max_velocity": (5,20),
        "max_steering_at_zero_v": (-np.pi/2, np.pi/2),
        "max_steering_at_max_v": (-np.pi/2, np.pi/2),
        "acceleration": (2, 10),
        "mu_static": (0.05, 2.5),
        "width": (0.1, 1),
        "length": (0.1, 1)
        }

        self.maps = []
        self.paths = []
        self.robots = [] 
        self.planners = [] 

        self.map_indexes = []

        #TODO remove limit to the number of maps loaded 
        for i, folder in enumerate(os.listdir(path)[:5]):
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
                    robot_array = [0] * len(robot_variables)
                    for j, var in enumerate(robot_variables):
                        if var in robot:
                            robot_array[j] = 2 * (robot[var] - robot_normalization[var][0]) / (robot_normalization[var][1] - robot_normalization[var][0]) - 1
    

                    planner = data['planner'] 

                    self.robots.append(robot_array)
                    self.planners.append(planner)
                    self.paths.append(path_tensor)
                    self.map_indexes.append(i) 


            if len(path_files) > 0:
                self.maps.append(map_tensor)

        
        # TODO implement linear or sline interpolation b-spline 
        max_path_len = max(len(p) for p in self.paths)
        for i in range(len(self.paths)):
            path = self.paths[i]
            if len(path) < max_path_len:
                last_point = path[-1]
                padding = [last_point] * (max_path_len - len(path))
                self.paths[i] = path + padding
        

        #TODO propagation step size = 0.01

        self.maps = torch.tensor(np.array(self.maps) ,dtype=torch.bool) #(N,H,W)
        #chekc -1 
        self.paths = torch.tensor(self.paths).float().permute(0,2,1)[:,:-1,:]  # [N, 128, 7] -> [N, 7, 128]
        self.paths = 2 * (self.paths - torch.tensor([ [path_normalization[var][0] for var in path_variables] ]).unsqueeze(-1)) / \
                         torch.tensor([ [path_normalization[var][1] - path_normalization[var][0] for var in path_variables] ]).unsqueeze(-1) - 1

        self.robots = torch.tensor(self.robots)

        self.robot_dim = self.robots[1]
        self.path_dim = self.paths[1]

        #TODO do wee need that?  f.e only flip verticaly ?  
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
        
    #paths = N,7,max_path_len

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        #HERE WE HAVE 3 ITEMS WERE 2    
        return self.maps[self.map_indexes[idx]], self.robots[idx],self.paths[idx]
    
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



def kinematic_loss(predicted_path, wheelbase):
    # predicted_path shape: [batch, horizon, 7]
    # Indices: x=0, y=1, v=2, theta=3, delta=4, accel=5, dt=6
    
    dt = 0.01 # Assuming constant time step; adjust if variable
    v  = predicted_path[:, :-1, 2]
    theta = predicted_path[:, :-1, 3]
    steer = predicted_path[:, :-1, 4]
    
    # Expected next states based on physics
    # x_next = x_curr + v * cos(theta) * dt
    expected_x_next = predicted_path[:, :-1, 0] + v * torch.cos(theta) * dt
    expected_y_next = predicted_path[:, :-1, 1] + v * torch.sin(theta) * dt
    
    # Heading change: d_theta = (v / L) * tan(delta) * dt
    expected_theta_next = theta + (v / wheelbase) * torch.tan(steer) * dt
    
    # Calculate error between what the model "drew" and what physics "requires"
    loss_x = torch.mean((predicted_path[:, 1:, 0] - expected_x_next)**2)
    loss_y = torch.mean((predicted_path[:, 1:, 1] - expected_y_next)**2)
    loss_theta = torch.mean((predicted_path[:, 1:, 3] - expected_theta_next)**2)
    
    return loss_x + loss_y + loss_theta


class localConfig:
    def __init__(self):
        self.epochs = 2500
        self.batch_size = 64
        self.lr = 1e-4
        self.timesteps = 1000
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset_path = "data/slurm_10_01_12-01-2026_00:07"
        # self.dataset_path = "data/strrt_vel_1"
        
        self.timesteps = 1000
        # snappy_porcupine_2026-01-02_19:13
        # "checkpoint_freq": 250,
        # "resume_path": None

# --- Training Loop ---
def train():
    # wandb.init(project="sst-path-diffusion", config=CONFIG)
    # config = wandb.config 
    config = localConfig()
    #dataset_path 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    dataset = PathDataset(config.dataset_path)


    return 
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )

    path_length = dataset.paths.shape[2]


    diff = DiffusionManager(timesteps=config.timesteps, device=device)
    model = ConditionalPathDiffusion().to(device)
    
    # if CONFIG['resume_path'] is not None:
    #     print(f"Loading weights from {CONFIG['resume_path']}...")
    #     weights = torch.load(CONFIG['resume_path'], map_location=CONFIG['device'])
    #     model.load_state_dict(weights)
    #     print("Resuming training from checkpoint!")

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    criterion = torch.nn.MSELoss()
    
    print(f"Starting Training on {CONFIG['device']}...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        for maps,robot, paths in loader:
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
        

        for batch in train_loader:
    # 1. Get data
            path_gt, map_img, robot_params = batch
            
            # 2. Diffusion Step: Add noise to the path
            t = torch.randint(0, 1000, (batch_size,))
            noise = torch.randn_like(path_gt)
            path_noisy = diffusion.add_noise(path_gt, noise, t)

            # 3. Model Prediction
            # Predicted_path is the model's attempt to reconstruct the 7D sequence
            predicted_path = model(path_noisy, t, map_img, robot_params)

            # 4. Standard Diffusion Loss (Is it close to the original?)
            loss_mse = F.mse_loss(predicted_path, path_gt)

            # 5. KINEMATIC LOSS (Does it obey the bicycle model?)
            # We use the 'wheelbase' from the robot_params (index 0 usually)
            wheelbase = robot_params[:, 0] 
            loss_physics = kinematic_loss(predicted_path, wheelbase)

            # 6. Total Loss
            # lambda is a weight (e.g., 0.1) to balance physics vs. imitation
            total_loss = loss_mse + (lambda_physics * loss_physics)

            # 7. Step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
    
        # if epoch % 100 == 0:
        #     visualize_results(model, diff, dataset, epoch)
            
        # if (epoch + 1) % CONFIG['checkpoint_freq'] == 0:
        #      filename = f"model_checkpoint_{epoch+1}.pth"
        #      torch.save(model.state_dict(), filename)
        #      print(f"Saved checkpoint: {filename}")





def visualize_results(model, diff, dataset, epoch):
    
    #ineficient but good enoough rebuild the json and pass it to visualizer 
    #how to make this diffrent robots into accuaont 
    #TODO sample should also take robot params 

    # n = 4 

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