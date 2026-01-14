import json
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
import numpy as np
import matplotlib.pyplot as plt
from model import DiffusionDenoiser
import os 
from PIL import Image
from scipy.interpolate import CubicSpline
# --- Configuration ---
# CONFIG = {
#     "epochs": 2500,
#     "batch_size": 64,
#     "lr": 1e-4,
#     "timesteps": 1000,
#     "device": "cuda" if torch.cuda.is_available() else "cpu",
#     "dataset_path": "data/slurm_10_01_12-01-2026_00:07",
#     # snappy_porcupine_2026-01-02_19:13
#     "checkpoint_freq": 250,
#     "resume_path": None
# }

#TODO interpolation of path to same length around 5k rn 
#Is there a split for train validate 
#How to set up the values in model ?? 
 

 #TODO robot always starts in the same positions we should train it on multiple start,goal positions, shouldnt it be included somewhere in params nontheles???

#TODO combine with visualizer.py from dataset gen for more accurate reconstruction
# --- Dataset Class with Augmentation ---
class PathDataset(Dataset):
    def __init__(self, path, n_maps):
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
        #TODO 
        #TODO 
        #TODO 
        i = 0 
        for folder in os.listdir(path)[:n_maps]:
            map_folder = os.path.join(path, folder)

            map_file = os.path.join(map_folder, 'map.png')
            if not os.path.isfile(map_file):
                continue
            

            map_tensor = np.array(Image.open(map_file).convert('1'))[::-1,:]  # Invert Y-axis to match coordinate system
            
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
                i+=1 

        
 


        #TODO propagation step size = 0.01 is constant right now probably not a problem no need for nomalization
        #TODO move tensors to gpu for this operation and then back to cpu to not take space 
        #TODO can this acieve full speed on bools? > maps can't be bool as the convolution needs float 
        self.maps = torch.tensor(np.array(self.maps) ,dtype=torch.float32).unsqueeze(1) #(N,H,W)
        #chekc -1 
        self.paths = self.resample_path(self.paths,max(len(p) for p in self.paths) ,path_variables, dt=0.01)
        self.dts = [p[-1][-1] for p in self.paths]  
        # self.paths = torch.tensor(self.paths).float().permute(0,2,1)  # [N, 128, 7] -> [N, 7, 128]
        self.paths = torch.tensor(self.paths).float().permute(0,2,1)[:,:-1,:] 
        self.paths = 2 * (self.paths - torch.tensor([ [path_normalization[var][0] for var in path_variables] ]).unsqueeze(-1)) / \
                         torch.tensor([ [path_normalization[var][1] - path_normalization[var][0] for var in path_variables] ]).unsqueeze(-1) - 1
        #TODO do robot normalization here instead of in the loop for faster loading 
        self.robots = torch.tensor(self.robots)

        self.robot_dim = self.robots.shape[1]
        self.path_dim = self.paths.shape[1]

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
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        return self.maps[self.map_indexes[idx]], self.robots[idx],self.paths[idx]


    def resample_path(self, paths, target_len, path_variables, dt=0.01):
        # TODO implement linear or sline interpolation b-spline 
        # max_path_len = max(len(p) for p in self.paths)
        # for i in range(len(self.paths)):
        #     path = self.paths[i]
        #     if len(path) < max_path_len:
        #         last_point = path[-1]
        #         padding = [last_point] * (max_path_len - len(path))
        #         self.paths[i] = path + padding
        

        
        resampled_paths = []
        for path in self.paths:
            path_np = np.array(path)
            current_len = len(path_np)

            if current_len <= 3: # Cubic spline needs at least 4 points for good results
                # Fallback to padding for very short paths
                last_point = path_np[-1] if current_len > 0 else np.zeros(path_np.shape[1])
                padded_path = np.tile(last_point, (target_len, 1))
                padded_path[:, -1] = 0 # Set dt to 0 for padded paths
                resampled_paths.append(padded_path.tolist())
                continue

            total_duration = (current_len - 1) * dt
            original_time_points = np.linspace(0, total_duration, current_len)

            new_time_points = np.linspace(0, total_duration, target_len)
            new_dt = new_time_points[1] - new_time_points[0] if target_len > 1 else 0

            resampled_path_full = np.zeros((target_len, path_np.shape[1]))

            for i in range(path_np.shape[1] - 1): # Loop through x, y, theta, v, accel, delta
                #   path_variables = ['x','y','theta','v','accel','delta']
                if path_variables[i] == 'theta' or path_variables[i] == 'delta':
                    unwrapped_theta = np.unwrap(path_np[:, i])
                    spline = CubicSpline(original_time_points, unwrapped_theta)
                    interpolated_theta = spline(new_time_points)
                    resampled_path_full[:, i] = np.mod(interpolated_theta + np.pi, 2 * np.pi) - np.pi
                else:
                    spline = CubicSpline(original_time_points, path_np[:, i])
                    resampled_path_full[:, i] = spline(new_time_points)
            
            resampled_path_full[:, -1] = new_dt
            
            resampled_paths.append(resampled_path_full.tolist())
        return resampled_paths

    # def vis(self,idx):
    #     return self.maps[idx], self.paths[self.map_indexes[idx]], self.planner[idx], self.robot[idx]

class DiffusionManager:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    # is this necessary or just onliner in train loop 
    def add_noise(self, x_start, t):
        noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None]
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise, noise

    #TODO what it does ? 
    def sample(self, model, map_cond,robot_params, real_path):
        model.eval()
        with torch.no_grad():
            batch_size = map_cond.shape[0]
            x = torch.randn((batch_size, real_path.shape[1], real_path.shape[2]), device=self.device)
            
            for i in reversed(range(self.timesteps)):
                t = torch.full((batch_size,), i, device=self.device, dtype=torch.long)
                
                # Predict noise
                predicted_noise = model(x, t, map_cond, robot_params)
                
                # Get scheduling parameters
                alpha_t = self.alphas[i]
                alpha_cumprod_t = self.alphas_cumprod[i]
                beta_t = self.betas[i]
                
                # Compute coefficients for DDPM sampling
                # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_cumprod_t)) * noise) + sqrt(beta_t) * z
                
                # Coefficient for x_t
                coef1 = 1 / torch.sqrt(alpha_t)
                
                # Coefficient for predicted noise
                coef2 = beta_t / torch.sqrt(1 - alpha_cumprod_t)
                
                # Mean of the reverse distribution
                mean = coef1 * (x - coef2 * predicted_noise)
                
                # Add noise (except at last step)
                if i > 0:
                    noise = torch.randn_like(x)
                    sigma = torch.sqrt(beta_t)
                    x = mean + sigma * noise
                else:
                    x = mean
                
                # CRITICAL: Clamp to prevent explosion
                # Paths should stay in normalized range [-1, 1]
                x = torch.clamp(x, -3, 3)  # Allow some wiggle room, then clamp hard
            
            # Final clamp to ensure output is in valid range
            x = torch.clamp(x, -1, 1)
            
        return x



            # result[2] = (state[3] / self.robot.wheelbase) * math.tan(np.clip(control[1], -MAX_DELTA, MAX_DELTA)) 
def kinematic_loss(predicted_path, robot_params, dt=0.01):
    """
    Compute physics-based kinematic loss for predicted paths.
    
    Args:
        predicted_path: [B, state_dim, horizon] - Model output
                        Channels: [x, y, theta, v, accel, delta]
        robot_params: [B, robot_param_dim] - Robot parameters
                      Indices: [wheelbase, max_velocity, max_steering_at_zero_v, 
                               max_steering_at_max_v, acceleration, mu_static, width, length]
        dt: float - Time step (default 0.01)
    
    Returns:
        loss: scalar - Physics violation penalty
    """
    # Extract robot parameters (denormalize from [-1, 1])
    robot_normalization = {
        "wheelbase": (0.04, 1),
        "max_velocity": (5, 20),
        "max_steering_at_zero_v": (-np.pi/2, np.pi/2),
        "max_steering_at_max_v": (-np.pi/2, np.pi/2),
        "acceleration": (2, 10),
        "mu_static": (0.05, 2.5),
        "width": (0.1, 1),
        "length": (0.1, 1)
    }
    
    # Denormalize robot params
    wheelbase = (robot_params[:, 0] + 1) / 2 * (1 - 0.04) + 0.04  # [B]
    mu_static = (robot_params[:, 5] + 1) / 2 * (2.5 - 0.05) + 0.05  # [B]
    max_steer_zero_v = (robot_params[:, 2] + 1) / 2 * (np.pi) - np.pi/2  # [B]
    
    # Add dimensions for broadcasting: [B, 1]
    wheelbase = wheelbase.unsqueeze(1)
    mu_static = mu_static.unsqueeze(1)
    max_steer_zero_v = max_steer_zero_v.unsqueeze(1)
    
    # Extract state variables [B, horizon]
    # Indices: x=0, y=1, theta=2, v=3, accel=4, delta=5
    x = predicted_path[:, 0, :-1]      # [B, horizon-1]
    y = predicted_path[:, 1, :-1]
    theta = predicted_path[:, 2, :-1]
    v = predicted_path[:, 3, :-1]
    accel = predicted_path[:, 4, :-1]
    delta = predicted_path[:, 5, :-1]
    
    # Compute effective steering angle with lateral force limit
    # At high speeds, lateral force constraint limits turning radius
    lateral_force_min_v = 0.1  # Minimum velocity for lateral force calculation
    
    # Compute angle with lateral force constraint
    # F_lateral = m * v^2 / r < mu * m * g
    # r = L / tan(delta) => tan(delta) = L / r
    # r_min = v^2 / (mu * g)
    # delta_max = arctan(L / r_min) = arctan(L * mu * g / v^2)
    
    g = 9.81
    # Avoid division by zero at low speeds
    v_safe = torch.clamp(v, min=lateral_force_min_v)
    
    # Maximum steering angle from lateral force constraint
    delta_max_lateral = torch.atan(wheelbase * mu_static * g / (v_safe ** 2))  # [B, horizon-1]
    
    # Apply constraint: use lateral force limit at high speed, mechanical limit at low speed
    # Use smooth transition based on velocity
    use_lateral_limit = (v >= lateral_force_min_v).float()
    
    # Constrain steering angle
    delta_constrained = torch.where(
        v >= lateral_force_min_v,
        torch.clamp(delta, -delta_max_lateral, delta_max_lateral),  # High speed: lateral force limit
        torch.clamp(delta, -max_steer_zero_v, max_steer_zero_v)     # Low speed: mechanical limit
    )
    
    # Compute effective angle for dynamics
    angle = delta_constrained  # Could also use tan approximation for small angles
    
    # Expected next states based on bicycle model with constraints
    # dx/dt = v * cos(theta)
    # dy/dt = v * sin(theta)
    # dtheta/dt = (v / L) * tan(delta)  [or simplified: (v / L) * angle]
    # dv/dt = accel
    
    expected_x_next = x + v * torch.cos(theta) * dt
    expected_y_next = y + v * torch.sin(theta) * dt
    expected_theta_next = theta + (v / wheelbase) * torch.tan(angle) * dt
    expected_v_next = v + accel * dt
    
    # Get actual next states from predicted path
    x_next = predicted_path[:, 0, 1:]      # [B, horizon-1]
    y_next = predicted_path[:, 1, 1:]
    theta_next = predicted_path[:, 2, 1:]
    v_next = predicted_path[:, 3, 1:]
    
    # Compute violations (L2 loss)
    loss_x = torch.mean((x_next - expected_x_next) ** 2)
    loss_y = torch.mean((y_next - expected_y_next) ** 2)
    loss_theta = torch.mean((theta_next - expected_theta_next) ** 2)
    loss_v = torch.mean((v_next - expected_v_next) ** 2)
    
    # Optional: Add penalty for violating steering constraints
    # This encourages the model to stay within physical limits
    steering_violation = torch.mean(torch.relu(torch.abs(delta) - delta_max_lateral) ** 2)
    
    # Total kinematic loss
    total_loss = loss_x + loss_y + loss_theta + loss_v + 0.1 * steering_violation
    
    return total_loss


local_config = {
    # batch size = 64 - approximately 5GB vram
    "epochs": 2500,
    "batch_size": 64,
    "lr": 1e-4,
    "timesteps": 1000,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dataset_path": "data/slurm_10_01_12-01-2026_00:07",
    "checkpoint_freq": 250,
    'visualization_freq': 50,
    "resume_path": None,
    'n_maps': 100,
    'beta_start': 1e-4,
    'beta_end': 0.02
}

def train():

    wandb.init( project="Motion planning", config=local_config)
    config = wandb.config 

    device = config.device


    dataset = PathDataset(config.dataset_path,config.n_maps)


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




    diff = DiffusionManager(timesteps=config.timesteps,beta_start=config.beta_start,beta_end=config.beta_end, device=device)

    #TODO only works for squere maps 
    model = DiffusionDenoiser(state_dim=dataset.path_dim,robot_param_dim=dataset.robot_dim,map_size=dataset.maps.shape[2]).to(device) 
    
    # if CONFIG['resume_path'] is not None:
    #     print(f"Loading weights from {CONFIG['resume_path']}...")
    #     weights = torch.load(CONFIG['resume_path'], map_location=CONFIG['device'])
    #     model.load_state_dict(weights)
    #     print("Resuming training from checkpoint!")

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
  
    criterion = torch.nn.MSELoss()
    
    print(f"Starting Training on {device}...")
    
    for epoch in range(config.epochs):
        model.train()
        train_loss = 0
        for map,robot, path in train_loader:
            map, robot, path = map.to(device), robot.to(device), path.to(device)
            t = torch.randint(0, config.timesteps, (map.size(0),), device=device)
            noisy_paths, noise = diff.add_noise(path, t)
            noise_pred = model(noisy_paths, t, map, robot)
            loss = criterion(noise_pred, noise)
            #loss = loss + (0.1 *kinematic_loss(predicted_path, wheelbase))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)


        #TODO stop training when val loss doesnt increase over the best after N epchos 
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for map, robot, path in val_loader:
                map, robot, path = map.to(device), robot.to(device), path.to(device)
                t = torch.randint(0, config.timesteps, (map.size(0),), device=device)
                noisy_paths, noise = diff.add_noise(path, t)
                noise_pred = model(noisy_paths, t, map, robot)
                loss = criterion(noise_pred, noise)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        #TODO is the optimizer needed if so whcih one ? 
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        wandb.log({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "epoch": epoch})
        

        if epoch % config.visualization_freq == 0:
            visualize_results(model, diff, val_dataset, epoch,device)
            
        if (epoch + 1) % config.checkpoint_freq == 0:
             run_name = wandb.run.name or f"run-{wandb.run.id}"
             model_dir = os.path.join("models", run_name)
             os.makedirs(model_dir, exist_ok=True)
             filename = os.path.join(model_dir, f"model_checkpoint_{epoch+1}.pth")
             torch.save(model.state_dict(), filename)
             print(f"Saved checkpoint: {filename}")

def simulate_path(path,robot_params, dt=0.01):
    #TODO implement , needs te normalization of the parameters 
    import math 
    
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
    robot_params = {key: val for key, val in zip(robot_normalization.keys(), robot_params)}
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

    return path




def visualize_results(model, diff, dataset, epoch,device):
    n = 4 
    idxs = np.random.choice(len(dataset), size=min(n, len(dataset)), replace=False).astype(int)
    samples = [dataset[i] for i in idxs]
    map_tensor, robot, real_path = [torch.stack(tensors) for tensors in zip(*samples)]
    # map_tensor, robot, real_path = dataset[idx]
    map_tensor, robot, real_path = map_tensor.to(device), robot.to(device),real_path.to(device)
    
    
    #TODO add simulated path from control using propaget function 


    generated_path = diff.sample(model, map_tensor,robot, real_path).squeeze(0) 


    gen_path = generated_path.cpu().numpy()
    real_path = real_path.cpu().numpy()
    map_img = map_tensor.cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i in range(n):
        ax = axes[i]
        ax.imshow(map_img[i,0], cmap='gray', extent=[-1, 1, -1, 1], origin='lower')
        ax.plot(real_path[i,0,:], real_path[i,1,:], 'g--', linewidth=2, label='GT')

        #TODO add plotting of the robot at start position of gt and generated path 
        # robot_x, robot_y = real_path[i, 0, 0], real_path[i, 1, 0]
        # robot_theta = real_path[i, 2, 0]
        # robot_width = (robot[i, 6].item() + 1) / 2 * (1 - 0.1) + 0.1
        # robot_length = (robot[i, 7].item() + 1) / 2 * (1 - 0.1) + 0.1

        # corners = np.array([
        #     [-robot_length/2, -robot_width/2],
        #     [robot_length/2, -robot_width/2],
        #     [robot_length/2, robot_width/2],
        #     [-robot_length/2, robot_width/2],
        #     [-robot_length/2, -robot_width/2]
        # ])

        # cos_theta = np.cos(robot_theta)
        # sin_theta = np.sin(robot_theta)
        # rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        # rotated_corners = corners @ rotation_matrix.T
        # rotated_corners[:, 0] += robot_x
        # rotated_corners[:, 1] += robot_y

        # ax.plot(rotated_corners[:, 0], rotated_corners[:, 1], 'b-', linewidth=2, label='Robot')

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
    train()