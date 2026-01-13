import torch
import torch.nn as nn
import math


class ResnetBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim):
        super().__init__()
        
        # 1. Main Convolutional Path
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.norm1 = nn.GroupNorm(8, out_channels)
        
        # 2. FiLM Conditioning (Scale and Shift)
        # We predict both a Multiplier (gamma) and an Offset (beta)
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, out_channels * 2) 
        )
        
        # 3. Second Convolutional Path
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        self.act = nn.SiLU()
        
        # 4. Residual matching
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        x: [Batch, Channels, Horizon] (The noisy 7D path)
        cond: [Batch, Cond_Dim] (The fused 10 robot params + Map + Time)
        """
        # First layer
        h = self.conv1(x)
        h = self.norm1(h)
        
        # FiLM injection: scale and shift the features
        # This is where 'max_velocity' or 'wheelbase' shapes the trajectory features
        cond_features = self.cond_mlp(cond).unsqueeze(-1) # [B, 2*out_channels, 1]
        gamma, beta = torch.chunk(cond_features, 2, dim=1) # Split into two halves
        h = h * (1 + gamma) + beta # Apply scale and shift
        
        # Second layer
        h = self.act(h)
        h = self.conv2(h)
        h = self.norm2(h)
        
        return self.act(h + self.residual_conv(x))



class DiffusionDenoiser(nn.Module):

    #TODO proper sizes for robot and state dim either remove unimportant or set the size for all of them  
    def __init__(self, state_dim=6, robot_param_dim=8, map_size=500):
        super().__init__()

        #TODO should be settable 
        #map_feature_dim 
        #128 nr of nodes per layer 
        # 4 layers (2 internal layers)
        self.map_feat_dim = 256
        self.robot_feat_dim = 128
        self.time_feat_dim = 64
        # 1. Map Encoder
        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # Output: 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # Output: 16x16
            nn.Flatten(),
            nn.Linear(32 * (map_size//4) * (map_size//4), self.map_feat_dim)
        )
        
        # 2. Robot Param Encoder


        self.param_mlp = nn.Sequential(
            nn.Linear(robot_param_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.robot_feat_dim)
        )

        # 3. Time Embedding (Standard for Diffusion)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, self.time_feat_dim)
        )

        # Combined Condition Dim
        total_cond_dim = self.map_feat_dim + self.robot_feat_dim + self.time_feat_dim
        
        # 4. Denoiser Layers
        self.input_conv = nn.Conv1d(state_dim, 128, 1)
        self.res_block1 = ResnetBlock1D(128, 128, total_cond_dim)
        self.res_block2 = ResnetBlock1D(128, 128, total_cond_dim)
        self.final_conv = nn.Conv1d(128, state_dim, 1)

    def forward(self, x_noisy, t, map_img, robot_params):
        # x_noisy: [B, Horizon, 6] -> [B, 6, Horizon]
        x = x_noisy.transpose(1, 2)
        
        # Encode conditions
        m_feat = self.map_cnn(map_img)
        p_feat = self.param_mlp(robot_params)
        t_feat = self.time_mlp(t.unsqueeze(-1).float())
        
        # Fuse features (Summing is efficient, Concat is more expressive)
        combined_cond = m_feat + p_feat + t_feat
        
        # Denoise
        x = self.input_conv(x)
        x = self.res_block1(x, combined_cond)
        x = self.res_block2(x, combined_cond)
        out = self.final_conv(x)
        
        return out.transpose(1, 2) # Back to [B, Horizon, 6]
    

