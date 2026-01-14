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
    def __init__(self, state_dim=6, robot_param_dim=8, map_size=500, map_feat_dim=256, robot_feat_dim=128, time_feat_dim=64, num_internal_layers=4, base_layer_dim=128):
        super().__init__()

        #TODO should be settable 
        #map_feature_dim ,robot_feat_dim ,robot_feat_dim
        #128 nr of nodes per layer 
        # 4 layers (2 internal layers)
        # 1. Map Encoder
        self.map_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1), # Output: 32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # Output: 16x16
            nn.Flatten(),
            nn.Linear(32 * (map_size//4) * (map_size//4), map_feat_dim)
        )
        
        # 2. Robot Param Encoder


        self.param_mlp = nn.Sequential(
            nn.Linear(robot_param_dim, 64),
            nn.ReLU(),
            nn.Linear(64, robot_feat_dim)
        )

        # 3. Time Embedding (Standard for Diffusion)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, time_feat_dim)
        )

        # Combined Condition Dim
        total_cond_dim = map_feat_dim + robot_feat_dim + time_feat_dim
        
        # 4. Denoiser Layers
        # self.input_conv = nn.Conv1d(state_dim, 128, 1)
        # self.res_block1 = ResnetBlock1D(128, 128, total_cond_dim)
        # self.res_block2 = ResnetBlock1D(128, 128, total_cond_dim)
        # self.final_conv = nn.Conv1d(128, state_dim, 1)


        #U-net approach:
        self.input_conv = nn.Conv1d(state_dim, 128, 1)

        # --- Encoder ---
        self.res_block_enc1 = ResnetBlock1D(128, 128, total_cond_dim)
        self.res_block_enc2 = ResnetBlock1D(128, 256, total_cond_dim) # Increase channels

        # --- Bottleneck ---
        self.bottleneck = ResnetBlock1D(256, 256, total_cond_dim)

        # --- Decoder ---
        self.res_block_dec1 = ResnetBlock1D(512, 128, total_cond_dim) # 256+256 from concatenation
        self.res_block_dec2 = ResnetBlock1D(256, 128, total_cond_dim) # 128+128 from concatenation
        
        self.final_conv = nn.Conv1d(128, state_dim, 1)

    def forward(self, x_noisy, t, map_img, robot_params):
        x = x_noisy
        
        m_feat = self.map_cnn(map_img)
        p_feat = self.param_mlp(robot_params)
        t_feat = self.time_mlp(t.unsqueeze(-1).float())
        
        combined_cond = torch.cat([m_feat, p_feat, t_feat], dim=1)
        # x = self.input_conv(x)
        # x = self.res_block1(x, combined_cond)
        # x = self.res_block2(x, combined_cond)
        # out = self.final_conv(x)
        
    
        x = self.input_conv(x_noisy)

    # Encoder
        x1 = self.res_block_enc1(x, combined_cond)
        x2 = self.res_block_enc2(x1, combined_cond) # e.g., 128 -> 256 channels

        # Bottleneck
        b = self.bottleneck(x2, combined_cond)

        # Decoder with Skip Connections
        # The input to dec1 is the bottleneck output + the output from enc2
        d1_in = torch.cat([b, x2], dim=1) 
        d1 = self.res_block_dec1(d1_in, combined_cond) # Note: dec1 in_channels must be 256+256

        # The input to dec2 is d1 + the output from enc1
        d2_in = torch.cat([d1, x1], dim=1)
        d2 = self.res_block_dec2(d2_in, combined_cond) # Note: dec2 in_channels must be 128+128

        out = self.final_conv(d2)
    

        return out 
