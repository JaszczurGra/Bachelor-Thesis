import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Block1D(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1)
        # Handle cases where dim_out is small (e.g., < 8) to avoid GroupNorm errors
        norm_groups = groups if dim_out % groups == 0 else 1 
        self.norm = nn.GroupNorm(norm_groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ConditionalUnet1D(nn.Module):
    def __init__(self, 
                 input_dim=7,      # x, y, theta, v, etc.
                 cond_dim=16,      # Dimension of physics/goal embedding
                 dim_mults=(1, 2, 4), 
                 channels=64):
        super().__init__()
        
        self.channels = channels
        
        # 1. Time Embedding
        time_dim = channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(channels),
            nn.Linear(channels, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # 2. Map Encoder (Simple CNN)
        # Assumes map is 64x64 grayscale
        self.map_encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.MaxPool2d(2), nn.ReLU(), # 32x32
            nn.Conv2d(16, 32, 3, padding=1), nn.MaxPool2d(2), nn.ReLU(), # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.MaxPool2d(2), nn.ReLU(), # 8x8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, time_dim),
            nn.ReLU()
        )

        # 3. Physics/Goal Encoder
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # 4. U-Net Backbone
        self.init_conv = nn.Conv1d(input_dim, channels, 3, padding=1)
        
        # Correct dimensions calculation
        dims = [channels, *map(lambda m: channels * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Downsample
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(nn.ModuleList([
                Block1D(dim_in, dim_in),
                Block1D(dim_in, dim_in),
                nn.Conv1d(dim_in, dim_out, 4, 2, 1) if ind < (num_resolutions - 1) else nn.Conv1d(dim_in, dim_out, 3, 1, 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = Block1D(mid_dim, mid_dim)
        self.mid_block2 = Block1D(mid_dim, mid_dim)

        # Upsample
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.ups.append(nn.ModuleList([
                # --- FIX IS HERE: dim_out + dim_in instead of dim_out * 2 ---
                Block1D(dim_out + dim_in, dim_in), 
                Block1D(dim_in, dim_in),
                nn.ConvTranspose1d(dim_in, dim_in, 4, 2, 1) if ind < (num_resolutions - 1) else nn.Conv1d(dim_in, dim_in, 3, 1, 1)
            ]))

        self.final_res_block = Block1D(channels, channels)
        self.final_conv = nn.Conv1d(channels, input_dim, 1)

    def forward(self, x, time, map_img, cond_vec):
        # Embeddings
        t = self.time_mlp(time)
        m = self.map_encoder(map_img)
        c = self.cond_mlp(cond_vec)
        
        # Combine global context (simple summation)
        global_cond = t + m + c 
        
        h = self.init_conv(x)
        r = h.clone()

        skips = []
        
        # Down
        for block1, block2, downsample in self.downs:
            h = block1(h)
            h = block2(h)
            skips.append(h)
            h = downsample(h)

        # Middle
        h = self.mid_block1(h)
        h = self.mid_block2(h)

        # Up
        for block1, block2, upsample in self.ups:
            # Concatenate skip connection
            skip = skips.pop()
            h = torch.cat((h, skip), dim=1)
            
            h = block1(h)
            h = block2(h)
            h = upsample(h)

        h = h + r
        h = self.final_res_block(h)
        return self.final_conv(h)