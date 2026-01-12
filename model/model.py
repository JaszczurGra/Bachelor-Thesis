import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block1D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.norm = nn.BatchNorm1d(out_ch)

    def forward(self, x):
        return self.norm(self.act(self.conv2(self.act(self.conv1(x)))))

class ConditionalPathDiffusion(nn.Module):
    def __init__(self, path_len=128):
        super().__init__()
        
        # 1. Vision Encoder (processes the 64x64 map)
        self.map_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # -> 32x32
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # -> 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # -> 8x8
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU()
        )

        # 2. Time Embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(64),
            nn.Linear(64, 256),
            nn.ReLU()
        )

        # 3. 1D U-Net for Path
        # Downsample
        self.down1 = Block1D(2, 64)  # Input is (x,y) coordinates
        self.down2 = Block1D(64, 128)
        self.down3 = Block1D(128, 256)
        
        self.pool = nn.MaxPool1d(2)

        # Bottleneck (Conditioning is injected here)
        self.bottleneck = Block1D(256 + 256 + 256, 512) # + map_emb + time_emb

        # Upsample
        self.up1 = nn.ConvTranspose1d(512, 256, 2, stride=2)
        self.up_conv1 = Block1D(512, 256) # 512 due to skip connection
        
        self.up2 = nn.ConvTranspose1d(256, 128, 2, stride=2)
        self.up_conv2 = Block1D(256, 128)
        
        self.up3 = nn.ConvTranspose1d(128, 64, 2, stride=2)
        self.up_conv3 = Block1D(128, 64)
        
        self.final = nn.Conv1d(64, 2, 1)

    def forward(self, x, t, map_img):
        # x: [Batch, 2, 128] (The path)
        # t: [Batch] (Timesteps)
        # map_img: [Batch, 3, 64, 64] (Occupancy grid)

        # Encode condition and time
        map_emb = self.map_encoder(map_img) # [B, 256]
        time_emb = self.time_mlp(t)         # [B, 256]
        
        # Expand embeddings to concatenate with 1D features
        # Assuming bottleneck size is path_len / 8 = 16
        map_emb = map_emb.unsqueeze(-1).repeat(1, 1, 16) 
        time_emb = time_emb.unsqueeze(-1).repeat(1, 1, 16)

        # U-Net Down
        d1 = self.down1(x)        # [B, 64, 128]
        d2 = self.down2(self.pool(d1)) # [B, 128, 64]
        d3 = self.down3(self.pool(d2)) # [B, 256, 32]
        b_in = self.pool(d3)           # [B, 256, 16]

        # Concatenate Conditioning at Bottleneck
        # We combine Path Features + Map Features + Time Features
        cat_emb = torch.cat([b_in, map_emb, time_emb], dim=1) 
        neck = self.bottleneck(cat_emb) # [B, 512, 16]

        # U-Net Up
        u1 = self.up1(neck)               # [B, 256, 32]
        u1 = self.up_conv1(torch.cat([u1, d3], dim=1))
        
        u2 = self.up2(u1)                 # [B, 128, 64]
        u2 = self.up_conv2(torch.cat([u2, d2], dim=1))
        
        u3 = self.up3(u2)                 # [B, 64, 128]
        u3 = self.up_conv3(torch.cat([u3, d1], dim=1))
        
        return self.final(u3) # [B, 2, 128] predicted noise