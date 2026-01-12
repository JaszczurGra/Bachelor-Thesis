import torch
import matplotlib.pyplot as plt
import numpy as np
import random

# Configuration
DATASET_FILE = "old_diffusion_test/100k_iters-first_found-5k_samples.pt"
OUTPUT_IMAGE = "dataset_preview.png"
SAMPLES_TO_SHOW = 16

def inspect():
    print(f"Loading {DATASET_FILE}...")
    data = torch.load(DATASET_FILE, weights_only=False)
    print(f"Dataset contains {len(data)} trajectories.")

    indices = random.sample(range(len(data)), SAMPLES_TO_SHOW)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f"Random Samples from {DATASET_FILE}", fontsize=16)

    for i, idx in enumerate(indices):
        ax = axes.flat[i]
        sample = data[idx]
        
        obs_map = sample['map'][0] 
        
        path = sample['path']
        
        ax.imshow(obs_map, cmap='gray_r', extent=[-1, 1, -1, 1], origin='lower')

        ax.plot(path[:, 0], path[:, 1], color='red', linewidth=2, label='Path')
        
        ax.scatter(path[0, 0], path[0, 1], color='green', s=50, zorder=5) # Start
        ax.scatter(path[-1, 0], path[-1, 1], color='blue', s=50, zorder=5) # Goal

        ax.set_title(f"Sample {idx}")
        ax.axis('off')

    plt.tight_layout()
    print(f"Saving visualization to {OUTPUT_IMAGE}...")
    plt.savefig(OUTPUT_IMAGE)
    print("Done! Open the image to verify your data.")

if __name__ == "__main__":
    inspect()