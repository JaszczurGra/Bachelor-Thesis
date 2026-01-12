import torch
import numpy as np
import multiprocessing
from main import SSTPlanner, Obstacle
import time

# Configuration
NUM_SAMPLES_NEEDED = 5000
MAP_SIZE = 64
ENV_SIZE = 50.0
FIXED_PATH_LEN = 128

# --- Helper Functions (Same as before) ---
def rasterize_map(obstacles, start, goal, size=MAP_SIZE, env_limit=ENV_SIZE):
    grid = np.zeros((3, size, size), dtype=np.float32)
    scale = size / env_limit
    for obs in obstacles:
        x1, y1 = int(obs.x * scale), int(obs.y * scale)
        w, h = int(obs.w * scale), int(obs.h * scale)
        grid[0, y1:y1+h, x1:x1+w] = 1.0
    for ch, point in [(1, start), (2, goal)]:
        px, py = int(point[0]*scale), int(point[1]*scale)
        y_min, y_max = max(0, py-1), min(size, py+2)
        x_min, x_max = max(0, px-1), min(size, px+2)
        grid[ch, y_min:y_max, x_min:x_max] = 1.0
    return grid

def interpolate_path(path, target_len=FIXED_PATH_LEN):
    path = np.array(path)[:, :2]
    if len(path) < 2: return None
    dists = np.linalg.norm(path[1:] - path[:-1], axis=1)
    cum_dist = np.insert(np.cumsum(dists), 0, 0)
    if cum_dist[-1] == 0: return None
    new_dists = np.linspace(0, cum_dist[-1], target_len)
    new_x = np.interp(new_dists, cum_dist, path[:, 0])
    new_y = np.interp(new_dists, cum_dist, path[:, 1])
    return np.stack([new_x, new_y], axis=1)

# --- Worker Function ---
def generate_single_sample(seed):
    """Generates one sample. Returns dict or None."""
    # Re-seed numpy for this process
    np.random.seed(seed)
    
    start_state = [5.0, 5.0, 1.57]
    goal_state = [45.0, 45.0, 0.0]
    
    # Create Obstacles
    obstacles = [
        Obstacle(np.random.uniform(10,40), np.random.uniform(10,40), 
                 np.random.uniform(3,7), np.random.uniform(3,7)) 
        for _ in range(12) 
    ]
    
    planner = SSTPlanner(start_state, goal_state, obstacles)
    
    # Try to plan
    try:
        # ensure first_solution=True is in your sst_planner.py!
        path = planner.plan(animation=False, first_solution=True)
    except Exception:
        return None

    if path and len(path) > 2:
        map_img = rasterize_map(obstacles, start_state, goal_state)
        fixed_path = interpolate_path(path)
        if fixed_path is None: return None
        
        # Normalize
        norm_path = (fixed_path / ENV_SIZE) * 2 - 1 
        return {"map": map_img, "path": norm_path}
        
    return None

# --- Main Execution ---
if __name__ == "__main__":
    cores = multiprocessing.cpu_count()
    print(f"--- Optimized Streaming Generator ---")
    print(f"Target: {NUM_SAMPLES_NEEDED} paths")
    print(f"Cores: {cores} (using imap_unordered)")

    data_buffer = []
    
    # We create a huge list of seeds (more than we need)
    # because many will fail. We just stop when we have enough.
    task_seeds = [np.random.randint(0, 1000000) for _ in range(NUM_SAMPLES_NEEDED * 5)]

    with multiprocessing.Pool(processes=cores) as pool:
        # imap_unordered yields results AS SOON as they finish.
        # chunksize=1 means we don't batch them, keeping the stream smooth.
        result_iterator = pool.imap_unordered(generate_single_sample, task_seeds, chunksize=1)
        
        for res in result_iterator:
            if res is not None:
                #map, path 

                data_buffer.append(res)
                count = len(data_buffer)
                
                # Print update every 20 samples (more responsive now)
                if count % 20 == 0:
                    print(f"Generated {count} / {NUM_SAMPLES_NEEDED}")
                
                # STOP immediately when we hit the target
                if count >= NUM_SAMPLES_NEEDED:
                    print("Target reached. Stopping workers...")
                    pool.terminate() # Kill remaining tasks
                    break

    print("Saving data...")
    torch.save(data_buffer, "100k_iters-first_found-5k_samples.pt")
    print(f"Done! Saved {len(data_buffer)} samples.")