import pygame
import torch
from data_generator import DataGenerator
import numpy as np

import time
import decorify

from pathfinders import batched_astar,batched_bfs_shortest_paths,batched_rstar,batched_rstar_gpu

def visualize():

    n = 100
    m = n
    animation_time = None
    data_gen = DataGenerator(n,m,(11 ,10),(10, n // 20))

    batch = 100
    current = batch 
    batch_data = []
    batch_path = []
    batch_starts = None
    batch_ends = None

    pathfinding = True

    @decorify.timeit(accuracy=5)
    def pathfind():
        nonlocal batch_data, batch_path
        nonlocal batch_starts, batch_ends
        # ensure starts/ends are on traversable cells (i.e., not inside obstacles)
        # batch_data: [B, n, m] boolean tensor, True = obstacle
        B = batch_data.shape[0]
        starts = torch.zeros((B, 2), dtype=torch.long, device=batch_data.device)
        ends = torch.zeros((B, 2), dtype=torch.long, device=batch_data.device)
        for b in range(B):
            traversable = (~batch_data[b]).to(torch.bool)
            idxs = torch.nonzero(traversable, as_tuple=False)
            if idxs.shape[0] == 0:
                # fallback to center
                starts[b, 0] = batch_data.shape[1] // 2
                starts[b, 1] = batch_data.shape[2] // 2
                ends[b] = starts[b]
                continue
            if idxs.shape[0] == 1:
                starts[b] = idxs[0]
                ends[b] = idxs[0]
                continue
            # pick two distinct random traversable indices
            perm = torch.randperm(idxs.shape[0], device=batch_data.device)
            s_idx = idxs[perm[0]]
            e_idx = idxs[perm[1]]
            starts[b, 0] = s_idx[0]
            starts[b, 1] = s_idx[1]
            ends[b, 0] = e_idx[0]
            ends[b, 1] = e_idx[1]

        batch_starts = starts
        batch_ends = ends
        batch_path = batched_rstar_gpu(batch_data, starts, ends)


    def next_data():    
        if batch == 1:
            @decorify.timeit(accuracy=5)
            def gen():
                return torch.tensor(data_gen.generate_data_cpu())
            data= gen()
            print ('Time taken to generate:', data[1])
            return data[0]

        nonlocal current, batch_data
        if current < batch:
            current += 1
            return batch_data[current - 1]
        else:
            current = 0
            @decorify.timeit(accuracy=5)
            def gen():
                return data_gen.generate_data(batch=batch)

            
            batch_data = gen()
            print ('Time taken to generate:', batch_data[1])
            batch_data = batch_data[0]
            if pathfinding:
                print('Pathfinding took:', pathfind()[1])

            return next_data()
        





    running = True
    data = next_data()

    # If we're running with batch == 1, compute a safe start/end and path for the single sample
    if batch == 1 and pathfinding:
        # data is a single grid (n,m) boolean tensor on device
        single = data
        if isinstance(single, torch.Tensor):
            traversable = (~single).to(torch.bool)
            idxs = torch.nonzero(traversable, as_tuple=False)
            if idxs.shape[0] == 0:
                s = (single.shape[0] // 2, single.shape[1] // 2)
                e = s
            elif idxs.shape[0] == 1:
                s = (int(idxs[0,0].item()), int(idxs[0,1].item()))
                e = s
            else:
                perm = torch.randperm(idxs.shape[0])
                s = (int(idxs[perm[0],0].item()), int(idxs[perm[0],1].item()))
                e = (int(idxs[perm[1],0].item()), int(idxs[perm[1],1].item()))
            starts = torch.tensor([s], dtype=torch.long)
            ends = torch.tensor([e], dtype=torch.long)
            batch_starts = starts
            batch_ends = ends
            batch_path = batched_rstar_gpu(single.unsqueeze(0), starts, ends)

    pixelsize = (pygame.display.get_window_size()[0]//n, pygame.display.get_window_size()[1]//m)

    time_start = animation_time
    rendered = False
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                data = next_data()
                rendered = False


        if animation_time is not None:
            if pygame.time.get_ticks() / 1000 >= time_start:
                data = next_data()
                time_start =  pygame.time.get_ticks() / 1000 + animation_time
                rendered = False



        if not rendered:
            # fill background black
            screen.fill((0, 0, 0))

            arr = (data.to(dtype=torch.uint8).cpu().numpy() * 255)  # uint8 (n,m) with 0/255
            surf_arr = np.stack([arr.T, arr.T, arr.T], axis=2)  # (m, n, 3) uint8
            # colorize: choose obstacle and background colors (R,G,B)
            obstacle_color = np.array([255, 255, 255], dtype=np.uint8)   # white
            background_color = np.array([0, 0, 0], dtype=np.uint8)       # black
            mask = (arr.T != 0)  # shape (m, n)
            surf_arr = np.empty((arr.T.shape[0], arr.T.shape[1], 3), dtype=np.uint8)
            surf_arr[:] = background_color
            surf_arr[mask] = obstacle_color
            surf = pygame.surfarray.make_surface(surf_arr)     # make surface
            surf = pygame.transform.scale(surf, screen.get_size())
            screen.blit(surf, (0, 0))

            if pathfinding and batch_path[current-1] is not None:
                for y,x in batch_path[current-1]:
                    pygame.draw.rect(screen, (30, 144, 255), (x * pixelsize[0], y * pixelsize[1], pixelsize[0], pixelsize[1]))

            # draw start (green) and end (red) if available
            try:
                if pathfinding and batch_starts is not None:
                    bs = batch_starts[current-1]
                    be = batch_ends[current-1]
                    # bs/be may be torch tensors or numpy; ensure ints
                    sx, sy = int(bs[0].item()) if hasattr(bs[0], 'item') else int(bs[0]), int(bs[1].item()) if hasattr(bs[1], 'item') else int(bs[1])
                    ex, ey = int(be[0].item()) if hasattr(be[0], 'item') else int(be[0]), int(be[1].item()) if hasattr(be[1], 'item') else int(be[1])
                    # draw as single-cell rectangles (col = y, row = x)
                    pygame.draw.rect(screen, (0, 255, 0), (sy * pixelsize[0], sx * pixelsize[1], pixelsize[0], pixelsize[1]))
                    pygame.draw.rect(screen, (255, 0, 0), (ey * pixelsize[0], ex * pixelsize[1], pixelsize[0], pixelsize[1]))
            except Exception:
                # defensive: don't crash rendering if indexing fails
                pass


            pygame.display.flip()

            rendered = True









if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    pygame.display.set_caption("Visualization Module")
    visualize()
    pygame.quit()
