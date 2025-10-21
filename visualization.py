import pygame
import torch
from data_generator import DataGenerator
import numpy as np

import time
import decorify

from pathfinders import batched_astar,batched_bfs_shortest_paths,batched_rstar,batched_rstar_gpu

def visualize():

    n = 1000
    m = n
    animation_time = None
    data_gen = DataGenerator(n,m,(11 ,10),(10, n // 20))

    batch = 100
    current = batch 
    batch_data = []
    batch_path = []

    pathfinding = True

    @decorify.timeit(accuracy=5)
    def pathfind():
        nonlocal batch_data, batch_path
        batch_path = batched_rstar_gpu(batch_data)


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
            screen.fill((40, 245, 35))


            arr = (data.to(dtype=torch.uint8).cpu().numpy() * 255)  # uint8 (n,m) with 0/255
            surf_arr = np.stack([arr.T, arr.T, arr.T], axis=2)  # (m, n, 3) uint8
            # colorize: choose obstacle and background colors (R,G,B)
            obstacle_color = np.array([200, 30, 30], dtype=np.uint8)     # red-ish
            background_color = np.array([40, 245, 35], dtype=np.uint8)   # green-ish
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


            pygame.display.flip()

            rendered = True









if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((1000, 1000))
    pygame.display.set_caption("Visualization Module")
    visualize()
    pygame.quit()
