import torch


def random_starts_ends(viable: torch.Tensor):   
    """
    viable: [B, W, H] boolean tensor, True = traversable
    Returns: starts, ends: [B, 2] tensors of (x, y) coords
    """
    device = viable.device
    B, W, H = viable.shape

    starts = torch.zeros(viable.shape[0], 2, dtype=torch.long, device=viable.device)
    starts[:, 0] = torch.randint(0, W, (B,), device=device)
    starts[:, 1] = torch.randint(0, H, (B,), device=device)

    ends = torch.zeros(viable.shape[0], 2, dtype=torch.long, device=viable.device)
    ends[:, 0] = torch.randint(0, W, (B,), device=device)
    ends[:, 1] = torch.randint(0, H, (B,), device=device)

    return starts, ends


def batched_astar(viable: torch.Tensor, starts: torch.Tensor = None, ends: torch.Tensor = None):
    """
    viable: [B, W, H] boolean tensor, True = traversable
    starts: [B, 2] tensor of (x, y) start coords
    ends: [B, 2] tensor of (x, y) goal coords
    Returns: list of paths (as list of coordinates tensors)
    """
    device = viable.device
    B, W, H = viable.shape

    viable = ~viable


    starts,ends = random_starts_ends(viable) if (starts is None or ends is None) else (starts, ends)
    # ends[:,0] = viable.shape[1] - 1
    # ends[:,1] = viable.shape[2] - 1



    # Initialize costs
    g_cost = torch.full((B, W, H), float('inf'), device=device)
    f_cost = torch.full((B, W, H), float('inf'), device=device)
    open_set = torch.zeros((B, W, H), dtype=torch.bool, device=device)
    came_from = -torch.ones((B, W, H, 2), dtype=torch.long, device=device)

    # Heuristic: Manhattan distance 
    def heuristic(a, b):
        return (a[:, 0] - b[:, 0]).abs() + (a[:, 1] - b[:, 1]).abs()

    # Initialize starts
    for b in range(B):
        sx, sy = starts[b]
        gx, gy = ends[b]
        g_cost[b, sx, sy] = 0
        f_cost[b, sx, sy] = heuristic(starts[b:b+1], ends[b:b+1])
        open_set[b, sx, sy] = True

    # 4-neighborhood
    dirs = torch.tensor([[1,0],[-1,0],[0,1],[0,-1]], device=device)

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    max_steps = W * H

    for _ in range(max_steps):
        # pick next node per batch
        mask = open_set.clone()
        if not mask.any():
            break

        # Get node with min f_cost in each batch
        fc = f_cost.masked_fill(~mask, float('inf'))
        flat_idx = fc.view(B, -1).argmin(dim=1)
        cx, cy = flat_idx // H, flat_idx % H

        for b in range(B):
            if finished[b]: continue
            x, y = cx[b].item(), cy[b].item()
            open_set[b, x, y] = False

            # goal reached
            if (x == ends[b,0]) and (y == ends[b,1]):
                finished[b] = True
                continue

            for dx, dy in dirs:
                nx, ny = x + dx.item(), y + dy.item()
                if nx < 0 or ny < 0 or nx >= W or ny >= H:
                    continue
                if not viable[b, nx, ny]:
                    continue

                tentative_g = g_cost[b, x, y] + 1
                if tentative_g < g_cost[b, nx, ny]:
                    came_from[b, nx, ny] = torch.tensor([x, y], device=device)
                    g_cost[b, nx, ny] = tentative_g
                    f_cost[b, nx, ny] = tentative_g + heuristic(
                        torch.tensor([[nx, ny]], device=device),
                        ends[b:b+1]
                    )
                    open_set[b, nx, ny] = True

        if finished.all():
            break

    # reconstruct paths
    paths = []
    for b in range(B):
        gx, gy = ends[b]
        path = []
        if not finished[b]:
            paths.append(None)
            continue
        x, y = gx.item(), gy.item()
        while (x >= 0 and y >= 0) and not (x == starts[b,0] and y == starts[b,1]):
            path.append((x, y))
            px, py = came_from[b, x, y]
            x, y = px.item(), py.item()
        path.append((starts[b,0].item(), starts[b,1].item()))
        path.reverse()
        paths.append(torch.tensor(path, device=device))
    return paths


if __name__ == "__main__":
    # simple test
    viable = torch.tensor([[[1,1,1,0,1],
                            [0,0,1,0,1],
                            [1,1,1,1,1],
                            [1,0,0,0,0],
                            [1,1,1,1,1]], 
                            [[1,1,1,0,1],
                            [0,0,1,0,1],
                            [1,1,1,1,1],
                            [1,0,0,0,0],
                            [1,1,1,1,1]]
                            ], dtype=torch.bool)
    starts = torch.tensor([[0,0],[0,0]])
    ends = torch.tensor([[4,4],[2,2]])
    paths = batched_astar(viable, starts, ends)
    print(paths)


def batched_bfs_shortest_paths(viable: torch.Tensor, starts: torch.Tensor = None, ends: torch.Tensor = None, max_steps: int = None):
    """
    GPU-friendly batched BFS shortest paths on boolean grids.
    - viable: [B, W, H] bool, True = obstacle (kept same as your data)
    - starts, ends: optional [B,2] long tensors (x=row, y=col)
    Returns list of paths (torch.LongTensor per path on same device) or None if no path.
    Memory: uses O(B*W*H) for distances and a few O(B*W*H) temporaries (no extra k-dim).
    """
    device = viable.device
    B, W, H = viable.shape

    traversable = ~viable  # True = can visit

    starts,ends = random_starts_ends(viable) if (starts is None or ends is None) else (starts, ends)

    # distances: -1 = unvisited, else distance
    dist = -torch.ones((B, W, H), dtype=torch.int32, device=device)
    # frontier boolean
    frontier = torch.zeros((B, W, H), dtype=torch.bool, device=device)

    batch_idx = torch.arange(B, device=device)
    dist[batch_idx, starts[:, 0], starts[:, 1]] = 0
    frontier[batch_idx, starts[:, 0], starts[:, 1]] = True

    # quick check if any end equals start
    reached = (starts[:, 0] == ends[:, 0]) & (starts[:, 1] == ends[:, 1])
    if reached.all():
        return [torch.tensor([[s[0], s[1]]], device=device, dtype=torch.long) for s in starts]

    max_iters = (max_steps if max_steps is not None else (W * H))
    step = 0
    while step < max_iters:
        step += 1

        # expand frontier by 4-neighborhood using roll (no huge temporaries)
        up = torch.roll(frontier, shifts=1, dims=1); up[:, 0, :] = False
        down = torch.roll(frontier, shifts=-1, dims=1); down[:, -1, :] = False
        left = torch.roll(frontier, shifts=1, dims=2); left[:, :, 0] = False
        right = torch.roll(frontier, shifts=-1, dims=2); right[:, :, -1] = False

        nbrs = (up | down | left | right)

        # new frontier: neighbors that are traversable and unvisited
        new_frontier = nbrs & traversable & (dist == -1)

        if not new_frontier.any():
            break

        dist[new_frontier] = step
        frontier = new_frontier

        # check reached ends
        end_dists = dist[batch_idx, ends[:, 0], ends[:, 1]]
        newly_reached = end_dists >= 0
        if newly_reached.all():
            break

    # reconstruct paths by following decreasing distance
    paths = []
    for b in range(B):
        ed = dist[b, ends[b, 0], ends[b, 1]].item()
        if ed < 0:
            paths.append(None)
            continue
        path_coords = []
        cx, cy = ends[b, 0].item(), ends[b, 1].item()
        curd = ed
        path_coords.append((cx, cy))
        while curd > 0:
            # check 4 neighbors for dist == curd-1
            found = False
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < W and 0 <= ny < H:
                    if dist[b, nx, ny].item() == curd - 1:
                        cx, cy = nx, ny
                        curd -= 1
                        path_coords.append((cx, cy))
                        found = True
                        break
            if not found:
                # defensive: should not happen
                break
        path_coords.reverse()
        paths.append(torch.tensor(path_coords, dtype=torch.long, device=device))
    return paths