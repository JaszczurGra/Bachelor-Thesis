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




def batched_rstar(viable: torch.Tensor, starts: torch.Tensor = None, ends: torch.Tensor = None,
                  diagonal_cost: float = 1.41, max_steps: int = 1000):
    """
    Simplified R*-like batched planner (GPU-aware, implemented similar to batched_astar).
    - viable: [B, W, H] bool, True = obstacle (same as your data); function will treat traversable = ~viable
    - starts, ends: optional [B,2] long tensors (x=row, y=col)
    - diagonal_cost: cost for diagonal moves (default 1.41)
    Returns: list of paths (torch.LongTensor per path on same device) or None if not found.

    Note: This is a simplified R*-style planner (uses global A*-like search with randomized local tie-breaking).
    It supports 8-connected moves with diagonal_cost. For large batches/grids consider streaming to avoid OOM.
    """
    device = viable.device
    B, W, H = viable.shape

    traversable = ~viable  # True = can visit

    # prepare starts/ends
    starts,ends = random_starts_ends(viable) if (starts is None or ends is None) else (starts, ends)

    # setup
    INF = float('inf')
    g_cost = torch.full((B, W, H), INF, device=device, dtype=torch.float32)
    f_cost = torch.full((B, W, H), INF, device=device, dtype=torch.float32)
    open_set = torch.zeros((B, W, H), dtype=torch.bool, device=device)
    came_from = -torch.ones((B, W, H, 2), dtype=torch.long, device=device)

    # heuristic: euclidean (works with diagonal cost ~ sqrt2)
    def heuristic_xy(sx, sy, gx, gy):
        dx = (sx - gx).abs().to(torch.float32)
        dy = (sy - gy).abs().to(torch.float32)
        return torch.sqrt(dx * dx + dy * dy)  # float tensor

    # initialize
    batch_idx = torch.arange(B, device=device)
    for b in range(B):
        sx, sy = starts[b]
        gx, gy = ends[b]
        g_cost[b, sx, sy] = 0.0
        f_cost[b, sx, sy] = heuristic_xy(starts[b:b+1,0], starts[b:b+1,1], ends[b:b+1,0], ends[b:b+1,1]).item()
        open_set[b, sx, sy] = True

    # 8-neighborhood with costs
    neigh = [ (1,0,1.0), (-1,0,1.0), (0,1,1.0), (0,-1,1.0),
              (1,1,diagonal_cost), (1,-1,diagonal_cost), (-1,1,diagonal_cost), (-1,-1,diagonal_cost) ]

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    max_iters = (max_steps if max_steps is not None else (W * H))
    for _ in range(max_iters):
        # if no open nodes, break
        if not open_set.any():
            break

        # pick min f per batch
        fc = f_cost.masked_fill(~open_set, INF)
        flat_idx = fc.view(B, -1).argmin(dim=1)
        cx = flat_idx // H
        cy = flat_idx % H

        for b in range(B):
            if finished[b]:
                continue
            x, y = int(cx[b].item()), int(cy[b].item())
            open_set[b, x, y] = False

            # goal reached
            if (x == ends[b,0].item()) and (y == ends[b,1].item()):
                finished[b] = True
                continue

            for dx, dy, cost in neigh:
                nx, ny = x + dx, y + dy
                if nx < 0 or ny < 0 or nx >= W or ny >= H:
                    continue
                if not traversable[b, nx, ny]:
                    continue

                tentative = g_cost[b, x, y] + cost
                if tentative < g_cost[b, nx, ny]:
                    came_from[b, nx, ny, 0] = x
                    came_from[b, nx, ny, 1] = y
                    g_cost[b, nx, ny] = tentative
                    # heuristic for neighbor
                    h = heuristic_xy(torch.tensor([nx], device=device), torch.tensor([ny], device=device),
                                     ends[b:b+1,0], ends[b:b+1,1]).item()
                    f_cost[b, nx, ny] = tentative + h
                    open_set[b, nx, ny] = True

        if finished.all():
            break

    # reconstruct paths
    paths = []
    for b in range(B):
        if not finished[b]:
            paths.append(None)
            continue
        path = []
        x, y = int(ends[b,0].item()), int(ends[b,1].item())
        while not (x == starts[b,0].item() and y == starts[b,1].item()):
            path.append((x, y))
            px = came_from[b, x, y, 0].item()
            py = came_from[b, x, y, 1].item()
            # safety if broken
            if px < 0 or py < 0:
                break
            x, y = px, py
        path.append((starts[b,0].item(), starts[b,1].item()))
        path.reverse()
        paths.append(torch.tensor(path, dtype=torch.long, device=device))
    return paths


def batched_rstar_gpu(viable: torch.Tensor, starts: torch.Tensor = None, ends: torch.Tensor = None,
                      diagonal_cost: float = 1.41, max_iters: int | None = None):
    """
    GPU-first batched A*/R*-style planner (vectorized).
    - viable: [B, W, H] bool, True = obstacle
    - starts, ends: optional [B,2] long tensors (x=row, y=col)
    Returns: list of paths (torch.LongTensor per path on same device) or None for no path.
    Notes:
    - All heavy work stays on device. Only path reconstruction loops on CPU per found path.
    - Uses flattened index arithmetic and batched scatter/gather to update neighbors.
    - Diagonal moves supported with specified diagonal_cost.
    """
    device = viable.device
    B, W, H = viable.shape
    N = W * H
    traversable = ~viable  # True = can step here

    # prepare starts / ends
    if starts is None or ends is None:
        starts, ends = random_starts_ends(viable)
    starts = starts.to(device)
    ends = ends.to(device)

    # flatten indices
    start_idx = (starts[:, 0] * H + starts[:, 1]).to(torch.long)  # (B,)
    goal_idx  = (ends[:, 0] * H + ends[:, 1]).to(torch.long)      # (B,)

    INF = 1e9
    # flattened arrays (B, N)
    g_flat = torch.full((B, N), INF, device=device, dtype=torch.float32)
    f_flat = torch.full((B, N), INF, device=device, dtype=torch.float32)
    open_flat = torch.zeros((B, N), dtype=torch.bool, device=device)
    came_from = -torch.ones((B, N), dtype=torch.long, device=device)  # store parent flat idx

    # neighbor offsets and costs (row-major: idx = x*H + y)
    neigh_dx = torch.tensor([1, -1,  0,  0,  1,  1, -1, -1], device=device, dtype=torch.long)
    neigh_dy = torch.tensor([0,  0,  1, -1,  1, -1,  1, -1], device=device, dtype=torch.long)
    neigh_costs = torch.tensor([1.0, 1.0, 1.0, 1.0,
                                diagonal_cost, diagonal_cost, diagonal_cost, diagonal_cost],
                               device=device, dtype=torch.float32)  # (K,)
    K = neigh_costs.numel()

    # helper to compute heuristic (Euclidean)
    gx = ends[:, 0].to(torch.float32)  # (B,)
    gy = ends[:, 1].to(torch.float32)

    def heuristic_xy(nx, ny):
        dx = (nx.to(torch.float32) - gx.view(B, 1))
        dy = (ny.to(torch.float32) - gy.view(B, 1))
        return torch.sqrt(dx * dx + dy * dy)  # (B, K)

    # initialize starts
    g_flat[torch.arange(B, device=device), start_idx] = 0.0
    # initial f = heuristic from start
    sx = starts[:, 0].to(torch.float32)
    sy = starts[:, 1].to(torch.float32)
    f_flat[torch.arange(B, device=device), start_idx] = torch.sqrt((sx - gx) ** 2 + (sy - gy) ** 2)
    open_flat[torch.arange(B, device=device), start_idx] = True

    # precompute bounds
    Wm1 = W - 1
    Hm1 = H - 1

    active = torch.ones(B, dtype=torch.bool, device=device)
    max_iters = max_iters or N
    for _iter in range(max_iters):
        if not open_flat.any():
            break

        # pick current flat idx per batch (min f among open)
        masked_f = f_flat.clone()
        masked_f[~open_flat] = INF
        cur_flat = masked_f.argmin(dim=1)  # (B,)

        # deactivate batches with no open nodes (argmin gives some index but mask says none)
        has_open = open_flat.any(dim=1)
        active = active & has_open
        if not active.any():
            break

        # compute x,y of current
        cur_x = (cur_flat // H).to(torch.long)  # (B,)
        cur_y = (cur_flat % H).to(torch.long)   # (B,)

        # gather g at current
        g_at_cur = g_flat.gather(1, cur_flat.view(B, 1)).view(B)  # (B,)

        # build neighbor coords (B, K)
        cx = cur_x.view(B, 1).expand(B, K)
        cy = cur_y.view(B, 1).expand(B, K)
        nx = cx + neigh_dx.view(1, K)
        ny = cy + neigh_dy.view(1, K)

        # validity mask
        valid = (nx >= 0) & (nx <= Wm1) & (ny >= 0) & (ny <= Hm1)
        # neighbor flat indices (clamped to safe range for gathers)
        nbr_flat = (nx * H + ny).clamp(0, N - 1)  # (B,K)

        # neighbor g and f values
        nbr_g = g_flat.gather(1, nbr_flat)  # (B,K)
        # tentative g: (B,1) + (K,) -> (B,K)
        tentative = g_at_cur.view(B, 1) + neigh_costs.view(1, K)

        # mask out non-traversable and already obstacles
        # build traversable_flat to check traversability per neighbor
        # traversable is [B,W,H] -> flatten
        traversable_flat = traversable.view(B, N)
        nbr_traversable = traversable_flat.gather(1, nbr_flat)  # (B,K)

        # don't update where invalid or not traversable or batch inactive
        allowed = valid & nbr_traversable & active.view(B, 1)

        # compare tentative < neighbor g (and allowed)
        improve = allowed & (tentative < nbr_g)

        if not improve.any():
            # mark current as closed
            open_flat[torch.arange(B, device=device), cur_flat] = False
            continue

        # compute heuristic for neighbors
        h = heuristic_xy(nx, ny)  # (B,K)
        f_new = tentative + h

        # prepare sources for scatter: where improve -> new value else old
        g_src = torch.where(improve, tentative, nbr_g)
        f_src = torch.where(improve, f_new, f_flat.gather(1, nbr_flat))
        parent_src = torch.where(improve, cur_flat.view(B, 1).expand(B, K), came_from.gather(1, nbr_flat))

        # scatter updates into flattened arrays
        g_flat = g_flat.scatter(1, nbr_flat, g_src)
        f_flat = f_flat.scatter(1, nbr_flat, f_src)
        came_from = came_from.scatter(1, nbr_flat, parent_src)

        # open those improved neighbors
        open_flat = open_flat.scatter(1, nbr_flat, improve)

        # close current nodes
        open_flat[torch.arange(B, device=device), cur_flat] = False

        # check finished batches (goal reached)
        goal_g = g_flat.gather(1, goal_idx.view(B, 1)).view(B)
        finished = goal_g < INF
        if finished.all():
            break

    # reconstruct paths per batch (loop on CPU side but accessing came_from on device)
    paths = []
    came_from_cpu = came_from  # keep on device but index per-batch below
    for b in range(B):
        gg = g_flat[b, goal_idx[b]].item()
        if gg >= INF:
            paths.append(None)
            continue
        cur = int(goal_idx[b].item())
        path = []
        start = int(start_idx[b].item())
        # follow parents
        while cur >= 0 and cur != start:
            x = cur // H
            y = cur % H
            path.append((x, y))
            parent = came_from_cpu[b, cur].item()
            if parent == cur or parent < 0:
                break
            cur = int(parent)
        # append start
        sx = start // H
        sy = start % H
        path.append((sx, sy))
        path.reverse()
        paths.append(torch.tensor(path, dtype=torch.long, device=device))
    return paths
# ...existing code...