import math
import random
from collections import deque
from typing import List, Tuple
import itertools

def generate_graph_from_obstacles(obstacles: List[Tuple[float, float, float]],
                                  num_nodes_per_circle: int = 8,
                                  max_connection_distance: float = 120,
                                  num_poisson_points: int = 30,
                                  k_nearest: int = 12):
    """
    Build node graph using:
      - ring nodes around each obstacle
      - Poisson-disc sampled free-space nodes for even coverage
      - k-nearest connections with visibility checks
    Returns: graph(dict), nodes(list), start, goal
    """
    nodes: List[Tuple[float, float]] = []

    # ring nodes around obstacles (keeps obstacle boundary coverage)
    for cx, cy, r in obstacles:
        for i in range(num_nodes_per_circle):
            angle = 2 * math.pi * i / num_nodes_per_circle
            x = cx + math.cos(angle) * (r + 4)  # slightly outside obstacle
            y = cy + math.sin(angle) * (r + 4)
            nodes.append((x, y))

    # start & goal
    start = (10.0, 10.0)
    goal = (190.0, 190.0)
    nodes = [start, goal] + nodes

    # bounding box for sampling
    if obstacles:
        min_x = min(cx - r for cx, cy, r in obstacles) - 20
        max_x = max(cx + r for cx, cy, r in obstacles) + 20
        min_y = min(cy - r for cx, cy, r in obstacles) - 20
        max_y = max(cy + r for cx, cy, r in obstacles) + 20
    else:
        min_x, max_x, min_y, max_y = 0.0, 200.0, 0.0, 200.0

    # helper: test if point is inside any obstacle
    def inside_any_circle(px, py):
        for cx, cy, r in obstacles:
            if (px - cx) ** 2 + (py - cy) ** 2 <= (r + 1e-6) ** 2:
                return True
        return False

    # Poisson-disc sampling (Bridson) for even distribution
    def poisson_disk_samples(width, height, radius, k=30):
        cell_size = radius / math.sqrt(2)
        nx = int(math.ceil(width / cell_size))
        ny = int(math.ceil(height / cell_size))
        grid = [[None for _ in range(ny)] for _ in range(nx)]
        samples = []
        active = []

        def insert(p):
            samples.append(p)
            gx = int((p[0] - min_x) / cell_size)
            gy = int((p[1] - min_y) / cell_size)
            if 0 <= gx < nx and 0 <= gy < ny:
                grid[gx][gy] = p
            active.append(p)

        # initial random point
        for _ in range(30):
            rx = random.uniform(min_x, max_x)
            ry = random.uniform(min_y, max_y)
            if not inside_any_circle(rx, ry):
                insert((rx, ry))
                break
        while active and len(samples) < num_poisson_points:
            idx = random.randrange(len(active))
            base = active[idx]
            found = False
            for _ in range(k):
                a = random.uniform(0, 2 * math.pi)
                r = random.uniform(radius, 2 * radius)
                nxp = base[0] + math.cos(a) * r
                nyp = base[1] + math.sin(a) * r
                if nxp < min_x or nxp > max_x or nyp < min_y or nyp > max_y:
                    continue
                if inside_any_circle(nxp, nyp):
                    continue
                # check neighbors in grid
                gx = int((nxp - min_x) / cell_size)
                gy = int((nyp - min_y) / cell_size)
                ok = True
                for i in range(max(0, gx - 2), min(nx, gx + 3)):
                    for j in range(max(0, gy - 2), min(ny, gy + 3)):
                        p = grid[i][j]
                        if p is None:
                            continue
                        if (p[0] - nxp) ** 2 + (p[1] - nyp) ** 2 < radius * radius:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    insert((nxp, nyp))
                    found = True
                    break
            if not found:
                # remove from active
                active.pop(idx)
        return samples

    # sample free points
    span_x = max_x - min_x
    span_y = max_y - min_y
    # target poisson radius tuned by area and desired count
    approx_area = max(1.0, span_x * span_y)
    radius_guess = max(4.0, math.sqrt(approx_area / max(10, num_poisson_points)) )
    poisson_pts = poisson_disk_samples(span_x, span_y, radius_guess)
    nodes.extend(poisson_pts)

    # visibility test (segment-circle intersection)
    def line_intersects_circle(p1, p2, circle):
        cx, cy, r = circle
        x1, y1 = p1
        x2, y2 = p2
        dx, dy = x2 - x1, y2 - y1
        fx, fy = x1 - cx, y1 - cy
        a = dx * dx + dy * dy
        b = 2 * (fx * dx + fy * dy)
        c = (fx * fx + fy * fy) - r * r
        disc = b * b - 4 * a * c
        if disc < 0:
            return False
        disc = math.sqrt(disc)
        t1 = (-b - disc) / (2 * a)
        t2 = (-b + disc) / (2 * a)
        return (0 <= t1 <= 1) or (0 <= t2 <= 1)

    # connect using k nearest neighbors (fast brute force for small N)
    N = len(nodes)
    dists = [[0.0]*N for _ in range(N)]
    for i in range(N):
        xi, yi = nodes[i]
        for j in range(i+1, N):
            xj, yj = nodes[j]
            d = math.hypot(xi - xj, yi - yj)
            dists[i][j] = d
            dists[j][i] = d

    graph = {i: [] for i in range(N)}
    for i in range(N):
        # get k nearest indices sorted by distance
        idxs = sorted(range(N), key=lambda j: dists[i][j] if i!=j else float('inf'))[:k_nearest+1]
        for j in idxs:
            if i == j:
                continue
            if dists[i][j] > max_connection_distance:
                continue
            p1 = nodes[i]; p2 = nodes[j]
            if any(line_intersects_circle(p1, p2, obs) for obs in obstacles):
                continue
            # add undirected edge
            if j not in graph[i]:
                graph[i].append(j)
            if i not in graph[j]:
                graph[j].append(i)

    return graph, nodes, start, goal


def find_paths(graph, start, goal, max_paths: int = None, max_depth: int = None):
    """
    Find all simple paths from start to goal (no repeated nodes in a path).
    - graph: adjacency dict {node: [neigh,...]}
    - start, goal: node indices
    - max_paths: optional limit to stop early (useful for large graphs)
    - max_depth: optional limit on path length (nodes) to avoid explosion
    Returns: list of paths (each a list of node indices)
    """
    paths = []
    # iterative DFS stack: (current_node, path_list, visited_set)
    stack = [(start, [start], {start})]
    while stack:
        node, path, visited = stack.pop()
        if node == goal:
            paths.append(path)
            if max_paths is not None and len(paths) >= max_paths:
                break
            continue
        if max_depth is not None and len(path) >= max_depth:
            continue
        for nbr in graph.get(node, []):
            if nbr in visited:
                continue
            stack.append((nbr, path + [nbr], visited | {nbr}))
    return paths


def rank_paths( paths, nodes, goal_index):
    #TODO impement ranking based on length
    return paths
    ranked = []
    for path in paths:
        if len(path) == 0:
            ranked.append((float('inf'), path))
            continue
        end_node = path[-1]
        dist_to_goal = math.dist(nodes[end_node], nodes[goal_index])
        ranked.append((dist_to_goal, path))
    ranked.sort(key=lambda x: x[0])
    return [p for _, p in ranked]


import matplotlib.pyplot as plt

def plot_graph_paths(ax,nodes, obstacles, paths,title):

    # draw paths if any
    for i, path in enumerate(paths or []):
        path_x = [nodes[node][0] for node in path]
        path_y = [nodes[node][1] for node in path]
        ax.plot(path_x, path_y, label=f'Path {i+1}', alpha=0.8, linewidth=3)

    # draw nodes

    # start & goal
    ax.scatter([nodes[0][0]], [nodes[0][1]], c='green', s=100, marker='o', label='Start')
    ax.scatter([nodes[1][0]], [nodes[1][1]], c='red', s=100, marker='x', label='Goal')

    # obstacles
    for cx, cy, r in obstacles:
        circle = plt.Circle((cx, cy), r, color='orange', alpha=0.3)
        ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_title(title)
    # ax.legend()



def plot_graph(ax,nodes, obstacles, graph,title):
    for i, neighbors in graph.items():
                x1, y1 = nodes[i]
                for j in neighbors:
                    x2, y2 = nodes[j]
                    ax.plot([x1, x2], [y1, y2], color='gray', alpha=0.5)

    # draw nodes
    xs, ys = zip(*nodes)
    ax.scatter(xs, ys, c='blue', s=30)

    # start & goal
    ax.scatter([nodes[0][0]], [nodes[0][1]], c='green', s=100, marker='o', label='Start')
    ax.scatter([nodes[1][0]], [nodes[1][1]], c='red', s=100, marker='x', label='Goal')

    # obstacles
    for cx, cy, r in obstacles:
        circle = plt.Circle((cx, cy), r, color='orange', alpha=0.3)
        ax.add_patch(circle)

    ax.set_aspect('equal')
    ax.set_title(title)
    # ax.legend()


def plot(nodes, obstacles, graph, paths=None, figsize=(12, 6), titles=("Left", "Right")):
    fig, axs = plt.subplots(1, 2)
    plot_graph(axs[0], nodes, obstacles, graph, titles[0])
    plot_graph_paths(axs[1], nodes, obstacles, paths, titles[1])
    plt.tight_layout()
    plt.show()



# Example usage:
while True:
    obstacles = []
    for i in range(random.randint(5,30)):
        obstacles.append((random.randint(50, 150), random.randint(50, 400), random.randint(15, 30)))
    graph, nodes, start, goal = generate_graph_from_obstacles(obstacles,max_connection_distance=200,num_poisson_points=150)
    paths = find_paths(graph, 0, 1,max_depth=8)
    print(f"Found {len(paths)} paths")
    ranked_paths = rank_paths(paths, nodes, 1)

    plot(nodes, obstacles, graph,ranked_paths[:200])
