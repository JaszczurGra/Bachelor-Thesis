# full RRT* implementation with pygame visualization
import sys
import math
import numpy as np
from PIL import Image
import pygame

# optional dubins steering for kinodynamic connections
try:
    import dubins
    _HAS_DUBINS = True
except Exception:
    _HAS_DUBINS = False

# color defs
COLOR_FREE = (255, 255, 255)
COLOR_OBS = (0, 0, 0)
COLOR_START = (0, 200, 0)
COLOR_GOAL = (200, 0, 0)
COLOR_TREE = (0, 120, 255)
COLOR_TRAJ = (0, 0, 200)
COLOR_PATH = (200, 0, 200)
COLOR_BEST_PATH = (200,50,50)
BG = (50, 50, 50)


def load_img(path, treshold=0.8):
    img = Image.open(path).convert("L")

    print('Loaded map image size:', img.size    )
    arr = np.array(img).astype(np.uint8)
    arr = (arr / 255.0)
    arr = (arr >= treshold).astype(np.uint8)
    return arr


def _to_surface_from_gray(img8):
    # img8: HxW uint8 or 0/1
    h, w = img8.shape
    values = (img8 * 255).astype(np.uint8)
    rgb = np.stack([values, values, values], axis=2)  # H,W,3
    rgb_swapped = np.transpose(rgb, (1, 0, 2)).copy()  # W,H,3 for blit_array
    surf = pygame.Surface((w, h))
    pygame.surfarray.blit_array(surf, rgb_swapped)
    return surf


def coord_to_screen(pt, img_h):
    x, y = pt[0], pt[1]
    return int(round(x)), int(round(y))


def is_trajectory_collision_free(img, trajectory, robot_radius):
    h, w = img.shape
    for p in trajectory:
        x = int(round(p[0])); y = int(round(p[1]))
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        rx = max(0, x - robot_radius); ry = max(0, y - robot_radius)
        rx2 = min(w - 1, x + robot_radius); ry2 = min(h - 1, y + robot_radius)
        patch = img[ry:ry2+1, rx:rx2+1]
        if patch.size == 0 or np.any(patch == 0):
            return False
    return True


def compute_distance(state1, state2):
    pos_diff = np.linalg.norm(np.array(state1[:2]) - np.array(state2[:2]))
    ang_diff = abs(angle_diff(state1[2], state2[2]))
    return pos_diff + 0.1 * ang_diff


def angle_diff(a1, a2):
    return (a1 - a2 + np.pi) % (2 * np.pi) - np.pi


def find_nearest_node(nodes, point):
    arr = np.array(nodes)
    distances = np.linalg.norm(arr[:, :2] - np.array(point[:2]), axis=1)
    idx = np.argmin(distances)
    return int(idx), arr[idx]


def sample_control(v_max, omega_max, min_turning_radius, straight_prob=1):
    # if np.random.rand() < straight_prob:
    #     v = np.clip(abs(np.random.randn() * (v_max / 2)), 0.5, v_max)
    #     return np.array([v, np.random.random()* 60])
    v = abs(np.random.randn() * (v_max / 2))
    v = np.clip(v, 0.5, v_max)
    omega = np.random.randn() * (omega_max / 3)
    omega = np.clip(omega, -omega_max, omega_max)
    if abs(omega) > 1e-6:
        radius = abs(v / omega)
        if radius < min_turning_radius:
            omega = np.sign(omega) * abs(v / min_turning_radius)
    return np.array([v, omega])


def generate_trajectory_euler(state, control, step_size,dt = (0.1, 1)):
    dt = dt[0] + np.random.random() * dt[1]
    num_steps = step_size
    trajectory = np.zeros((num_steps + 1, 3))
    trajectory[0, :] = state
    current_state = np.array(state, dtype=float).copy()
    v, omega = control
    if abs(omega) < 0.07:
        omega = 0.0
    for i in range(num_steps):
        x_dot = v * np.cos(current_state[2])
        y_dot = v * np.sin(current_state[2])
        theta_dot = omega
        current_state[0] += x_dot * dt
        current_state[1] += y_dot * dt
        current_state[2] += theta_dot * dt
        current_state[2] = (current_state[2] + np.pi) % (2 * np.pi) - np.pi
        trajectory[i + 1, :] = current_state
    return current_state.copy(), trajectory


def reconstruct_path(tree, goal_idx=None):
    if goal_idx is None:
        idx = len(tree['nodes']) - 1
    else:
        idx = goal_idx
    if idx <= 0:
        return np.empty((0, 3)), float('inf')
    segments = []
    while idx != 0:
        traj = tree['trajectories'][idx]
        if traj is None:
            break
        segments.insert(0, traj)
        idx = tree['parent'][idx]
    if len(segments) == 0:
        return np.empty((0, 3)), tree['path_costs'][-1] if tree['path_costs'] else float('inf')
    full = np.vstack(segments)
    return full, tree['path_costs'][-1]


# kinodynamic helpers for RRT*
def compute_traj_cost(traj):
    if traj is None or len(traj) < 2:
        return 0.0
    pts = traj[:, :2]
    return np.linalg.norm(pts[1:] - pts[:-1], axis=1).sum()


# ...existing code...
def kinodynamic_connect(q_from, q_to, min_turning_radius, dt = (0.1, 1), v_nom=5):
    #TODO check lower stepsicze 
    q0 = (float(q_from[0]), float(q_from[1]), float(q_from[2]))
    q1 = (float(q_to[0]), float(q_to[1]), float(q_to[2]))


    # if _HAS_DUBINS:
    #     try:
    #         path = dubins.shortest_path(q0, q1, min_turning_radius)
    #         configurations, _ = path.sample_many(step_size)
    #         traj = np.array(configurations)
    #         return traj
    #     except Exception:
    #         return None


    # fallback: forward-simulate a feasible kinodynamic trajectory
    # simple P-controller for heading + constant forward speed, respects min_turning_radius
    # returns None if connector fails to reach the goal within limits
    omega_max = max(1e-3, v_nom / max(1e-3, min_turning_radius))
    dt = dt[0] + dt[1] * np.random.rand()  # use step_size as time step for simulation
    max_steps = int(max(50, math.ceil(math.hypot(q1[0]-q0[0], q1[1]-q0[1]) / max(1e-3, v_nom)) * 3))

    cur = np.array([q0[0], q0[1], q0[2]], dtype=float)
    traj = [cur.copy()]
    pos_tol = 2.0   # pixels tolerance
    ang_tol = 0.2   # radians tolerance

    for _ in range(max_steps):
        # heading to goal
        dx = q1[0] - cur[0]
        dy = q1[1] - cur[1]
        dist = math.hypot(dx, dy)
        if dist < pos_tol and abs(angle_diff(q1[2], cur[2])) < ang_tol:
            return np.array(traj)

        desired_yaw = math.atan2(dy, dx) if dist > 1e-6 else q1[2]
        yaw_err = angle_diff(desired_yaw, cur[2])

        # choose omega to reduce yaw error; scaled so omega*dt approximates yaw_err
        omega = yaw_err / max(dt, 1e-6)
        omega = max(-omega_max, min(omega_max, omega))

        # if close and orientation mismatch, try turning in place by reducing v
        if dist < 4.0:
            v = max(0.5, v_nom * (dist / 4.0))
        else:
            v = v_nom

        # integrate (same form as generate_trajectory_euler)
        cur[0] += v * math.cos(cur[2]) * dt
        cur[1] += v * math.sin(cur[2]) * dt
        cur[2] += omega * dt
        cur[2] = (cur[2] + math.pi) % (2 * math.pi) - math.pi

        traj.append(cur.copy())

    # final check
    dx = q1[0] - cur[0]
    dy = q1[1] - cur[1]
    if math.hypot(dx, dy) < pos_tol and abs(angle_diff(q1[2], cur[2])) < ang_tol:
        return np.array(traj)
    return None
# ...existing code...

# def kinodynamic_connect(q_from, q_to, min_turning_radius, step_size=0.5):
#     q0 = (float(q_from[0]), float(q_from[1]), float(q_from[2]))
#     q1 = (float(q_to[0]), float(q_to[1]), float(q_to[2]))
#     if _HAS_DUBINS:
#         try:
#             path = dubins.shortest_path(q0, q1, min_turning_radius)
#             configurations, _ = path.sample_many(step_size)
#             traj = np.array(configurations)
#             return traj
#         except Exception:
#             return None

#     # If dubins not available, do NOT fall back to straight interpolation.
#     # Returning None ensures rewiring won't accept non-kinodynamic connectors.
#     return None
# def kinodynamic_connect(q_from, q_to, min_turning_radius, step_size=0.5):
#     q0 = (float(q_from[0]), float(q_from[1]), float(q_from[2]))
#     q1 = (float(q_to[0]), float(q_to[1]), float(q_to[2]))
#     if _HAS_DUBINS:
#         try:
#             path = dubins.shortest_path(q0, q1, min_turning_radius)
#             configurations, _ = path.sample_many(step_size)
#             traj = np.array(configurations)
#             return traj
#         except Exception:
#             pass
#     # fallback straight interpolation with heading interpolation
#     dist = math.hypot(q1[0] - q0[0], q1[1] - q0[1])
#     if dist < 1e-6:
#         return np.array([[q0[0], q0[1], q0[2]]])
#     steps = max(1, int(math.ceil(dist / step_size)))
#     xs = np.linspace(q0[0], q1[0], steps + 1)
#     ys = np.linspace(q0[1], q1[1], steps + 1)
#     thetas = np.linspace(q0[2], q1[2], steps + 1)
#     traj = np.zeros((steps + 1, 3))
#     traj[:, 0] = xs
#     traj[:, 1] = ys
#     traj[:, 2] = thetas
#     return traj


def rewire_neighbors(tree, new_idx, img, min_turning_radius, robot_radius, neighbor_indices, max_rewires=7):
    # Attempt to reparent each neighbor to new_idx when it reduces cost
    new_q = tree['nodes'][new_idx]
    rewires_done = 0
    for nb in neighbor_indices:
        if rewires_done >= max_rewires: 
            print('Max rewires')
            break
        if nb == new_idx:
            continue
        traj = kinodynamic_connect(new_q, tree['nodes'][nb], min_turning_radius)
        if traj is None:
            continue
        if not is_trajectory_collision_free(img, traj, robot_radius):
            continue
        new_cost = tree['path_costs'][new_idx] + compute_traj_cost(traj)
        if new_cost + 1e-9 < tree['path_costs'][nb]:
            tree['parent'][nb] = new_idx
            tree['trajectories'][nb] = traj
            tree['path_costs'][nb] = new_cost
            rewires_done += 1 
            # propagate cost updates to descendants (BFS)
            q = [nb]
            while q:
                cur = q.pop(0)
                children = [i for i, p in enumerate(tree['parent']) if p == cur and i != cur]
                for c in children:
                    edge_cost = compute_traj_cost(tree['trajectories'][c]) if tree['trajectories'][c] is not None else 0.0
                    tree['path_costs'][c] = tree['path_costs'][cur] + edge_cost
                    q.append(c)


def neighbors_within_radius(tree, point, radius):
    arr = np.array(tree['nodes'])
    dists = np.linalg.norm(arr[:, :2] - np.array(point[:2]), axis=1)
    return [i for i, d in enumerate(dists) if d <= radius]


def kinodynamic_rrt_star(img, start, goal, max_iter, step_size, goal_radius,
                         v_max, omega_max, min_turning_radius, flag_show_runtime,
                         robot_radius=1,dt=(0.5, 1.5), scale=1):
    h, w = img.shape
    display_w = int(w * scale)
    display_h = int(h * scale)

    pygame.init()
    screen = pygame.display.set_mode((display_w, display_h))
    pygame.display.set_caption("RRT* Kinodynamic")
    clock = pygame.time.Clock()

    map_surf = _to_surface_from_gray((img ).astype(np.uint8))
    if scale != 1:
        map_surf = pygame.transform.scale(map_surf, (display_w, display_h))

    tree_surf = pygame.Surface((display_w, display_h))
    tree_surf.fill(BG)
    tree_surf.blit(map_surf, (0, 0))
    path_surf = pygame.Surface((display_w, display_h), pygame.SRCALPHA)
    path_surf.fill((0, 0, 0, 0))
 

    tree = {
        'nodes': [np.array(start, dtype=float)],
        'parent': [0],
        'controls': [None],
        'trajectories': [None],
        'path_costs': [0.0]
    }

    update_interval_ms = 200
    last_update = pygame.time.get_ticks()
    s_px = coord_to_screen(start, h); g_px = coord_to_screen(goal, h)
    if scale != 1:
        s_px = (int(s_px[0]*scale), int(s_px[1]*scale))
        g_px = (int(g_px[0]*scale), int(g_px[1]*scale))
    pygame.draw.circle(tree_surf, COLOR_START, s_px, max(3, int(4*scale)))
    pygame.draw.circle(tree_surf, COLOR_GOAL, g_px, max(3, int(4*scale)))
    pygame.draw.circle(tree_surf, (80, 80, 80), g_px, int(goal_radius*scale), 1)
    screen.blit(tree_surf, (0, 0))
    pygame.display.flip()


    min_path_cost = float("inf")
    current_path_cost = float('inf')
    best_path = []
    found_paths = []
    running = True
    for it in range(max_iter):

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break
        if not running:
            break

        # sample
        if np.random.rand() < 0.2:
            sample = goal
        else:
            sample = [np.random.randint(1, w), np.random.randint(1, h), np.random.rand() * 2 * np.pi]

        # nearest node
        nearest_idx, nearest_node = find_nearest_node(tree['nodes'], sample)

        # try multiple controls from nearest to produce candidate
        best_candidate = None
        best_parent_idx = nearest_idx
        best_candidate_cost = float('inf')
        best_candidate_traj = None
        # produce several trajectory candidates from nearest
        # TODO fine tune this 
        for _ in range(10):
            control = sample_control(v_max, omega_max, min_turning_radius)
            new_state, trajectory = generate_trajectory_euler(nearest_node, control, step_size,dt)
            if not is_trajectory_collision_free(img, trajectory, robot_radius):
                continue
            # use geometric distance to sample point as heuristic
            dist = np.linalg.norm(new_state[:2] - np.array(sample[:2]))
            if dist > 1000:
                continue
            # candidate initial cost (from nearest)
            cand_cost = tree['path_costs'][nearest_idx] + compute_traj_cost(trajectory)
            if cand_cost < best_candidate_cost:
                best_candidate = new_state
                best_parent_idx = nearest_idx
                best_candidate_cost = cand_cost
                best_candidate_traj = trajectory

        if best_candidate is None:
            continue

        # neighbor radius (RRT*). simple constant scaled by map size and log factor
        gamma = 50 #TODO  tuning parameter
        neighbor_radius = int(max(100, gamma * math.sqrt(math.log(max(2, len(tree['nodes']))) / max(1, len(tree['nodes'])))))
        neighbor_idx_list = neighbors_within_radius(tree, best_candidate, neighbor_radius)

        # choose best parent among neighbors (try kinodynamic_connect from neighbor -> candidate)
        chosen_parent = best_parent_idx
        chosen_traj = best_candidate_traj
        chosen_cost = best_candidate_cost
        for nb in neighbor_idx_list:
            if nb == chosen_parent:
                continue
            # attempt kinodynamic connect nb -> best_candidate
            traj_nb = kinodynamic_connect(tree['nodes'][nb], best_candidate, min_turning_radius,dt,v_max)
            if traj_nb is None:
                continue
            if not is_trajectory_collision_free(img, traj_nb, robot_radius):
                continue
            cost_nb = tree['path_costs'][nb] + compute_traj_cost(traj_nb)
            if cost_nb + 1e-9 < chosen_cost:
                chosen_parent = nb
                chosen_traj = traj_nb
                chosen_cost = cost_nb

        # append new node with chosen parent & trajectory
        new_idx = len(tree['nodes'])
        tree['nodes'].append(np.array(best_candidate, dtype=float))
        tree['parent'].append(int(chosen_parent))
        tree['controls'].append(None)
        tree['trajectories'].append(chosen_traj)
        tree['path_costs'].append(chosen_cost)

        # Rewire neighbors: attempt to reparent neighbors to new node if cheaper
        # TODO do this every turn 
        if it % 50 == 0:
            print('rewiring')
            rewire_neighbors(tree, new_idx, img, min_turning_radius, robot_radius, neighbor_idx_list)

        # draw the edge (parent->new)
        if chosen_traj is not None and len(chosen_traj) >= 2:
            pts = chosen_traj[:, :2]
            pts_draw = [(int(round(p[0]*scale)), int(round(p[1]*scale))) for p in pts]
            if len(pts_draw) >= 2:
                pygame.draw.lines(tree_surf, COLOR_TREE, False, pts_draw, max(1, int(1)))
        now = pygame.time.get_ticks()
        if flag_show_runtime and (now - last_update) >= update_interval_ms:
            screen.blit(tree_surf, (0, 0))
            screen.blit(path_surf, (0, 0))
            pygame.display.flip()
            last_update = now

        # check goal reach
        if np.linalg.norm(np.array(best_candidate[:2]) - np.array(goal[:2])) <= goal_radius:
            full_traj, final_cost = reconstruct_path(tree)
            path_surf.fill((0, 0, 0, 0))
            if full_traj.size > 0:
                pts = full_traj[:, :2]
                pts_draw = [(int(round(p[0]*scale)), int(round(p[1]*scale))) for p in pts]
                if len(pts_draw) >= 2:
                    pygame.draw.lines(path_surf, COLOR_PATH, False, pts_draw, max(2, int(2*scale)))
            if len(best_path) > 0: 
                pts = best_path[:, :2]
                pts_draw = [(int(round(p[0]*scale)), int(round(p[1]*scale))) for p in pts]
                if len(pts_draw) >= 2:
                    pygame.draw.lines(path_surf, COLOR_BEST_PATH, False, pts_draw, max(2, int(2*scale)))

            screen.blit(tree_surf, (0, 0))
            screen.blit(path_surf, (0, 0))
            pygame.display.flip()

            if final_cost < min_path_cost:
                min_path_cost = final_cost
                best_path = full_traj
            current_path_cost = final_cost
            # print(f"\nFound path with cost {final_cost:.2f} at iter {it}")
            # found_paths.append(full_traj)
        print(f"\rIteration: {it}, min cost: {min_path_cost}, current cost: {current_path_cost}",end=" ")

    print("RRT* finished")
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
        screen.blit(tree_surf, (0, 0))
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()
    return found_paths


if __name__ == "__main__":
    try:
        img = load_img("Paths\\Pictures\\map_1_d.png")
    except Exception:
        print('Failed to load map image, using default simple map.')
        img = np.ones((300, 300), dtype=np.uint8) * 255
        img[140:160, 50:250] = 0

    start = [30, 50, 0.0]
    goal = [270, 200, np.pi / 2]
    goal_radius = 15


    #TODO change the konection to respect 
    kinodynamic_rrt_star(img, start, goal, max_iter=200000, step_size=8,
                         goal_radius=goal_radius, v_max=30, omega_max=np.pi / 9,
                         min_turning_radius=10.0, flag_show_runtime=True, robot_radius=5, scale=3)