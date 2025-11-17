# pygame-based visualization for kinodynamic RRT
# replaces matplotlib plotting with a live pygame window

import sys
import numpy as np
from PIL import Image
import pygame

# color defs
COLOR_FREE = (255, 255, 255)
COLOR_OBS = (0, 0, 0)
COLOR_START = (0, 200, 0)
COLOR_GOAL = (200, 0, 0)
COLOR_TREE = (0, 120, 255)
COLOR_TRAJ = (0, 0, 200)
COLOR_PATH = (200, 0, 200)
BG = (50, 50, 50)



def load_img(path,treshold = 0.8):
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.uint8) 
    arr = (arr / 255.0)
    arr = (arr >= treshold).astype(np.uint8)


    return arr

def _to_surface_from_gray(img8):
    # img8: HxW uint8
    h, w = img8.shape
    rgb = np.stack([img8, img8, img8], axis=2)          # H,W,3
    rgb_swapped = np.transpose(rgb, (1, 0, 2)).copy()   # W,H,3 for blit_array
    surf = pygame.Surface((w, h))
    pygame.surfarray.blit_array(surf, rgb_swapped)
    return surf

def coord_to_screen(pt, img_h):
    x, y = pt[0], pt[1]
    return int(round(x)), int(round(y))

def draw_circle(surface, center, radius, color, img_h, width=1):
    cx, cy = coord_to_screen(center, img_h)
    pygame.draw.circle(surface, color, (cx, cy), int(round(radius)), width)

def draw_line(surface, a, b, color, img_h, width=1):
    ax, ay = coord_to_screen(a, img_h)
    bx, by = coord_to_screen(b, img_h)
    pygame.draw.line(surface, color, (ax, ay), (bx, by), width)

def draw_polyline(surface, points, color, img_h, width=2):
    pts = [coord_to_screen(p, img_h) for p in points]
    if len(pts) >= 2:
        pygame.draw.lines(surface, color, False, pts, width)

def is_trajectory_collision_free(img, trajectory, robot_radius):

    h, w = img.shape
    for p in trajectory:
        x = int(round(p[0])); y = int(round(p[1]))
        # bounds check
        if x < 0 or x >= w or y < 0 or y >= h:
            return False
        # circle footprint check (simple)
        rx = max(0, x - robot_radius); ry = max(0, y - robot_radius)
        rx2 = min(w - 1, x + robot_radius); ry2 = min(h - 1, y + robot_radius)
        patch = img[ry:ry2+1, rx:rx2+1]
        if np.any(patch == 0):
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

def sample_control(v_max, omega_max, min_turning_radius, straight_prob=0.2):
    # sometimes sample a straight control to produce long straight segments
    if np.random.rand() < straight_prob:
        v = np.clip(abs(np.random.randn() * (v_max / 2)), 0.5, v_max)
        return np.array([v, 0.0])
    v = abs(np.random.randn() * (v_max / 2))
    v = np.clip(v, 0.5, v_max)
    omega = np.random.randn() * (omega_max / 3)
    omega = np.clip(omega, -omega_max, omega_max)
    if abs(omega) > 1e-6:
        radius = abs(v / omega)
        if radius < min_turning_radius:
            omega = np.sign(omega) * abs(v / min_turning_radius)
    return np.array([v, omega])

   
    v = abs(np.random.randn() * (v_max / 2))
    v = np.clip(v, 0.5, v_max)
    omega = np.random.randn() * (omega_max / 3)
    omega = np.clip(omega, -omega_max, omega_max)
    if abs(omega) > 1e-6:
        radius = abs(v / omega)
        if radius < min_turning_radius:
            omega = np.sign(omega) * abs(v / min_turning_radius)
    return np.array([v, omega]) 



def generate_trajectory_euler(state, control, step_size):
    dt = 0.5 + 1.5 * np.random.rand()
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

def reconstruct_path(tree):
    idx = len(tree['nodes']) - 1
    if idx <= 0:
        return np.empty((0,3)), float('inf')
    segments = []
    while idx != 0:
        traj = tree['trajectories'][idx]
        if traj is None:
            break
        segments.insert(0, traj)
        idx = tree['parent'][idx]
    if len(segments) == 0:
        return np.empty((0,3)), tree['path_costs'][-1] if tree['path_costs'] else float('inf')
    full = np.vstack(segments)
    return full, tree['path_costs'][-1]

def kinodynamic_rrt(img, start, goal, max_iter, step_size, goal_radius,
                     v_max, omega_max, min_turning_radius, flag_show_runtime, robot_radius=1,
                     scale=1):
    """
    Runs RRT and visualizes with pygame.
    scale: integer pixel scaling for display (use >1 for big maps)
    """
    h, w = img.shape
    display_w = int(w * scale)
    display_h = int(h * scale)

    pygame.init()
    screen = pygame.display.set_mode((display_w, display_h))
    pygame.display.set_caption("RRT Kinodynamic")
    clock = pygame.time.Clock()

    map_surf = _to_surface_from_gray((img*255).astype(np.uint8))
    if scale != 1:
        map_surf = pygame.transform.scale(map_surf, (display_w, display_h))

    # working surface we draw on
    canvas = pygame.Surface((display_w, display_h))
    canvas.fill(BG)
    canvas.blit(map_surf, (0,0))

    # keep structures with index alignment (0 = start)
    tree = {
        'nodes': [np.array(start, dtype=float)],
        'parent': [0],
        'controls': [None],
        'trajectories': [None],
        'path_costs': [0.0]
    }

    update_interval_ms = 10  # ms
    last_update = pygame.time.get_ticks()

    # draw start/goal
    s_px = coord_to_screen(start, h); g_px = coord_to_screen(goal, h)
    if scale != 1:
        s_px = (int(s_px[0]*scale), int(s_px[1]*scale))
        g_px = (int(g_px[0]*scale), int(g_px[1]*scale))
    pygame.draw.circle(canvas, COLOR_START, s_px, max(3, int(4*scale)))
    pygame.draw.circle(canvas, COLOR_GOAL, g_px, max(3, int(4*scale)))
    pygame.draw.circle(canvas, (80,80,80), g_px, int(goal_radius*scale), 1)


    # initial blit
    screen.blit(canvas, (0,0))
    pygame.display.flip()

    running = True
    found_paths = []
    for it in range(max_iter):
        # process events to keep window responsive
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break
        if not running:
            break

        if np.random.rand() < 0.2:
            rand_point = goal
        else:
            rand_point = [np.random.randint(1, w), np.random.randint(1, h), np.random.rand() * 2 * np.pi]

        nearest_idx, nearest_node = find_nearest_node(tree['nodes'], rand_point)

        best_state = None
        best_control = None
        best_trajectory = None
        best_path_cost = float('inf')
        min_dist = float('inf')


        for _ in range(4):
            control = sample_control(v_max, omega_max, min_turning_radius)
            new_state, trajectory = generate_trajectory_euler(nearest_node, control, step_size)
            if is_trajectory_collision_free(img, trajectory, robot_radius):
                dist = compute_distance(new_state, rand_point)
                path_cost = np.linalg.norm(trajectory[1:, :2] - trajectory[:-1, :2], axis=1).sum()
                if dist < min_dist:
                    min_dist = dist
                    best_state = new_state
                    best_control = control
                    best_trajectory = trajectory
                    best_path_cost = path_cost


        if best_state is None:
            continue

        # current_idx = len(tree['nodes'])
        tree['nodes'].append(best_state)
        tree['parent'].append(int(nearest_idx))
        tree['controls'].append(best_control)
        tree['trajectories'].append(best_trajectory)
        tree['path_costs'].append(tree['path_costs'][nearest_idx] + best_path_cost)

        # draw trajectory onto canvas (scaled)
        pts = best_trajectory[:, :2]
        pts_draw = [ (int(p[0]*scale), int(p[1]*scale)) for p in pts]

        # convert to ints
        if len(pts_draw) >= 2:
            pygame.draw.lines(canvas, COLOR_TREE, False, pts_draw, max(1, int(1)))
        # update display occasionally (not every iteration)
        now = pygame.time.get_ticks()
        if flag_show_runtime and (now - last_update) >= update_interval_ms:
            screen.blit(canvas, (0,0))
            pygame.display.flip()
            last_update = now

        # check goal reach
        if np.linalg.norm(np.array(best_state[:2]) - np.array(goal[:2])) <= goal_radius:
            full_traj, final_cost = reconstruct_path(tree)
            # draw final path in different color (scaled)
            if full_traj.size > 0:
                pts = full_traj[:, :2]
                pts_draw = [(p[0]*scale, p[1]*scale) for p in pts]
                pts_draw = [(int(round(x)), int(round(y))) for x,y in pts_draw]
                if len(pts_draw) >= 2:
                    pygame.draw.lines(canvas, COLOR_PATH, False, pts_draw, max(2, int(2*scale)))
            screen.blit(canvas, (0,0))
            pygame.display.flip()
            print(f"\nFound path with cost {final_cost:.2f} at iter {it}")
            found_paths.append(full_traj)
            # keep window open until user closes or presses a key
            # waiting = True
            # while waiting and running:
            #     for ev in pygame.event.get():
            #         if ev.type == pygame.QUIT:
            #             running = False
            #             waiting = False
            #             break
            #         elif ev.type == pygame.KEYDOWN:
            #             # any key closes waiting and continues (allow more solutions)
            #             waiting = False
            #             break
            #     clock.tick(30)
            # if not running:
            #     break

    print("RRT finished")




    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
        screen.blit(canvas, (0,0))
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()
    return found_paths

if __name__ == "__main__":
    # demo usage - create a simple free map if file missing
    try:
        img = load_img("Paths\\Pictures\\map_1_d.png")
    except Exception:
        img = np.ones((300,300), dtype=np.uint8) * 255
        # add obstacle stripe
        img[140:160, 50:250] = 0


    start = [30, 50, 0.0]
    goal = [270, 200, np.pi/2]
    goal_radius = 15

    kinodynamic_rrt(img, start, goal, max_iter=200000, step_size=8,
                     goal_radius=goal_radius, v_max=15, omega_max=1.0,
                     min_turning_radius=5.0, flag_show_runtime=True,robot_radius=4, scale=3)
