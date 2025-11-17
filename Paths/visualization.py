#RRT dubins for nice path 
#RRT* for what? 
#RRT Reeds sheep for strange curves 


#TODO temporary fix for matplotlib version problem
import matplotlib
matplotlib.use("TkAgg", force=True)  # or "TkAgg" if tkinter is available

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_environment(img, start, goal, goal_radius,ax=None):
    plt.imshow(img, cmap='gray', origin='lower')
    plt.plot(start[0], start[1], 'go', markersize=10, label='Start')
    plt.plot(goal[0], goal[1], 'ro', markersize=10, label='Goal')
    plot_circle(goal[:2], goal_radius)
    plt.title('RRT Kinodynamic Path Planning')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.axis('equal')
    plt.grid()

def plot_tree(tree,ax=None):
    for i in range(1, len(tree['nodes'])):
        parent_idx = tree['parent'][i]
        plt.plot([tree['nodes'][parent_idx][0], tree['nodes'][i][0]],
                 [tree['nodes'][parent_idx][1], tree['nodes'][i][1]], 'b-', linewidth=0.5)

def plot_path(path,cost,ax=None):
    plt.plot(path[:, 0], path[:, 1], 'm-', linewidth=3, label=f'Path: {cost}')
    plt.legend()


# def visualize_rrt(img, start, goal, goal_radius, tree, path=None):
#     plt.figure(figsize=(10, 10))
#     plot_environment(img, start, goal, goal_radius)
#     plot_tree(tree)
#     if path is not None:
#         plot_path(path)
#     plt.show()

def load_img(path):
    img = Image.open(path).convert("L")   # "L" = 8-bit grayscale
    arr = np.array(img)                   # shape (H, W), dtype=uint8
    # optional normalize to [0,1]
    arr_f = arr.astype(float) / 255.0
    return arr_f



def plot_circle(center, radius,ax=None):
    theta = np.linspace(0, 2 * np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    plt.plot(x, y, 'k-', linewidth=1)

def is_trajectory_collision_free(img, trajectory, robot_radius):
    valid = True
    img_height, img_width = img.shape
    
    for point in trajectory:
        x = round(point[0])
        y = round(point[1])
        
        if x < 0 or x >= img_width or y < 0 or y >= img_height:
            valid = False
            break
        
        for dx in range(-robot_radius, robot_radius + 1):
            for dy in range(-robot_radius, robot_radius + 1):
                if dx**2 + dy**2 <= robot_radius**2:
                    check_x = round(x + dx)
                    check_y = round(y + dy)
                    
                    if 0 <= check_x < img_width and 0 <= check_y < img_height:
                        if img[check_y, check_x] == 0:  # Obstacle detected
                            valid = False
                            return valid
    return valid

def compute_distance(state1, state2):
    pos_diff = np.linalg.norm(state1[:2] - state2[:2])
    ang_diff = abs(angle_diff(state1[2], state2[2]))
    return pos_diff + 0.1 * ang_diff

def angle_diff(a1, a2):
    return (a1 - a2 + np.pi) % (2 * np.pi) - np.pi

def find_nearest_node(nodes, point):
    distances = np.linalg.norm(nodes[:, :2] - point[:2], axis=1)
    nearest_idx = np.argmin(distances)
    return nearest_idx, nodes[nearest_idx]

def sample_control(v_max, omega_max, min_turning_radius):
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
    
    current_state = state.copy()
    v, omega = control
    
    if abs(omega) < 0.07:
        omega = 0
    
    for i in range(num_steps):
        x_dot = v * np.cos(current_state[2])
        y_dot = v * np.sin(current_state[2])
        theta_dot = omega
        
        current_state[0] += x_dot * dt
        current_state[1] += y_dot * dt
        current_state[2] += theta_dot * dt
        
        current_state[2] = (current_state[2] + np.pi) % (2 * np.pi) - np.pi
        
        trajectory[i + 1, :] = current_state
    
    return current_state, trajectory

def reconstruct_path(tree):
    idx = len(tree['nodes']) - 1

    if len(tree['nodes']) <= idx or len(tree['parent']) <= idx or len(tree['trajectories']) <= idx:
        return [], float('inf')

    path = [tree['nodes'][idx]]
    full_trajectory = tree['trajectories'][idx]
    total_cost = tree['path_costs'][idx]
    
    while tree['parent'][idx] != 0:
        prev_idx = tree['parent'][idx]
        trajectory = tree['trajectories'][idx]
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'm-', linewidth=3)
        path.insert(0, tree['nodes'][prev_idx])
        full_trajectory = np.vstack((tree['trajectories'][prev_idx], full_trajectory))
        idx = prev_idx
    
    return full_trajectory, total_cost

def kinodynamic_rrt(img, start, goal, max_iter, step_size, goal_radius, v_max, omega_max, min_turning_radius, flag_show_runtime):
   
    tree = {
        'nodes': [start],
        'parent': [0],
        'cost': [0],
        'controls': [None],
        'trajectories': [None],
        'path_costs': [0]
    }
    
    robot_radius = 0
    img_height, img_width = img.shape
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # fig.ion()
    plt.ion()
    plot_environment(img, start, goal, goal_radius, ax=ax)
    fig.canvas.draw_idle()
    plt.pause(0.1)


    paths = []

    print()
    for iter in range(max_iter):
        if np.random.rand() < 0.2:
            rand_point = goal
        else:
            rand_point = [np.random.randint(1, img_width), np.random.randint(1, img_height), np.random.rand() * 2 * np.pi]
        



        nearest_idx, nearest_node = find_nearest_node(np.array(tree['nodes']), rand_point)
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
        
        current_idx = len(tree['nodes'])
        tree['nodes'].append(best_state)
        tree['parent'].append(nearest_idx)
        tree['controls'].append(best_control)
        tree['path_costs'].append(tree['path_costs'][nearest_idx] + best_path_cost)
        tree['trajectories'].append(best_trajectory)
        

        if iter % 100 == 0:
            print(f"\r{iter}/{max_iter}, size of tree {len(tree['nodes'])}",end="")
            # plot_tree(tree)

        if flag_show_runtime and iter  % 1000 == 0:
            plt.plot(best_trajectory[:, 0], best_trajectory[:, 1], 'b-', linewidth=2)
            plt.pause(0.001)
        

        if np.linalg.norm(best_state[:2] - goal[:2]) <= goal_radius:
            
            path, final_cost = reconstruct_path(tree)
           
            
            print(path)
            print(f'Found path with cost: {final_cost:.2f}')
            # plot_tree(tree)
            plot_path(path,final_cost)
            plt.pause(0.1)
            paths.append(path)

            # return  path
    
    print('Warning: Path not found within maximum iterations')
    # plt.ioff()
    return []





if __name__ == "__main__":
    # Example usage with dummy data
    img = load_img("Paths\\Pictures\\map_1_d.png")
    # img = np.ones((300,300) , dtype=np.float16) 
    img[0][0] = 0 
    img[-1][-1] = 0
    start = [30, 50, 0]  # Starting state [x, y, theta]
    goal = [270, 200, np.pi/2]  # Goal state [x, y, theta]
    goal_radius = 15

    #30 mil 
    path = kinodynamic_rrt(img,start,goal,1000000,10,goal_radius,15,1.0,5.0,False)
    while True:
        plt.pause(0.1)
    # print("Tree",tree)
    # print("Path ", path)
    # # Dummy tree data

    # # Dummy path data
    # # path = np.array([[50, 50], [120, 100], [150, 150], [200, 200], [250, 250]])
    # print(path)
    # if len(path) > 0:
    #     visualize_rrt(img, start, goal, goal_radius, tree, path)
    