import math
import random
import copy
from collections import deque

from matplotlib import path, scale
import pygame 

# --- Configuration Constants ---
MAX_ITERS = 1500000         # Maximum number of iterations for the planner
TIME_STEP = 0.1 / 10.0          # Integration time step for dynamics propagation (s)
PROPAGATION_TIME = 1 / 10.0   # Duration of control applied in one RRT step (s)
SPARSE_RADIUS = 2.5  / 15.0  # The radius for sparsification (Witness radius)
REWIRE_RADIUS = 10   # Radius to search for neighbors to rewire
GOAL_SAMPLE_RATE = 0.1    # Probability of sampling the goal state
GOAL_TOLERANCE = 1.5      # Distance tolerance to consider the goal reached (m)
MAX_CONTROL_INPUTS = 12   # Number of random control inputs to try per extension
ENV_BOUNDS = [0, 50, 0, 50] # [min_x, max_x, min_y, max_y]
PATH_SAMPLE_RATE = 1
PATH_SAMPLE_RADIUS = 15 # Radius around best path nodes to sample from

COLOR_FREE = (255, 255, 255)
COLOR_OBS = (0, 0, 0)
COLOR_START = (0, 200, 0)
COLOR_GOAL = (200, 0, 0)
COLOR_TREE = (0, 120, 255)
COLOR_TRAJ = (0, 0, 200)
COLOR_PATH = (200, 0, 200)
COLOR_BEST_PATH = (200,50,50)
COLOR_BEST_PATH_ROUGH = (255,150,150)
BG = (50, 50, 50)


# --- 1. Car Dynamics Model ---

class CarDynamics:
    """
    Implements the simplified Kinematic Bicycle Model (Dubins Car) for
    forward propagation, which is essential for kinodynamic planning.
    
    State: [x, y, theta] (position and heading)
    Control: [velocity, steering_angle]
    """
    def __init__(self, max_v=8.0, max_steer=math.pi/20):
        self.MAX_V = max_v        # Max linear velocity (m/s)
        self.MAX_STEER = max_steer # Max steering angle (radians)
        self.L = 3.0  * 0.2           # Wheelbase of the car (m)

    def propagate(self, state, control_input, dt):
        """
        Integrates the car's state over time dt given a constant control input.
        This is the core dynamics simulation step.

        State (input): [x, y, theta]
        Control: [v, phi] (velocity, steering angle)
        """
        x, y, theta = state
        v, phi = control_input
        
        # Clamp inputs to physical limits
        v = max(0, min(v, self.MAX_V))
        phi = max(-self.MAX_STEER, min(phi, self.MAX_STEER))

        # Kinematic Bicycle Model (approximated for short duration)
        # dx/dt = v * cos(theta)
        # dy/dt = v * sin(theta)
        # dtheta/dt = (v / L) * tan(phi)
        
        x_new = x + v * math.cos(theta) * dt
        y_new = y + v * math.sin(theta) * dt
        theta_new = theta + (v / self.L) * math.tan(phi) * dt

        # Normalize theta to [-pi, pi]
        theta_new = math.atan2(math.sin(theta_new), math.cos(theta_new))
        
        return [x_new, y_new, theta_new]

    def sample_control(self):
        """Generates a random, valid control input."""
        v = random.uniform(0.1, self.MAX_V)
        phi = random.uniform(-self.MAX_STEER, self.MAX_STEER)
        return [v, phi]

# --- 2. Node and Obstacle Definitions ---

class SSTNode:
    """Represents a state in the Sparse RRT search tree."""
    def __init__(self, state, cost=0.0, parent=None, control=None, time=0.0, trajectory=[]):
        self.state = state  # [x, y, theta]
        self.cost = cost    # Total path cost (time) from root
        self.parent = parent # Parent node
        self.control = control # Control input used to reach this state
        self.time = time    # Time duration of the last control input
        self.trajectory = trajectory 
        # SST specific property:
        # The node in the tree (V) that currently serves as the best witness
        # for this region of the state space. Initially, the node is its own witness.
        self.witness_node = self 

    def get_state_coords(self):
        """Returns just the (x, y) coordinates for distance calculations."""
        return self.state[0:2]

def euclidean_distance(state1, state2):
    """Calculates the 2D Euclidean distance between two states."""
    return math.sqrt((state1[0] - state2[0])**2 + (state1[1] - state2[1])**2)

class Obstacle:
    """Simple rectangular obstacle defined by (x, y, w, h)."""
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
    
    def is_collision(self, x, y, radius):
        """Checks if a point (x, y) with safety radius collides with the obstacle."""
        # Closest point on the rectangle to the circle's center
        closest_x = max(self.x, min(x, self.x + self.w))
        closest_y = max(self.y, min(y, self.y + self.h))
        
        # Distance between the closest point and the circle's center
        dist_x = x - closest_x
        dist_y = y - closest_y
        
        # If the distance is less than the car's radius, there is a collision
        return (dist_x**2 + dist_y**2) < radius**2

# --- 3. Stable Sparse RRT (SST) Planner ---

class SSTPlanner:
    def __init__(self, start_state, goal_state, obstacles):
        self.start = SSTNode(start_state)
        self.goal = SSTNode(goal_state)
        self.nodes = [self.start] # V: The set of nodes in the sparse tree
        self.car = CarDynamics()
        self.obstacles = obstacles
        self.car_radius = 1.5 # Safety/collision radius for the car
        self.best_solution_node = None
        self.best_cost = float('inf')

    def _is_valid_state(self, state):
        """Checks if a state is within bounds and obstacle-free."""
        x, y, _ = state
        # 1. Check bounds
        if not (ENV_BOUNDS[0] <= x <= ENV_BOUNDS[1] and 
                ENV_BOUNDS[2] <= y <= ENV_BOUNDS[3]):
            return False
        # 2. Check collision
        for obs in self.obstacles:
            if obs.is_collision(x, y, self.car_radius):
                return False
        return True

    def _check_trajectory_collision(self, start_state, control, duration, step=TIME_STEP):
        """Simulates the entire trajectory and checks for continuous collision."""
        current_state = list(start_state)
        num_steps = int(duration / step)
        path = [current_state]

        for _ in range(num_steps):
            current_state = self.car.propagate(current_state, control, step)
            path.append(current_state)
            if not self._is_valid_state(current_state):
                return False, [] # Collision detected
        
        return True, path

    def _find_nearest_node(self, state):
        """Finds the nearest node in the current sparse tree (V) to the given state."""
        min_dist = float('inf')
        nearest_node = None
        
        for node in self.nodes:
            # We use the cost-to-come of the witness node for comparison (SST rule)
            witness = node.witness_node
            dist = euclidean_distance(witness.state, state)
            
            if dist < min_dist:
                min_dist = dist
                nearest_node = node
        
        return nearest_node

    def _get_best_propagated_state(self, x_near):
        """
        Tries multiple random controls and finds the one that yields the best
        node (lowest cost) after propagation.
        """
        best_new_node = None
        
        for _ in range(MAX_CONTROL_INPUTS):
            control = self.car.sample_control()
            
            # Check the entire trajectory for collision
            is_valid, trajectory = self._check_trajectory_collision(
                x_near.state, control, PROPAGATION_TIME
            )
            
            if is_valid:
                # The final state of the propagation
                x_new_state = trajectory[-1]
                
                # Cost is simply the time taken (PROPAGATION_TIME)
                new_cost = x_near.cost + PROPAGATION_TIME
                
                new_node = SSTNode(
                    state=x_new_state,
                    cost=new_cost,
                    parent=x_near,
                    control=control,
                    time=PROPAGATION_TIME
                )
                
                # We need to find the best possible extension
                if best_new_node is None or new_node.cost < best_new_node.cost:
                    best_new_node = new_node
        
        return best_new_node

    def _perform_optimality_check_and_rewire(self, x_new):
        """
        NEW REWIRING LOGIC: Checks neighbors of x_new and attempts to find a
        better path to them by rewiring their parent to x_new.
        """
        neighbors_to_check = []
        
        # 1. Find neighbors within the REWIRE_RADIUS
        for node in self.nodes:
            if node == x_new.parent or node == x_new:
                continue

            dist = euclidean_distance(node.state, x_new.state)
            if dist < REWIRE_RADIUS:
                neighbors_to_check.append(node)

        # 2. Attempt to rewire neighbors
        for x_neighbor in neighbors_to_check:
            # We must check if connecting x_new to x_neighbor is kinodynamically feasible
            # and results in a lower cost for x_neighbor.

            # Determine the control/time needed to drive from x_new to x_neighbor
            # NOTE: For true kinodynamic RRT*, this step requires a costly BVP (Boundary Value Problem) solver.
            # To simplify and keep the approach feasible, we'll use a heuristic:
            # We assume a fixed-time propagation *from* x_new *can* reach x_neighbor's state,
            # and we check if the cost of that hypothetical path is better.

            # Heuristic Cost Check:
            # Check if the path through x_new is cheaper than x_neighbor's current cost
            # Since we use fixed PROPAGATION_TIME, the incremental cost is simply PROPAGATION_TIME.
            new_cost_to_neighbor = x_new.cost + PROPAGATION_TIME 
            
            if new_cost_to_neighbor < x_neighbor.cost:
                # Potential Improvement found! Now, verify feasibility (The BVP step)
                # Since BVP is too complex for this format, we use an approximation:
                # If the distance is small enough, we assume feasibility and check trajectory.
                
                # Use x_new's current control sampling mechanism as an approximation for BVP
                best_rewire_prop = None
                best_rewire_cost = float('inf')

                # Try to propagate from x_new and see if we end up near x_neighbor
                # This is a very rough approximation, but necessary without a full BVP solver.
                for _ in range(5): # Try 5 random controls from x_new
                    control = self.car.sample_control()
                    is_valid, trajectory = self._check_trajectory_collision(
                        x_new.state, control, PROPAGATION_TIME
                    )
                    
                    if is_valid:
                        final_state = trajectory[-1]
                        
                        # If the final state is close to x_neighbor AND the path is cheaper
                        if euclidean_distance(final_state, x_neighbor.state) < GOAL_TOLERANCE * 0.5:
                            
                            temp_node = SSTNode(
                                state=final_state,
                                cost=x_new.cost + PROPAGATION_TIME,
                                parent=x_new,
                                control=control,
                                time=PROPAGATION_TIME,
                                trajectory=trajectory
                            )
                            
                            if temp_node.cost < best_rewire_cost:
                                best_rewire_prop = temp_node
                                best_rewire_cost = temp_node.cost

                # If a good, cheaper connection was approximated, perform the rewire
                if best_rewire_prop is not None and best_rewire_prop.cost < x_neighbor.cost:
                    # REWIRE!
                    x_neighbor.parent = x_new
                    x_neighbor.control = best_rewire_prop.control
                    x_neighbor.time = best_rewire_prop.time
                    x_neighbor.cost = best_rewire_prop.cost
                    x_neighbor.trajectory = best_rewire_prop.trajectory
                    
                    # Recursively update the costs of all children of the rewired node
                    self._update_children_costs(x_neighbor)
                    
                    # Update witness for this region (crucial for SST)
                    self._update_witness(x_neighbor, x_neighbor)

    def _update_children_costs(self, parent_node):
        """Recursively update the costs of all descendants after a rewire."""
        queue = deque([parent_node])
        while queue:
            node = queue.popleft()
            
            # Find direct children of the node
            children = [n for n in self.nodes if n.parent == node]
            
            for child in children:
                # Calculate new cost for the child
                child.cost = node.cost + child.time
                
                # Update witness
                self._update_witness(child, child)
                
                # Add child to the queue for recursive update
                queue.append(child)


    def _sparsification_check(self, x_new):
        """
        The core SST logic. Determines if x_new should be added to the tree (V)
        and if it should update any witnesses.
        """
        # Find the witness node closest to x_new in the tree V
        min_dist = float('inf')
        nearest_witness = None
        
        # Search the entire tree V (self.nodes)
        for node in self.nodes:
            dist = euclidean_distance(node.state, x_new.state)
            if dist < min_dist:
                min_dist = dist
                nearest_witness = node
        
        # --- Sparsification Rule ---
        
        if nearest_witness is not None:
            # Check 1: Is x_new far enough away from the nearest node?
            if min_dist > SPARSE_RADIUS:
                # Case 1: x_new is far. It should be added.
                return True, nearest_witness
            
            # Check 2: If x_new is close, is it a better path to the witness region?
            else:
                # Use the cost of the nearest node *as its own witness cost* for comparison
                if x_new.cost < nearest_witness.witness_node.cost:
                    # Case 2: x_new is close but offers a better cost path.
                    return True, nearest_witness
        
        # If no witness/node was found (shouldn't happen after the root), or neither rule passed
        return False, None

    def _update_witness(self, x_new, nearest_witness):
        """
        If x_new is better than the existing witness in its region, update the witness.
        """
        # In this implementation, the actual tree nodes (self.nodes) *are* the witnesses.
        # We check if x_new is better than the best node in its neighborhood.
        
        if x_new.cost < nearest_witness.witness_node.cost:
            # Re-assign the best node for the nearest_witness's region
            nearest_witness.witness_node = x_new 
            return True
        return False

    def _get_current_path_nodes(self):
        """Helper to get a list of nodes forming the best path."""
        if not self.best_solution_node: return []
        path_nodes = []
        current = self.best_solution_node
        while current:
            path_nodes.append(current)
            current = current.parent
        return path_nodes[::-1] 


    def plan(self, animation = True):
        """The main SST planning loop."""
        print(f"Starting SST Planning (Max Iterations: {MAX_ITERS})")
        


        scale = 20 

        if animation:
            # h, w = img.shape
            display_w = abs(ENV_BOUNDS [1] - ENV_BOUNDS[0]) * scale
            display_h = abs(ENV_BOUNDS [3] - ENV_BOUNDS[2]) * scale
            pygame.init()
            screen = pygame.display.set_mode((display_w, display_h))
            pygame.display.set_caption("SST Planner")
            clock = pygame.time.Clock()
            map_surf = pygame.Surface((display_w, display_h))
            map_surf.fill(BG)
            for obs in self.obstacles:
                obs_rect = pygame.Rect(
                    int(obs.x * scale),
                    int(obs.y * scale),
                    int(obs.w * scale),
                    int(obs.h * scale)
                )
                pygame.draw.rect(map_surf, COLOR_OBS, obs_rect)

            

            map_surf = pygame.transform.scale(map_surf, (display_w, display_h))

            pygame.draw.circle(map_surf, COLOR_START, (int(self.start.state[0] * scale), int(self.start.state[1] * scale)), 6)
            pygame.draw.circle(map_surf, COLOR_GOAL, (int(self.goal.state[0] * scale), int(self.goal.state[1] * scale)), 6)


            tree_surf = pygame.Surface((display_w, display_h), pygame.SRCALPHA)
            tree_surf.fill((0,0,0,0))
            # tree_surf.blit(map_surf, (0, 0))
            path_surf = pygame.Surface((display_w, display_h), pygame.SRCALPHA)
            path_surf.fill((0, 0, 0, 0))
            update_interval_ms = 50
            last_update = pygame.time.get_ticks()
            print('Animation started.')

            screen.blit(map_surf, (0, 0))
            pygame.display.flip()

        for i in range(MAX_ITERS):
            if animation:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return self._reconstruct_path()

                current_time = pygame.time.get_ticks()
                if current_time - last_update >= update_interval_ms:
                    last_update = current_time
                    screen.blit(map_surf, (0, 0))
                    tree_surf.fill((0, 0, 0, 0))
                    for node in self.nodes:
                        if node.parent is not None:
                            start_pos = (int(node.parent.state[0] * scale), int(node.parent.state[1] * scale))
                            end_pos = (int(node.state[0] * scale), int(node.state[1] * scale))
                            pygame.draw.line(tree_surf, COLOR_TREE, start_pos, end_pos, 1)

                    screen.blit(tree_surf, (0, 0))
                    screen.blit(path_surf, (0, 0))
                    pygame.display.flip()
   
   
            if i % 1000 == 0:
                print(f"Iteration: {i}. Tree Size: {len(self.nodes)}. Best Cost: {self.best_cost:.2f}")

            # 1. Sample State
            if random.random() < GOAL_SAMPLE_RATE:
                # 1. High priority: Sample the Goal
                x_rand_state = self.goal.state
            elif self.best_solution_node is not None and random.random() < PATH_SAMPLE_RATE:
                # 2. Informed: Sample near a node on the current best path
                path_nodes = self._get_current_path_nodes()
                x_focus_node = random.choice(path_nodes)
                
                # Sample within a small radius around the chosen node
                x_rand_state = [
                    random.uniform(x_focus_node.state[0] - PATH_SAMPLE_RADIUS, x_focus_node.state[0] + PATH_SAMPLE_RADIUS),
                    random.uniform(x_focus_node.state[1] - PATH_SAMPLE_RADIUS, x_focus_node.state[1] + PATH_SAMPLE_RADIUS),
                    random.uniform(-math.pi, math.pi) # Full orientation exploration
                ]
                # Ensure the sample stays within the environment bounds
                x_rand_state[0] = max(ENV_BOUNDS[0], min(ENV_BOUNDS[1], x_rand_state[0]))
                x_rand_state[1] = max(ENV_BOUNDS[2], min(ENV_BOUNDS[3], x_rand_state[1]))
            else:
                # 3. Global: Sample across the entire environment
                x_rand_state = [
                    random.uniform(ENV_BOUNDS[0], ENV_BOUNDS[1]),
                    random.uniform(ENV_BOUNDS[2], ENV_BOUNDS[3]),
                    random.uniform(-math.pi, math.pi)
                ]

            # 2. Find Nearest Neighbor (x_near) in the current tree V
            x_near = self._find_nearest_node(x_rand_state)
            if x_near is None: continue

            # 3. Propagate and Find Best New State (x_new)
            x_new = self._get_best_propagated_state(x_near)
            if x_new is None: continue # Could not find a valid extension

            # 4. Sparsification Check
            # Check if x_new should be added to V (sparseness)
            can_add, nearest_witness_in_V = self._sparsification_check(x_new)
            
            if can_add:
                # Add x_new to the sparse tree V
                self.nodes.append(x_new)
                
                # Update the witness pointer in the nearest neighbor node to x_new
                # This ensures that for the region around nearest_witness_in_V, 
                # future expansions will be based on x_new's improved cost (if applicable).
                self._update_witness(x_new, nearest_witness_in_V)
                
                # TODO do it sometimes to rewire the tree
                if self.best_solution_node is not None:
                    self._perform_optimality_check_and_rewire(x_new)

                # 5. Check for Goal Connection (x_new is now in V)
                dist_to_goal = euclidean_distance(x_new.state, self.goal.state)
                
                if dist_to_goal < GOAL_TOLERANCE:
                    # Found a potentially new solution. Check if it's better than the current best.
                    potential_cost = x_new.cost + dist_to_goal # Estimate final cost
                    
                    if potential_cost < self.best_cost:
                        self.best_cost = potential_cost
                        self.best_solution_node = x_new
                        print(f"--- New Best Path Found! Cost: {self.best_cost:.2f} ---")
                        
                        # Draw the best path found so far
                        if animation:
                            path_surf.fill((0, 0, 0, 0))
                            smooth_animation = True

                            node_states, sim_states = self.reconstruct_simulated_trajectory(dt=0.02)
                            smoothed = self.smooth_simulated_positions(sim_states, iterations=3)

                            # simple pygame draw (if you use the same scale as planner)
                            for i in range(len(smoothed)-1):
                                a = (int(smoothed[i][0]*scale), int(smoothed[i][1]*scale))
                                b = (int(smoothed[i+1][0]*scale), int(smoothed[i+1][1]*scale))
                                pygame.draw.line(path_surf, COLOR_BEST_PATH_ROUGH, a, b, 3)


                            path = self._reconstruct_path()
                            for j in range(len(path) - 1):
                                start_pos = (int(path[j][0] * scale), int(path[j][1] * scale))
                                end_pos = (int(path[j+1][0] * scale), int(path[j+1][1] * scale))
                                pygame.draw.line(path_surf, COLOR_BEST_PATH, start_pos, end_pos, 4)
        
            elif nearest_witness_in_V is not None:
                # x_new was not added to V (too close), but check if it improves the existing witness's cost.
                self._update_witness(x_new, nearest_witness_in_V)
                
        # After loop, return the best path found
        return self._reconstruct_path()

    def _reconstruct_path(self):
        """Reconstructs the path from the goal node back to the start."""
        if self.best_solution_node is None:
            return []
            
        path = []
        current = self.best_solution_node
        while current is not None:
            path.append(current.state)
            current = current.parent
        
        # Path is built backwards, so reverse it
        return path[::-1]
    
    def reconstruct_simulated_trajectory(self, dt=TIME_STEP):
        """
        Reconstruct a dense simulated trajectory by replaying stored controls
        along the found solution (best_solution_node). Returns (node_states, sim_states)
        - node_states: list of sparse states at node positions (start..goal)
        - sim_states: list of simulated states sampled every ~dt (dense)
        """
        if self.best_solution_node is None:
            return [], []

        # collect nodes from start -> goal
        nodes = []
        cur = self.best_solution_node
        while cur is not None:
            nodes.append(cur)
            cur = cur.parent
        nodes.reverse()  # now start -> goal

        # replay controls from each child node (each node stores the control used to reach it)
        sim_states = []
        state = list(nodes[0].state)  # start state
        sim_states.append(state.copy())

        for node in nodes[1:]:
            control = node.control
            duration = node.time or PROPAGATION_TIME
            # if no control recorded, just snap to the node state
            if control is None:
                state = list(node.state)
                sim_states.append(state.copy())
                continue    
            steps = max(1, int(max(1e-6, duration) / dt))
            step_dt = duration / steps
            for _ in range(steps):
                state = self.car.propagate(state, control, step_dt)
                sim_states.append(state.copy())

            # ensure final state matches stored node state (numerical drift correction)
            state = list(node.state)
            sim_states[-1] = state.copy()
        node_states = [n.state for n in nodes]
        return node_states, sim_states
    def smooth_simulated_positions(self, sim_states, iterations=2):
        """
            Simple Chaikin corner-cutting smoothing applied to XY positions of sim_states.
        Returns smoothed sim_states with recomputed theta.
        """
        if not sim_states:
            return []
        pts = [(s[0], s[1]) for s in sim_states]
        for _ in range(iterations):
            if len(pts) < 2:
                break
            new_pts = [pts[0]]
            for i in range(len(pts) - 1):
                x0, y0 = pts[i]; x1, y1 = pts[i+1]
                q = (0.75 * x0 + 0.25 * x1, 0.75 * y0 + 0.25 * y1)
                r = (0.25 * x0 + 0.75 * x1, 0.25 * y0 + 0.75 * y1)
                new_pts.append(q); new_pts.append(r)
            new_pts.append(pts[-1])
            pts = new_pts

        # rebuild states with approximated heading (theta) from adjacent points
        out = []
        for i, (x, y) in enumerate(pts):
            if i < len(pts) - 1:
                nx, ny = pts[i+1]
                theta = math.atan2(ny - y, nx - x)
            else:
                # last: copy previous heading or 0
                if i > 0:
                    px, py = pts[i-1]
                    theta = math.atan2(y - py, x - px)
                else:
                    theta = 0.0
            out.append([x, y, theta])
        return out
# --- 4. Main Execution ---

def main():
    # --- Environment Setup ---
    # Start: (x, y, theta)
    start_state = [5.0, 5.0, math.pi / 2] 
    # Goal: (x, y, theta) - orientation often doesn't matter much in kinodynamic RRT
    goal_state = [45.0, 45.0, 0.0]

    # Obstacles: (x, y, width, height)
    obstacles = [
        Obstacle(15, 10, 5, 25),  # Vertical wall
        Obstacle(30, 15, 15, 5),  # Horizontal wall
        Obstacle(20, 30, 5, 15),  # Another vertical barrier
    ]

    obstacles = [
        Obstacle(random.uniform(10,40), random.uniform(10,40), random.uniform(3,7), random.uniform(3,7)
                 ) for _ in range(15)
    ]

    print(f"Start: {start_state[0]:.1f}, {start_state[1]:.1f} | Goal: {goal_state[0]:.1f}, {goal_state[1]:.1f}")

    planner = SSTPlanner(start_state, goal_state, obstacles)
    final_path = planner.plan()

    # --- Results ---
    if final_path:
        print("\n✅ Planning Successful!")
        print(f"Path Length (Number of States): {len(final_path)}")
        print(f"Optimal Path Cost (Time): {planner.best_cost:.2f} seconds (approx.)")
        
        # Simple visualization of the path (for console output)
        print("\nPath States (Start to Goal):")
        for i, state in enumerate(final_path):
            if i % 10 == 0 or i == len(final_path) - 1:
                print(f"  [{i:4d}] X: {state[0]:.2f}, Y: {state[1]:.2f}, Theta: {state[2]:.2f} rad")
    else:
        print("\n❌ Planning Failed. Could not find a path within the maximum iterations.")

if __name__ == "__main__":
    main()