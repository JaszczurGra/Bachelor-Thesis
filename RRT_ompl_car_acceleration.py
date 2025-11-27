import ompl.base as ob
import ompl.control as oc
import ompl.util as ou
import numpy as np
import math
from functools import partial
from ompl import geometric as og # needed for asGeometric()


import time 
# Set up logging for OMPL
# ou.setLogLevel(ou.L_INFO)

MAX_RUNTIME = 1000.0  # seconds
ROBOT_RADIUS = 0.2  

class Obstacle():
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def contains(self, x, y, r):
        return self.x_min - r <= x  <= self.x_max  + r and self.y_min -r <= y <= self.y_max + r

OBSTACLES = []
for obs in [(4.0, 6.0, 4.0, 6.0), (2.0, 3.0, 7.0, 8.0), (7.0, 8.0, 2.0, 3.0),(6.5, 7.0, 2.0, 5.0)]:
    OBSTACLES.append(Obstacle(*obs))

def is_state_valid(si, state):
    """
    Checks if a state is valid (collision-free).
    For this simple example, we define an obstacle in the center.
    """
    # Cast the state to the specific state type we are using (CompoundState)

    x = state[0][0]
    y = state[0][1]
    
    # Simple rectangular obstacle in the middle of the planning space
    # Obstacle: 4.0 <= x <= 6.0 and 4.0 <= y <= 6.0
    # if 4.0 <= x <= 6.0 and 4.0 <= y <= 6.0:
    #     return False
    
    if any(obs.contains(x, y,ROBOT_RADIUS) for obs in OBSTACLES):
        return False
    
    # Always check bounds, although OMPL usually handles this internally
    # when sampling, it's good practice for general validity check.
    bounds = si.getStateSpace().getSubspace(0).getBounds()

    if x < bounds.low[0] or x > bounds.high[0]:
        return False
    if y < bounds.low[1] or y > bounds.high[1]:
        return False


    return True

start_time = time.time()
t = 0

def propagate(state, control, result):
    global t
    if t % 10000 == 0:
        print(f"\r {time.time() - start_time:.2f} / {MAX_RUNTIME} seconds", end="")
    t += 1
    """
    State Propagator: Defines the dynamics of the car model.
    This function implements the Kinematic Car (Bicycle) Model ODEs,
    now including a velocity-dependent steering constraint.
    
    State: [x, y, theta, v]
    Control: [acceleration, steering_angle]
    
    The differential equations (ODE) are integrated over the 'duration' (dt).
    """
    # Vehicle parameters (often called 'L' for wheelbase)
    WHEELBASE = 1.0  # meters
    
    # Kinematic Limits
    MAX_VELOCITY = 10.0
    MAX_STEERING_AT_ZERO_V = math.pi / 4.0  # 45 degrees
    MAX_STEERING_AT_MAX_V = math.pi / 16.0 # 11.25 degrees
    
    # Cast input state and control
    # s_in = state.as_compound()

    s_in = state
    u = control
    
    # Extract current state variables
    x = s_in[0]
    y = s_in[1]
    theta = s_in[2]
    v = s_in[3]
    
    # Extract control inputs
    accel = u[0]
    delta_control = u[1] # This is the desired steering angle (delta)
    
    # --- IMPLEMENT VELOCITY-DEPENDENT STEERING LIMIT ---
    
    # 1. Calculate the current maximum allowable steering angle (MAX_DELTA)
    # This uses linear interpolation: MAX_DELTA decreases as v increases from 0 to MAX_VELOCITY.
    
    # Normalized speed (clamped between 0 and 1)
    v_norm = np.clip(v / MAX_VELOCITY, 0.0, 1.0)
    
    # MAX_DELTA = MAX_STEERING_AT_ZERO_V * (1 - v_norm) + MAX_STEERING_AT_MAX_V * v_norm 
    # A simplified version: linearly decreasing limit
    MAX_DELTA = MAX_STEERING_AT_ZERO_V - v_norm * (MAX_STEERING_AT_ZERO_V - MAX_STEERING_AT_MAX_V)

    # 2. Clamp the desired steering angle (delta_control) to the calculated MAX_DELTA
    delta = np.clip(delta_control, -MAX_DELTA, MAX_DELTA)
    
    # ----------------------------------------------------
    
    # Simple Euler integration (first-order approximation of the ODEs)
    
    # 1. Update position (x, y)
    # dx/dt = v * cos(theta)
    # dy/dt = v * sin(theta)
    x_new = x + v * math.cos(theta) #* duration
    y_new = y + v * math.sin(theta) #* duration
    
    # 2. Update yaw angle (theta)
    # d_theta/dt = (v / L) * tan(delta)
    theta_new = theta + (v / WHEELBASE) * math.tan(delta) #* duration
    
    # 3. Update velocity (v)
    # dv/dt = accel
    v_new = v + accel #* duration

    # Clamp velocity (minimum 0, maximum 10 m/s)
    v_new = np.clip(v_new, 0.0, MAX_VELOCITY)
    
    # Assign the new state to the result object
 
    result[0] = x_new
    result[1] = y_new
    result[2] = theta_new
    result[3] = v_new
    
    # Ensure the new state adheres to the defined state bounds
 #   s_out.enforceBounds(result)

def solve_planning_problem():
    """
    Sets up the OMPL environment, defines the problem, and solves for a path.
    """
    print("--- OMPL Car Path Planning Simulation (V-Dependent Steering) ---")

    # 1. DEFINE THE STATE SPACE
    # State: [x, y, theta, v]
    space = ob.CompoundStateSpace()
    
    # R^2 for x and y position
    pos_bounds = ob.RealVectorBounds(2)
    pos_bounds.setLow(0.0)
    pos_bounds.setHigh(10.0)
    r2 = ob.RealVectorStateSpace(2)
    r2.setBounds(pos_bounds)
    
    # SO(2) for orientation (theta, angle wraps around)
    so2 = ob.SO2StateSpace()
    
    # R^1 for velocity (v)
    v_bounds = ob.RealVectorBounds(1)
    v_bounds.setLow(0.0)  # Cannot go backwards
    v_bounds.setHigh(10.0) # Max speed 10 m/s
    r1_v = ob.RealVectorStateSpace(1)
    r1_v.setBounds(v_bounds)
    
    space.addSubspace(r2, 1.0)      # x, y
    space.addSubspace(so2, 1.0)     # theta
    space.addSubspace(r1_v, 1.0)    # v
    space.lock() # Finalize the state space structure

    # 2. DEFINE THE CONTROL SPACE
    # Control: [acceleration, steering_angle]
    # NOTE: The bounds here define the range OMPL samples controls from.
    # The V-dependent steering constraint is enforced in the 'propagate' function.
    cspace = oc.RealVectorControlSpace(space, 2)
    
    # Control Bounds:
    c_bounds = ob.RealVectorBounds(2)
    
    # Control 1 (Acceleration/Brake): -5.0 m/s^2 (max brake) to 5.0 m/s^2 (max throttle)
    c_bounds.setLow(0, -5.0) 
    c_bounds.setHigh(0, 5.0)
    
    # Control 2 (Steering Angle): Set to the maximum possible range (pi/4)
    # The actual applied angle will be clamped lower in 'propagate' based on velocity.
    c_bounds.setLow(1, -math.pi / 4.0)
    c_bounds.setHigh(1, math.pi / 4.0)
    
    cspace.setBounds(c_bounds)


    # 3. CREATE SPACE INFORMATION (SI)
    si = oc.SpaceInformation(space, cspace)
    si.setPropagationStepSize(0.01)
    si.setMinMaxControlDuration(3, 10)  # Min 0.03s, Max 0.3s per control


    
    # Set the state validity checker (collision check)
    validity_checker = ob.StateValidityCheckerFn(partial(is_state_valid, si))
    si.setStateValidityChecker(validity_checker)


    
    # Set the dynamics (State Propagator)
    # si.setStatePropagator(lambda state, control, duration, result: propagate(state, control, duration, result))
    

    ode = oc.ODE(propagate)
    odeSolver = oc.ODEBasicSolver(si, ode)
    propagator = oc.ODESolver.getStatePropagator(odeSolver)
    si.setStatePropagator(propagator)

    # Set the resolution for propagation time (dt)
    # si.setPropagationStepSize(0.1)

    # 4. DEFINE START AND GOAL
    # Create the Problem Definition
    pdef = ob.ProblemDefinition(si)


    #TODO
    # void 	samplesPerSecond (double &uniform, double &near, double &gaussian, unsigned int attempts) const
 	# Estimate the number of samples that can be drawn per second, using the sampler returned by allocStateSampler()


    # Start State (e.g., at x=1, y=1, theta=0, v=0)
    # start = space.allocState()
    # start_4d = start.as_compound()
    # start_4d[0] = 1.0   # x
    # start_4d[1] = 1.0   # y
    # start_4d[2] = 0.0   # theta (facing +x axis)
    # start_4d[3] = 0.0   # v (stopped)
    start = ob.State(si)
    start()[0][0] = 1.0
    start()[0][1] = 1.0
    start()[1].value = 0
    start()[2][0] = 0.0

    print ("Defining the Goal State...")
    # Goal State (e.g., reach x=9, y=9, with any orientation and stopped)
    goal = ob.State(si)
    goal()[0][0] = 9.0
    goal()[0][1] = 9.0
    goal()[1].value = 0
    goal()[2][0] = 0.0
    # Define a goal region instead of a single exact state
    # This goal is to be within 0.5 units of (9.0, 9.0) and have v=0
    # The SO2 part (theta) is ignored for this specific goal
    class KinematicGoalRegion(ob.Goal):
        def __init__(self, si, goal_state, threshold=0.5):
            # super(KinematicGoalRegion, self).__init__(si)
            super().__init__(si)
            self.si_ = si
            self.goal_state_ = goal_state
            self.threshold_ = threshold

        def isSatisfied(self, state):
            # Check position proximity
            x_diff = state[0][0] - self.goal_state_[0]
            y_diff = state[0][1] - self.goal_state_[1]

            pos_dist = math.sqrt(x_diff**2 + y_diff**2)

            # Check velocity
            # v_actual = state[3]
            # v_goal = self.goal_state_[3]
            # v_dist = abs(v_actual - v_goal)

            return pos_dist <= self.threshold_ # and v_dist <= 0.1 # Must be stopped and close

        def distanceGoal(self, state):
            x_diff = state[0][0] - self.goal_state_[0]
            y_diff = state[0][1] - self.goal_state_[1]

            pos_dist = math.sqrt(x_diff**2 + y_diff**2)
            
            # v_dist = abs(state[3] - self.goal_state_[3])
            
            # Combine distances (simple sum)
            return pos_dist  # + v_dist

    goal_region = KinematicGoalRegion(si, goal, threshold=0.5)

    # pdef.setStartAndGoalStates(start, goal_region)
   # pdef.setStartAndGoalStates(start, goal_region)
    si.setup()


    pdef.setStartAndGoalStates (start,goal)
    pdef.setGoal(goal_region)
    # pdef.setOptimizationObjective(ob.StateCostIntegralObjective(si, True))

    class MinimizeTimeObjective(ob.StateCostIntegralObjective):
        def __init__(self, si):
            super().__init__(si, True) 

        def motionCost(self, s1, s2):
            """
            Returns the cost of the motion between s1 and s2.
            For a minimum time objective, the cost is the duration (time).
            
            The difference between s2 and s1 is the total time elapsed. 
            OMPL tracks the duration of the control segment that generated this motion.
            """

            return ob.Cost(1.0) 


    pdef.setOptimizationObjective(MinimizeTimeObjective(si))

    # 5. SELECT A PLANNER
    # RRT is a great choice for control-based planning

    planner = oc.SST(si)
    planner.setProblemDefinition(pdef)
    
    #STT Parameters:
    # planner.setPruningRadius(0.5)
    # planner.setSelectionRadius(2.0)


    planner.setup()

    # 6. SOLVE THE PROBLEM
    # Attempt to solve within 5 seconds
    solved = planner.solve(MAX_RUNTIME)

    if solved:
        def visualize(path_data_str):
            import matplotlib.pyplot as plt
            import numpy as np
            import io

        
            """
            Parses OMPL path data (either PathControl or PathGeometric) and plots the trajectory,
            including the defined start, goal, and obstacle.
            """
            # --- PASTE YOUR PATH DATA HERE ---
            # Use the 'Path as geometric' format as it provides more trajectory points.


            # --- ENVIRONMENT SETUP (Matches the car_planner.py file) ---
            X_MAX = 10.0
            Y_MAX = 10.0
            
            # Start and Goal points (x, y)
            START_POINT = (1.0, 1.0)
            GOAL_POINT = (9.0, 9.0)
            GOAL_THRESHOLD = 0.5
            
            # Obstacle (4.0 <= x <= 6.0 and 4.0 <= y <= 6.0)
            OBSTACLE_RECT = [(4.0, 4.0), (6.0, 6.0)] # (x_min, y_min), (x_max, y_max)
            
            # -----------------------------------------------------------

            # 1. Load Data
            data = np.loadtxt(io.StringIO(path_data_str))

            # The first two columns are always X and Y position
            x_coords = data[:, 0]
            y_coords = data[:, 1]
            
            # Check if theta and velocity are present (4 or more columns)
            if data.shape[1] >= 4:
                theta_angles = data[:, 2]
                velocities = data[:, 3]
            else:
                theta_angles = None
                velocities = None

            # 2. Plot Setup
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.set_xlim(0, X_MAX)
            ax.set_ylim(0, Y_MAX)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel("X Position (m)")
            ax.set_ylabel("Y Position (m)")
            ax.set_title("OMPL Car Trajectory (X-Y Plane)")
            ax.grid(True, linestyle='--', alpha=0.5)

            # 3. Plot Environment
            
            # Obstacle

            for o in OBSTACLES:
                o_x_min = o.x_min
                o_y_min = o.y_min
                o_x_max = o.x_max
                o_y_max = o.y_max
                obstacle = plt.Rectangle((o_x_min, o_y_min), o_x_max - o_x_min, o_y_max - o_y_min, 
                                        color='red', alpha=0.3, label='Obstacle')
                ax.add_patch(obstacle)

            # x_min, y_min = OBSTACLE_RECT[0]
            # x_max, y_max = OBSTACLE_RECT[1]
            # obstacle = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
            #                         color='red', alpha=0.3, label='Obstacle')
            ax.add_patch(obstacle)

            # Start Point
            ax.plot(START_POINT[0], START_POINT[1], 'o', color='green', markersize=10, label='Start')

            # Goal Region (Circle for visual clarity of the threshold)
            goal_circle = plt.Circle(GOAL_POINT, GOAL_THRESHOLD, color='blue', alpha=0.2, label='Goal Region')
            ax.add_patch(goal_circle)
            ax.plot(GOAL_POINT[0], GOAL_POINT[1], 'x', color='blue', markersize=8, label='Goal Center')

            # 4. Plot Path
            
            # Trajectory Line
            ax.plot(x_coords, y_coords, color='black', linewidth=2, linestyle='-', label='Planned Path')
            
            # Trajectory Points (Nodes)
            ax.plot(x_coords, y_coords, 'o', color='black', markersize=4, alpha=0.6)

            # Optional: Plot Car Orientation (Theta) using arrows
            if theta_angles is not None:
                # Plot every 3rd point for cleaner visualization
                skip = max(1, len(x_coords) // 10) 
                ax.quiver(x_coords[::skip], y_coords[::skip], 
                        np.cos(theta_angles[::skip]), np.sin(theta_angles[::skip]),
                        color='purple', scale=20, width=0.005, headwidth=5, label='Orientation')

            # 5. Show Legend and Plot
            # Add a legend, but filter out redundant labels (e.g., from multiple 'Orientation' entries)
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = dict(zip(labels, handles))
            ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper left')

            plt.show()

        path_control = pdef.getSolutionPath()
        print ("\n\n--- SOLUTION FOUND ---")
        print ('Path time: {:.2f} seconds'.format(path_control.length()))
        print ('Time taken: {:.2f} seconds'.format(time.time() - start_time))
        # print('Path as matrix \n' + path_control.printAsMatrix())
        # print("Found solution:\n%s" % pdef.getSolutionPath().as_path().printAsMatrix())
        # print('Path as geometric \n' + path_control.asGeometric().printAsMatrix())
        visualize(path_control.asGeometric().printAsMatrix())


    else:
        print("\n--- NO SOLUTION FOUND ---")




if __name__ == "__main__":
    solve_planning_problem()