import ompl.base as ob
import ompl.control as oc
import ompl.util as ou
import numpy as np
import math
from functools import partial
from ompl import geometric as og # needed for asGeometric()
from base_pathfind_classes import Obstacle,Robot
from base_pathfind_classes import KinematicGoalRegion
import time 

import matplotlib.pyplot as plt
import numpy as np
import io


import ompl.util as ou
ou.setLogLevel(ou.LOG_NONE) 


#TODO 
#robot as squere 
#penelty for velocity in goal region
#different planners




class SSTCarOMPL_acceleration:
    def __init__(self,robot=None,Obstacles=None,start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5,max_runtime=30.0, propagate_step_size=0.02, control_duration=(3,40), selection_radius=2.0, pruning_radius=0.2):
        robot = robot if robot is not None else Robot()
        self.robot_radius = robot.radius
        self.wheelbase = robot.wheelbase
        self.max_velocity = robot.max_velocity
        self.max_steering_at_zero_v = robot.max_steering_at_zero_v
        self.max_steering_at_max_v = robot.max_steering_at_max_v
        self.obstacles = Obstacles if Obstacles is not None else []

        self.start_point = start
        self.goal_point = goal
        self.goal_threshold = goal_treshold
        self.max_runtime = max_runtime
        self.solved_path = None
        self.propagate_step_size = propagate_step_size
        self.control_duration = control_duration  # (min_steps, max_steps)
        self.selection_radius = selection_radius
        self.pruning_radius = pruning_radius



    def is_state_valid(self,si, state):
        """
        Checks if a state is valid (collision-free).
        For this simple example, we define an obstacle in the center.
        """
        x = state[0][0]
        y = state[0][1]
        
        
        if any(obs.contains(x, y,self.robot_radius) for obs in self.obstacles):
            return False
        

        bounds = si.getStateSpace().getSubspace(0).getBounds()

        if x < bounds.low[0] or x > bounds.high[0]:
            return False
        if y < bounds.low[1] or y > bounds.high[1]:
            return False


        return True
    

    def propagate(self,state, control, result):
     
        """
        State Propagator: Defines the dynamics of the car model.
        This function implements the Kinematic Car (Bicycle) Model ODEs,
        now including a velocity-dependent steering constraint.
        
        State: [x, y, theta, v]
        Control: [acceleration, steering_angle]
        
        The differential equations (ODE) are integrated over the 'duration' (dt).
        """
        # Vehicle parameters (often called 'L' for wheelbase)

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
        v_norm = np.clip(v / self.max_velocity, 0.0, 1.0)
        
        # MAX_DELTA = MAX_STEERING_AT_ZERO_V * (1 - v_norm) + MAX_STEERING_AT_MAX_V * v_norm 
        # A simplified version: linearly decreasing limit
        MAX_DELTA = self.max_steering_at_zero_v - v_norm * (self.max_steering_at_zero_v - self.max_steering_at_max_v)
        # TODO not constnat 
        MAX_DELTA = 30.0 * math.pi / 180.0
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
        theta_new = theta + (v / self.wheelbase) * math.tan(delta) #* duration
        
        # 3. Update velocity (v)
        # dv/dt = accel
        v_new = v + accel #* duration

        # Clamp velocity (minimum 0, maximum 10 m/s)
        v_new = np.clip(v_new, 0.0, self.max_velocity)
        
        # Assign the new state to the result object
    
        result[0] = x_new
        result[1] = y_new
        result[2] = theta_new
        result[3] = v_new

        return 

        v_norm = np.clip(state[3] / MAX_VELOCITY, 0.0, 1.0)
        MAX_DELTA = MAX_STEERING_AT_ZERO_V - v_norm * (MAX_STEERING_AT_ZERO_V - MAX_STEERING_AT_MAX_V)
        result[0] = state[0] +  state[3] * math.cos(state[2])
        result[1] = state[1] +  state[3] * math.sin(state[2])
        result[2] = state[2] + (state[3] / WHEELBASE) * math.tan(np.clip(control[1], -MAX_DELTA, MAX_DELTA))
        result[3] = np.clip(state[3] + control[0], 0.0, MAX_VELOCITY)



# Kinematic Limits



    # Ensure the new state adheres to the defined state bounds
 #   s_out.enforceBounds(result)

    def solve(self,max_runtime=30):
        self.max_runtime = max_runtime
        """
        Sets up the OMPL environment, defines the problem, and solves for a path.
        """
 
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
        si.setPropagationStepSize(self.propagate_step_size)
        si.setMinMaxControlDuration(self.control_duration[0], self.control_duration[1])  # Min 0.02s, Max 0.3s per control


        
        # Set the state validity checker (collision check)
        validity_checker = ob.StateValidityCheckerFn(partial(self.is_state_valid, si))
        si.setStateValidityChecker(validity_checker)


        
        # Set the dynamics (State Propagator)
        # si.setStatePropagator(lambda state, control, duration, result: propagate(state, control, duration, result))
        
        #TODO add adaptive ode solver
        ode = oc.ODE(self.propagate)
        odeSolver = oc.ODEBasicSolver(si, ode)
        propagator = oc.ODESolver.getStatePropagator(odeSolver)
        si.setStatePropagator(propagator)
    
        si.setup()


        pdef = ob.ProblemDefinition(si)


        start = ob.State(si)
        start()[0][0] = 1.0
        start()[0][1] = 1.0
        start()[1].value = math.pi / 2.0
        start()[2][0] = 0.0
        goal = ob.State(si)
        goal()[0][0] = 9.0
        goal()[0][1] = 9.0
        goal()[1].value = 0
        goal()[2][0] = 0.0


        goal_region = KinematicGoalRegion(si, goal, threshold=0.5)
        pdef.setStartAndGoalStates (start,goal)
        pdef.setGoal(goal_region)


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

        #TODO STT Parameters:
        pdef.setOptimizationObjective(MinimizeTimeObjective(si))
        pdef.setOptimizationObjective(ob.StateCostIntegralObjective(si, True))

        planner = oc.SST(si)
        planner.setProblemDefinition(pdef)
        #TODO STT Parameters:
        # 1/10 of selection radius 
        planner.setPruningRadius(self.pruning_radius)
        planner.setSelectionRadius(self.selection_radius)


        planner.setup()


        solved = planner.solve(max_runtime)
        if solved:

            path_control = pdef.getSolutionPath()
            # print ("\n\n--- SOLUTION FOUND ---")
            # print ('Path time: {:.2f} seconds'.format(path_control.length()))
            # print ('Time taken: {:.2f} seconds'.format(time.time() - self.start_time))
            # print('Path as matrix \n' + path_control.printAsMatrix())
            # print("Found solution:\n%s" % pdef.getSolutionPath().as_path().printAsMatrix())
            # print('Path as geometric \n' + path_control.asGeometric().printAsMatrix())
            # self.visualize(path_control.asGeometric().printAsMatrix())
            # fig, ax = plt.subplots(figsize=(8, 8))
            # self.visualize(path_control.asGeometric().printAsMatrix(), ax)
            # plt.show()
            self.solved_path = path_control.asGeometric().printAsMatrix()

        return solved 

    def visualize_with_labels(self, ax=None, path_data_str=None):
        pass

    def visualize(self, ax=None, path_data_str=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        if path_data_str is None:
            if self.solved_path is None:
                return
            path_data_str = self.solved_path
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

        
        # Obstacle (4.0 <= x <= 6.0 and 4.0 <= y <= 6.0)
        
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
        ax.set_xlim(0, X_MAX)
        ax.set_ylim(0, Y_MAX)
        # ax.set_aspect('equal', adjustable='box')
        # ax.set_title("OMPL Car Trajectory (X-Y Plane)")
        ax.grid(True, linestyle='--', alpha=0.5)

        # 3. Plot Environment
        
        # Obstacle
        if len(self.obstacles) > 0:
            for o in self.obstacles:
                o_x_min = o.x_min
                o_y_min = o.y_min
                o_x_max = o.x_max
                o_y_max = o.y_max
                obstacle = plt.Rectangle((o_x_min, o_y_min), o_x_max - o_x_min, o_y_max - o_y_min, 
                                        color='red', alpha=0.3, label='Obstacle')
                ax.add_patch(obstacle)


        ax.plot(self.start_point[0], self.start_point[1], 'o', color='green', markersize=10, label='Start')
        goal_circle = plt.Circle(self.goal_point, self.goal_threshold, color='blue', alpha=0.2, label='Goal Region')
        ax.add_patch(goal_circle)
        ax.plot(self.goal_point[0], self.goal_point[1], 'x', color='blue', markersize=8, label='Goal Center')
        ax.plot(x_coords, y_coords, color='black', linewidth=2, linestyle='-', label='Planned Path')
        ax.plot(x_coords, y_coords, 'o', color='black', markersize=4, alpha=0.6)

        if theta_angles is not None:

            skip = max(1, len(x_coords) // 10) 
            ax.quiver(x_coords[::skip], y_coords[::skip], 
                    np.cos(theta_angles[::skip]), np.sin(theta_angles[::skip]),
                    color='purple', scale=20, width=0.005, headwidth=5, label='Orientation')

  
        # Create legend with robot parameters
        # legend_text = (f'Radius: {self.robot_radius:.2f}m\n'
        #               f'Wheelbase: {self.wheelbase:.2f}m\n'
        #               f'Max Velocity: {self.max_velocity:.2f}m/s')
        # ax.text(0.02, -0.1, legend_text, transform=ax.transAxes, 
        #        verticalalignment='top', fontsize=9,
        #        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.8))



if __name__ == "__main__": 
    ou.setLogLevel(ou.LOG_DEBUG) 
    car_planner = SSTCarOMPL_acceleration()
    print(car_planner.solve(10))
    car_planner.visualize()
    plt.show()

