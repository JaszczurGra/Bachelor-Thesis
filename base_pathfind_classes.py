import io
import math
from ompl import base as ob
import matplotlib.pyplot as plt
import numpy as np

class BasePathfinding():
    def __init__(self,robot=None,Obstacles=[],start=(0,0),goal=(10,10),bounds=(0,10,0,10),max_runtime=30.0,goal_treshold=0.0):
        self.solved_path = None 
        self.obstacles = Obstacles
        self.robot = robot if robot is not None else Robot()
        self.start_point = start
        self.goal_point = goal
        self.bounds = bounds
        self.max_runtime = max_runtime
        self.goal_threshold = goal_treshold

    def solve(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    

    def is_state_valid(self,si, state):
        x = state[0][0]
        y = state[0][1]
        return x >= self.bounds[0] and x <= self.bounds[1] and y >= self.bounds[2] and y <= self.bounds[3] and not any(obs.contains(x, y,self.robot.radius) for obs in self.obstacles)


    def visualize(self, ax=None, path_data_str=None):
        show = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            show = True

        if path_data_str is None:
            if self.solved_path is None:
                return
            path_data_str = self.solved_path

        data = np.loadtxt(io.StringIO(path_data_str))   

        x_coords = data[:, 0]
        y_coords = data[:, 1]
        theta_angles = data[:, 2] if data.shape[1] > 2 else None


        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.grid(True, linestyle='--', alpha=0.5)

        for o in self.obstacles:
            o.draw(ax)

        ax.plot(self.goal_point[0], self.goal_point[1], 'x', color='blue', markersize=8, label='Goal Center')
        if self.goal_threshold > 0:
            ax.add_patch(plt.Circle(self.goal_point, self.goal_threshold, color='blue', alpha=0.2, label='Goal Region'))
        ax.plot(x_coords, y_coords, color='black', linewidth=2, linestyle='-', label='Planned Path')
        ax.plot(x_coords, y_coords, 'o', color='black', markersize=4, alpha=0.6)

        if theta_angles is not None:
            skip = max(1, len(x_coords) // 10) 
            ax.quiver(x_coords[::skip], y_coords[::skip], 
                    np.cos(theta_angles[::skip]), np.sin(theta_angles[::skip]),
                    color='purple', scale=20, width=0.005, headwidth=5, label='Orientation')
            
        if show:
            plt.show()



class Obstacle():
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def contains(self, x, y, r):
        return self.x - r <= x  <= self.x + self.w  + r and self.y -r <= y <= self.y + self.h + r
    
    def draw(self, ax):
        rect = plt.Rectangle((self.x, self.y), self.w, self.h, color='gray')
        ax.add_patch(rect)

class Robot():
    def __init__(self,radius=0.2,wheelbase=1.0,max_velocity=15.0,max_steering_at_zero_v=math.pi / 4.0,max_steering_at_max_v=math.pi / 16.0, acceleration=10):
        self.radius = radius
        self.wheelbase = wheelbase
        self.max_velocity = max_velocity
        self.max_steering_at_zero_v = max_steering_at_zero_v
        self.max_steering_at_max_v = max_steering_at_max_v
        self.acceleration = acceleration

    def check_bounds(self,state,bounds):
        x = state[0][0]
        y = state[0][1]
        return x >= bounds[0] + self.radius and x <= bounds[1] - self.radius and y >= bounds[2] + self.radius and y <= bounds[3] - self.radius
    
    def check_collision(self,state,obstacles):
        x = state[0][0]
        y = state[0][1]
        return any(obs.contains(x, y,self.radius) for obs in obstacles)

class KinematicGoalRegion(ob.Goal):
    def __init__(self, si, goal_state, threshold=0.5):
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