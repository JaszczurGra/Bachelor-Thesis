import io
import math
from ompl import base as ob
import matplotlib.pyplot as plt
import numpy as np

class BasePathfinding():
    def __init__(self,robot=None,Obstacles=[],start=(0,0),goal=(10,10),bounds=(0,10,0,10),max_runtime=30.0,goal_treshold=0.0):
        """bounds = (xmin,xmax,ymin,ymax)"""
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


    def visualize(self, ax=None, path_data_str=None,point_iteration=10,path_iteration=1,quiver_iteration=10):
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


        points_indices = list(range(0, len(data) - 1, point_iteration)) + [len(data) - 1]
        path_indices = list(range(0, len(data) - 1, path_iteration)) + [len(data) - 1]
        ax.plot(x_coords[path_indices], y_coords[path_indices], color='black', linewidth=2, linestyle='-', label='Planned Path')
        ax.plot(x_coords[points_indices], y_coords[points_indices], 'o', color='black', markersize=4, alpha=0.6)

        if theta_angles is not None:
            quiver_indices = list(range(0, len(data) - 1, quiver_iteration)) + [len(data) - 1]
            ax.quiver(x_coords[quiver_indices], y_coords[quiver_indices], 
                    np.cos(theta_angles[quiver_indices]) , np.sin(theta_angles[quiver_indices]) ,
                    color='purple', scale=12, width=0.005, headwidth=8, label='Orientation')
        self.robot.draw(ax, data[0, :])  # Draw robot at start
        self.robot.draw(ax, data[-1, :])  # Draw robot at goal

        if show:
            plt.show()





class Obstacle():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def contains(self, x, y, r):
        return False
    
    def draw(self, ax):
        pass


class RectangleObstacle(Obstacle):
    def __init__(self, x, y,w,h):
        super().__init__(x, y)
        self.w = w
        self.h = h

    def contains(self, x, y, r):
        return self.x - r <= x  <= self.x + self.w  + r and self.y -r <= y <= self.y + self.h + r
    
    def draw(self, ax):
        rect = plt.Rectangle((self.x, self.y), self.w, self.h, color='gray')
        ax.add_patch(rect)

class CircleObstacle(Obstacle):
    def __init__(self, x, y, radius):
        super().__init__(x, y)
        self.radius = radius

    def contains(self, x, y, r):
        return (x - self.x) ** 2 + (y - self.y) ** 2 <= (self.radius + r) ** 2
    
    def draw(self, ax):
        circle = plt.Circle((self.x, self.y), self.radius, color='gray')
        ax.add_patch(circle)


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
    
    def draw(self, ax, state):
        circle = plt.Circle((state[0], state[1]), self.radius, color='green', alpha=0.5)
        ax.add_patch(circle)



class KinematicGoalRegion(ob.Goal):
    def __init__(self, si, goal_state, pos_threshold=0.5):
        super().__init__(si)
        self.goal_state_ = goal_state
        self.pos_threshold = pos_threshold

    def isSatisfied(self, state):
        return  self.distanceGoal(state) <= self.pos_threshold
    def distanceGoal(self, state):
        return  math.sqrt((state[0][0] - self.goal_state_[0])**2 + (state[0][1] - self.goal_state_[1])**2)
    
class KinematicGoalRegionWithVelocity(KinematicGoalRegion):
    def __init__(self, si, goal_state, pos_threshold=0.5,velocity_threshold=3.0,velocity_weight=0.01):
        super().__init__(si, goal_state, pos_threshold)
        self.vel_threshold = velocity_threshold
        self.velocity_weight = velocity_weight  

    def isSatisfied(self, state):
 
        pos_dist = math.sqrt((state[0][0] - self.goal_state_[0])**2 + (state[0][1] - self.goal_state_[1])**2)

        if self.vel_threshold:
            v_actual = state[2][0]
            v_goal = self.goal_state_[2]
            v_dist = abs(v_actual - v_goal)
            return pos_dist <= self.pos_threshold   and v_dist <= self.vel_threshold # Must be stopped and close
        
        return pos_dist <= self.pos_threshold
    def distanceGoal(self, state):
        pos_dist = math.sqrt((state[0][0] - self.goal_state_[0])**2 + (state[0][1] - self.goal_state_[1])**2)
        if self.vel_threshold:
            return pos_dist  + abs(state[2][0] - self.goal_state_[2]) * self.velocity_weight
        
        return pos_dist