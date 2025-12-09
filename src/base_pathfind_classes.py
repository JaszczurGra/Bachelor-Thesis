import io
import math
from ompl import base as ob
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import binary_dilation
import numpy as np



class BasePathfinding():
    def __init__(self,robot=None,map=None,start=(0,0),goal=(10,10),bounds=(0,10,0,10),max_runtime=30.0,goal_treshold=0.0):
        """bounds = (xmin,xmax,ymin,ymax)"""
        self.solved_path = None 
        self.map = map 
        self.robot = robot if robot is not None else Robot()
        self.robot.set_map(map)
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.max_runtime = max_runtime
        self.goal_threshold = goal_treshold

    def print_info(self):
        data = {'class':self.__class__.__name__} | {
            key: value 
            for key, value in self.__dict__.items() 
            if not key.startswith('_') and not callable(value)
        } 
        data.pop('map', None)
        data['solved_path'] = np.loadtxt(io.StringIO(self.solved_path)).tolist()
        data['robot'] = self.robot.print_info()
        return data

    def solve(self):
        raise NotImplementedError("This method should be overridden by subclasses")
    

    def is_state_valid(self,si, state):
        return self.robot.check_bounds(state,self.bounds) and not self.robot.check_collision(state)


    def visualize(self, ax=None, path_data_str=None,point_iteration=10,path_iteration=1,quiver_iteration=10):
        show = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            show = True

        if path_data_str is None:
            if self.solved_path is None:
                return
            path_data_str = self.solved_path

        if isinstance(path_data_str, list):
            data = np.array(path_data_str)
        else:
            data = np.loadtxt(io.StringIO(path_data_str))   


        x_coords = data[:, 0]
        y_coords = data[:, 1]

        theta_angles = data[:, 2] if data.shape[1] > 2 else None


        ax.set_xlim(self.bounds[0], self.bounds[1])
        ax.set_ylim(self.bounds[2], self.bounds[3])
        ax.grid(True, linestyle='--', alpha=0.5)


        #gray_r for reveresed
        if self.map is not None:
            ax.imshow(self.map, extent=self.bounds, origin='lower', cmap='gray', alpha=1)


        # ax.plot(self.start_point[0], self.start_point[1], '*', color='red', markersize=15, label='Start Center')

        ax.plot(self.goal[0], self.goal[1], 'x', color='blue', markersize=8, label='Goal Center')
        if self.goal_threshold > 0:
            ax.add_patch(plt.Circle(self.goal, self.goal_threshold, color='blue', alpha=0.2, label='Goal Region'))


        points_indices = list(range(0, len(data) - 1, point_iteration)) + [len(data) - 1]
        path_indices = list(range(0, len(data) - 1, path_iteration)) + [len(data) - 1]

        # LineCollection for path
        path_points = np.column_stack([x_coords[path_indices], y_coords[path_indices]])
        path_segments = np.stack([path_points[:-1], path_points[1:]], axis=1)
        lc = LineCollection(path_segments, colors='black', linewidths=2, label='Planned Path')
        ax.add_collection(lc)

        ax.scatter(x_coords[points_indices], y_coords[points_indices], 
           c='black', s=16, alpha=0.6, zorder=5)
        

        if theta_angles is not None:
            quiver_indices = list(range(0, len(data) - 1, quiver_iteration)) + [len(data) - 1]
            ax.quiver(x_coords[quiver_indices], y_coords[quiver_indices], 
                    np.cos(theta_angles[quiver_indices]) , np.sin(theta_angles[quiver_indices]) ,
                    color='purple', scale=12, width=0.005, headwidth=8, label='Orientation')
        self.robot.draw(ax, data[0, :])  # Draw robot at start
        self.robot.draw_velocity_cones(ax, data[0, :])  # Draw velocity cones at start
        self.robot.draw(ax, data[-1, :])  # Draw robot at goal

        if show:
            plt.show()




def get_robot(robot_data):
    if robot_data is None:
        return Robot()
    
    robot_types = {
        'Robot': Robot,
        'RectangleRobot': RectangleRobot,
    }

    robot = robot_types.get(robot_data.get('class', 'Robot'), Robot)
    filtered_params = {k: v for k, v in robot_data.items() if k in robot.__init__.__code__.co_varnames}
    return robot(**filtered_params)


class Robot():
    def __init__(self,radius=0.2,wheelbase=1.0,max_velocity=15.0,max_steering_at_zero_v=math.pi / 4.0,max_steering_at_max_v=math.pi / 16.0, acceleration=10):
        self.radius = radius
        self.wheelbase = wheelbase
        self.max_velocity = max_velocity
        self.max_steering_at_zero_v = max_steering_at_zero_v
        self.max_steering_at_max_v = max_steering_at_max_v
        self.acceleration = acceleration
        self._dilated_map = None


    def set_map(self, map):
        if map is None:
            return
        radius_px = int((self.radius / 10) * map.shape[0])
        y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
        circle = x**2 + y**2 <= radius_px**2
        dilated = binary_dilation(map == 0, structure=circle)
        self._dilated_map = dilated

    def check_bounds(self,state,bounds):
        x = state[0][0]
        y = state[0][1]
        return x >= bounds[0] + self.radius and x <= bounds[1] - self.radius and y >= bounds[2] + self.radius and y <= bounds[3] - self.radius
    
    def check_collision(self, state):
        if self._dilated_map is None:
            return False
        
        x, y = state[0][0], state[0][1]
        px = int((x / 10) * self._dilated_map.shape[1])
        py = int((y / 10) * self._dilated_map.shape[0])
        
        if 0 <= px < self._dilated_map.shape[1] and 0 <= py < self._dilated_map.shape[0]:
            return self._dilated_map[py, px]
        return True
    
    def draw(self, ax, state):
        
        circle = plt.Circle((state[0], state[1]), self.radius, color='green', alpha=0.5)
        ax.add_patch(circle)

    def draw_velocity_cones(self, ax, state):
        x, y, theta = state[0], state[1], state[2]
        cone_length = 1.5  # Fixed visual length
        
        # Calculate cone angles
        left_angle_max = theta + self.max_steering_at_zero_v
        right_angle_max = theta - self.max_steering_at_zero_v
        left_angle_min = theta + self.max_steering_at_max_v
        right_angle_min = theta - self.max_steering_at_max_v
        
        # Generate cone outline points
        n_points = 20
        
        # Outer cone (max steering at zero velocity) - lighter, wider
        angles_outer = np.linspace(right_angle_max, left_angle_max, n_points)
        outer_x = x + cone_length * np.cos(angles_outer)
        outer_y = y + cone_length * np.sin(angles_outer)
        cone_outer_x = np.concatenate([[x], outer_x, [x]])
        cone_outer_y = np.concatenate([[y], outer_y, [y]])
        ax.fill(cone_outer_x, cone_outer_y, color='red', alpha=0.2, edgecolor='red', linewidth=1.5)
        
        # Inner cone (max steering at max velocity) - darker, narrower
        angles_inner = np.linspace(right_angle_min, left_angle_min, n_points)
        inner_x = x + cone_length * 0.9 * np.cos(angles_inner)
        inner_y = y + cone_length * 0.9 * np.sin(angles_inner)
        cone_inner_x = np.concatenate([[x], inner_x, [x]])
        cone_inner_y = np.concatenate([[y], inner_y, [y]])
        ax.fill(cone_inner_x, cone_inner_y, color='red', alpha=0.4, edgecolor='red', linewidth=1.5)

    def print_info(self):
        return {'class':self.__class__.__name__} | {
                    key: value 
                    for key, value in self.__dict__.items() 
                    if not key.startswith('_') and not callable(value)
                } 


class RectangleRobot(Robot):
    def __init__(self, width=0.5, lenght=1.0 ,max_velocity=15.0,max_steering_at_zero_v=math.pi / 4.0,max_steering_at_max_v=math.pi / 16.0, acceleration=10):
        super().__init__( 0 ,width,max_velocity,max_steering_at_zero_v,max_steering_at_max_v,acceleration)
        self.width = width
        self.lenght = lenght

    def set_map(self, map):
        self.radius = math.sqrt((self.width/2.0)**2 + (self.lenght/2.0)**2)
        super().set_map(map)

    def draw(self, ax, state):
        rect = plt.Rectangle((state[0] - self.lenght / 2.0,   state[1] - self.width / 2.0), self.lenght, self.width, angle=math.degrees(state[2]), color='green', alpha=0.5,rotation_point='center')
        ax.add_patch(rect)
        circle = plt.Circle((state[0], state[1]), self.radius, color='green', alpha=0.3)
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