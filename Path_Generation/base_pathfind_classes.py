import math
from ompl import base as ob

class BasePathfindClass:
    def __init__(self):
        pass

    def find_path(self, start, end):
        raise NotImplementedError("This method should be overridden by subclasses")


class Obstacle():
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def contains(self, x, y, r):
        return self.x_min - r <= x  <= self.x_max  + r and self.y_min -r <= y <= self.y_max + r

class Robot():
    def __init__(self,radius=0.2,wheelbase=1.0,max_velocity=15.0,max_steering_at_zero_v=math.pi / 4.0,max_steering_at_max_v=math.pi / 16.0):
        self.radius = radius
        self.wheelbase = wheelbase
        self.max_velocity = max_velocity
        self.max_steering_at_zero_v = max_steering_at_zero_v
        self.max_steering_at_max_v = max_steering_at_max_v



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