import math
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

