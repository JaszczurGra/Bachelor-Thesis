import io
import math
from ompl import base as ob
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.ndimage import binary_dilation
import numpy as np



class BasePathfinding():
    def __init__(self,robot=None,map=None,start=(0,0,0),goal=(10,10,0),bounds=(10,10),max_runtime=30.0,goal_threshold=0.0 , solved_time = None):
        """bounds = (xmin,xmax,ymin,ymax)"""

        #TODO remove this as this is only needed for old data 
        if len(bounds) > 2:
            bounds = (10,10)
        self.solved_path = None 
        self.map = map 
        self.robot = robot if robot is not None else Robot()
        self.robot.set_map(map,bounds)
        self.start = start
        self.goal = goal
        self.bounds = bounds
        self.max_runtime = max_runtime
        self.goal_threshold = goal_threshold
        self.solved_time = solved_time 


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
        return self.robot.check_bounds(state) and not self.robot.check_collision(state)


    def visualize(self, ax=None, path_data_str=None,point_iteration=9,path_iteration=1,velocity_scale =0.2):

        '''x, y, theta, v'''
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


        ax.set_xlim(0, self.bounds[0])
        ax.set_ylim(0, self.bounds[1])
        ax.grid(True, linestyle='--', alpha=0.5)


        #gray_r for reveresed
        if self.map is not None:
            ax.imshow(self.map, extent=(0,self.bounds[0],0,self.bounds[1]), origin='lower', cmap='gray', alpha=1)


        # ax.plot(self.start_point[0], self.start_point[1], '*', color='red', markersize=15, label='Start Center')
        #Plot max velocity at end goal 
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

        ax.scatter(x_coords[points_indices], y_coords[points_indices], color='black', s=20)


        if len(data[0]) > 3:
            # velocity_scale = math.sqrt(self.bounds[0] ** 2 + self.bounds[1] ** 2) / velocity_scale
            # print(velocity_scale)
            for p in points_indices:
                state = data[p, :]
                #TODO
                if not state[3] == 0:
                    arrow_scale = state[3]  * velocity_scale
                    dx = arrow_scale * np.cos(state[2])
                    dy = arrow_scale * np.sin(state[2])
                    
                    ax.arrow(state[0], state[1], dx, dy,
                            head_width=0.2, head_length=0.08,
                            fc='red', ec='red', linewidth=1.5, alpha=0.7,label='Velocity at scale {:.2f}'.format(velocity_scale))
                    

        # for p in points_indices:
        #     self.robot.draw(ax, data[p, :])  # Draw robot at sampled points along the path
        self.robot.draw(ax, data[0, :])  # Draw robot at start
        self.robot.draw(ax, data[-1, :])  # Draw robot at goal
    

        self.robot.draw_velocity_cones(ax, data[0, :],velocity_scale) 
      

        if show:
            plt.show()


        if  len(data[0]) > 3 :
            vel  = data[:,3]
            max_vel = np.max(vel)
            return max_vel
        return 0




def get_robot(robot_data):
    if robot_data is None:
        return Robot()
    
    robot_types = {
        'Robot': Robot,
        'RectangleRobot': RectangleRobot,
        'PacejkaRectangleRobot': PacejkaRectangleRobot,
    }

    robot = robot_types.get(robot_data.get('class', 'Robot'), Robot)
    filtered_params = {k: v for k, v in robot_data.items() if k in robot.__init__.__code__.co_varnames}
    return robot(**filtered_params)


class Robot():
    def __init__(self,radius=0.2,wheelbase=1.0,max_velocity=15.0,max_steering_at_zero_v=math.pi / 4.0,max_steering_at_max_v=math.pi / 16.0, acceleration=10.0, bounds = (10,10) ,mu_static = 1.0):
        self.radius = radius
        self.wheelbase = wheelbase
        self.max_velocity = max_velocity
        self.max_steering_at_zero_v = max_steering_at_zero_v
        self.max_steering_at_max_v = max_steering_at_max_v
        self.acceleration = acceleration
        self.mu_static = mu_static 
        self._dilated_map = None
        self._bounds = bounds


    def set_map(self, map, bounds):
        if map is None:
            return
        self._bounds = bounds
        radius_px = int((self.radius / self._bounds[0]) * map.shape[0])
        y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
        circle = x**2 + y**2 <= radius_px**2
        dilated = binary_dilation(map == 0, structure=circle)
        self._dilated_map = [dilated]




    def check_bounds(self,state):
        x = state[0][0]
        y = state[0][1]
        return x >=  self.radius and x <= self._bounds[0] - self.radius and y >=  self.radius and y <= self._bounds[1] - self.radius
    

    def check_collision(self, state, a=0):
        if self._dilated_map is None:
            print('Dialtation map not set ')
            return True
        x, y = state[0][0], state[0][1]

        px = int(x / self._bounds[0] * self._dilated_map[a].shape[1])
        py = int(y / self._bounds[1] * self._dilated_map[a].shape[0])
        if 0 <= px < self._dilated_map[a].shape[1] and 0 <= py < self._dilated_map[a].shape[0]:
            # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            return self._dilated_map[a][py, px]
        return True
    
    def draw(self, ax, state):
        circle = plt.Circle((state[0], state[1]), self.radius, color='green', alpha=0.5)
        
        ax.add_patch(circle)
        #TODO show wheelbase

    def draw_velocity_cones(self, ax, state, velocity_scale=1.0):

        #TODO change cone into curves representing  steering
        x, y, theta = state[0], state[1], state[2]
        cone_length = self.max_velocity * velocity_scale
        
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
        ax.fill(cone_outer_x, cone_outer_y, color='red', alpha=0.2, edgecolor='red', linewidth=1.5,label='Velocity at scale {:.2f}'.format(velocity_scale))
        
        # Inner cone (max steering at max velocity) - darker, narrower
        angles_inner = np.linspace(right_angle_min, left_angle_min, n_points)
        inner_x = x + cone_length * 1 * np.cos(angles_inner)
        inner_y = y + cone_length * 1 * np.sin(angles_inner)
        cone_inner_x = np.concatenate([[x], inner_x, [x]])
        cone_inner_y = np.concatenate([[y], inner_y, [y]])
        ax.fill(cone_inner_x, cone_inner_y, color='red', alpha=0.4, edgecolor='red', linewidth=1.5)

    def print_info(self):
        return {'class':self.__class__.__name__} | {
                    key: value.print_info() if hasattr(value, 'print_info') and callable(getattr(value, 'print_info')) else value
                    for key, value in self.__dict__.items() 
                    if not key.startswith('_') and not callable(value) 
                } 


class RectangleRobot(Robot):
    def __init__(self, width=0.5, length=1.0 ,wheelbase=0.7,max_velocity=15.0,max_steering_at_zero_v=math.pi / 4.0,max_steering_at_max_v=math.pi / 16.0, acceleration=10.0,bounds = (10,10), collision_check_angle_res = 180,mu_static=1.0):
        "collision_check_angle_res : number of angles to check for collision "
        
        super().__init__( 0,wheelbase,max_velocity,max_steering_at_zero_v,max_steering_at_max_v,acceleration,bounds,mu_static)
        self.width = width
        self.length = length
        self.collision_check_angle_res = collision_check_angle_res

    def check_collision(self, state):
        #TODO do only second and make adiffrent robot for dubins
        #TODO we could check 2 nearest ones if it's valid and that it has to be valid in both
        angle =  state[1].value if hasattr(state[1], 'value') else state[0][2]
        angle_index = int(angle * self.collision_check_angle_res / math.pi) % self.collision_check_angle_res
        return super().check_collision(state,angle_index)

    def check_bounds(self, state):
        #AABB 
        angle =  state[1].value if hasattr(state[1], 'value') else state[0][2]
        x = state[0][0]
        y = state[0][1]
        h  = self.width /2.0 * abs(math.cos(angle)) + self.length /2.0 * abs( math.sin(angle))
        w  = self.width /2.0 * abs(math.sin(angle)) + self.length /2.0 * abs( math.cos(angle))
        return w <= x <= self._bounds[0] - w  and   h <= y <= self._bounds[1] 

    def set_map(self, map,bounds):

        if map is None:
            return
        self._bounds = bounds
        diagonal = math.sqrt((self.width/2.0)**2 + (self.length/2.0)**2)
        radius_px = int((diagonal / self._bounds[0]) * map.shape[0])


        rect_width_px = int((self.width / self._bounds[0]) * map.shape[0])
        rect_lenght_px = int((self.length / self._bounds[0]) * map.shape[0])
        self._dilated_map = []

        for i in range(self.collision_check_angle_res):
            #TODO is this rotaing to the right? as it rotates the map ?  ??
            angle = (math.pi / self.collision_check_angle_res) * i
            
            y_grid, x_grid = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
            
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            x_rot = x_grid * cos_a + y_grid * sin_a
            y_rot = -x_grid * sin_a + y_grid * cos_a
            

            rect_mask = (np.abs(y_rot) <= rect_width_px) & (np.abs(x_rot) <= rect_lenght_px)
            # np.set_printoptions(threshold=np.inf, linewidth=np.inf)
            # print(rect_mask.astype(int),end='\n\n\n')        
            self._dilated_map.append(binary_dilation(map == 0, structure=rect_mask))

        # self.radius = math.sqrt((self.width/2.0)**2 + (self.lenght/2.0)**2)
        self.radius = 0  
        # print(f'Set {self.collision_check_angle_res} dilated maps for rectangle robot collision checking.')

    def draw(self, ax, state):
        super().draw(ax, state)
        rect = plt.Rectangle((state[0] - self.length / 2.0,   state[1] - self.width / 2.0), self.length, self.width, angle=math.degrees(state[2]), color='green', alpha=0.5,rotation_point='center')
        ax.add_patch(rect)




class PacejkaTireModel():
    def __init__(self, Sx_p, Alpha_p, By, Cy, Dy, Ey, Bx, Cx, Dx, Ex) -> None:
        self.Sx_p = Sx_p
        self.Alpha_p = Alpha_p
        self.By = By
        self.Cy = Cy
        self.Dy = Dy
        self.Ey =   Ey
        self.Bx = Bx
        self.Cx = Cx
        self.Dx = Dx
        self.Ex = Ex

    def tire_forces_model(self, slip_angle_rad, slip_ratio):
        #[1] Bakker, E., Nyborg, L., and Pacejka, H.B.
        #   Tyre modelling for use in vehicle dynamics studies. United States: N. p., 1987. Web
        
        #[2] W. F. Milliken and D. L. Milliken,
        #   Race Car Vehicle Dynamics.Warrendale, PA, USA: SAE, 1995.
        
        # https://skill-lync.com/student-projects/Combined-slip-correction-using-Pacejka-tire-model-25918
        # https://www.researchgate.net/publication/344073372_Tire_Modeling_Using_Pacejka_Model
        # Unpack the state variables



        # alpha from radians to degrees
        slip_angle_deg = slip_angle_rad * 180.0 / math.pi
        
        # Calculate normalized slip ratio and slip angle
        Sx_norm = slip_ratio / self.Sx_p
        Alpha_norm = slip_angle_deg / self.Alpha_p
        
        # Compute the resultant slip
        S_resultant = math.sqrt(Sx_norm**2 + Alpha_norm**2)
        
        # Find the modified slip factors
        Sx_mod = S_resultant * self.Sx_p
        Alpha_mod = S_resultant * self.Alpha_p
        

        #TODO can't divide by 0 
        if S_resultant < 1e-6:
            print(S_resultant)

            S_resultant = 1e-6
        # Calculate the Lateral Force using Pacejka formula
        Alpha_final = Alpha_mod #+ self.Shy
        Fy = ((Alpha_norm / S_resultant) * self.Dy * math.sin(self.Cy * math.atan((self.By * Alpha_final) - 
            self.Ey * -1.0 * (self.By * Alpha_final - math.atan(self.By * Alpha_final))))) #+ self.Svy
        
        # Calculate the Longitudinal Force using Pacejka formula
        Sx_final = Sx_mod #+ self.Shx
        Fx = ((Sx_norm / S_resultant) * self.Dx * math.sin(self.Cx * math.atan((self.Bx * Sx_final) - 
            self.Ex * -1.0 * (self.Bx * Sx_final - math.atan(self.Bx * Sx_final))))) #+ self.Svx
        
        return Fx, - Fy


    def forward_front(self, robot, x):
        fric_xf, fric_yf = self.tire_forces_model(self.slip_angle_front_func(x, robot),
                                                  self.slip_ratio_front_func(x, robot))
        
        Fxf = self.Fz_front(robot) * fric_xf
        Fyf = self.Fz_front(robot) * fric_yf
        return [Fxf, Fyf]
    


    def forward_rear(self,robot, x):
        fric_xr, fric_yr = self.tire_forces_model(self.slip_angle_rear_func(x, robot),
                                                  self.slip_ratio_func(x, robot))
        
        Fxr = self.Fz_rear(robot) * fric_xr
        Fyr = self.Fz_rear(robot) * fric_yr

        return [Fxr, Fyr]
    
        
    def Fz_front(self, wp):
        return wp.m * wp.g * wp.lr / wp.L   
    
    def Fz_rear(self, wp):
        return wp.m * wp.g * PacejkaTireModel.lf(wp) / wp.L


    @staticmethod
    def lf(wp):
        return wp.L - wp.lr

    @staticmethod
    def slip_angle_front_func(wx, wp):        
        return math.atan((wx.v_y + PacejkaTireModel.lf(wp) * wx.r) / (wx.v_x + wp.eps)) - wx.delta

    @staticmethod
    def slip_angle_rear_func(wx, wp):
        return math.atan((wx.v_y - PacejkaTireModel.lf(wp) * wx.r) / (wx.v_x + wp.eps))
    @staticmethod
    def slip_ratio_func(wx, wp):
        slip_ratio = (wx.omega_wheels - wx.v_x) / \
            (wx.v_x + wp.eps)
        return slip_ratio

    @staticmethod
    def slip_ratio_front_func(wx, wp):
        v_front = wx.v_x * \
            math.cos(wx.delta) + (wx.v_y + wx.r *
                                   wp.lr) * math.sin(wx.delta)
        slip_ratio = (wx.omega_wheels - v_front) / \
            (v_front + wp.eps)
        return slip_ratio
    
    def print_info(self):
        return {'class':self.__class__.__name__} | {
                    key: value
                    for key, value in self.__dict__.items() 
                    if not key.startswith('_') and not callable(value) 
                } 
    


from dataclasses import dataclass



@dataclass
class State:
    x: float
    y: float
    yaw: float
    v_x: float
    v_y: float
    r: float
    omega_wheels: float
    delta: float 


class PacejkaRectangleRobot(RectangleRobot):
    def __init__(self, width: float=0.5, length: float=1.0, max_velocity=10,m=5.1,g=9.81,I_z=0.1435,L=0.33,lr=0.1703,Cd0=0.0008,Cd1=0.0006,Cd2=0.0003,R=0.05,tau_omega=0.022,tau_delta=0.022,eps=0.1,front_tire=PacejkaTireModel(  Sx_p= 0.1117,
  Alpha_p= 7.5586,
  By= 0.0569,
  Cy= 0.5797,
  Dy= 0.8745,
  Ey= 0.2432,
  Bx= 2.8452,
  Cx= 0.6443,
  Dx= 0.8216,
  Ex= 0.0051),rear_tire=PacejkaTireModel(Sx_p= 0.1215,
  Alpha_p= 6.9388,
  By= 0.0682,
  Cy= 0.5788,
  Dy= 0.8723,
  Ey= 4.6495,
  Bx= 4.4566,
  Cx= 1.5544,
  Dx= 0.8968,
  Ex= 8.0218)) -> None:
        super().__init__(width, length,wheelbase=L,max_velocity=max_velocity,bounds=(10,10))


        #TODO check if the tire is dict then reconstruct it 
        self.front_tire = front_tire
        self.rear_tire = rear_tire
        self.m = m
        self.g = g
        self.I_z = I_z
        self.L = L
        self.lr = lr
        self.Cd0 = Cd0
        self.Cd1 = Cd1
        self.Cd2 = Cd2
        self.R = R
        self.tau_omega = tau_omega
        self.tau_delta = tau_delta
        self.eps = eps  

        self._i = 0
    
    def forward(self, state, control, result):
        """
        control: [omega_wheels_ref, delta_ref]
        state: [x,y, yaw, v_x, v_y, r, omega_wheels, delta]
        """

        omega_wheels_ref = control[0]
        delta_ref = control[1]



        wx  = State(*state)

        
        # d = 100000
        # if self._i % d == 0: 
        #     print(self._i , '        '      ,wx)
        # self._i += 1

        Fx_f, Fy_f, Fx_r, Fy_r = self.front_tire.forward_front(self, wx) + self.rear_tire.forward_rear(self, wx)


        #TODO glowna rzecz  sily przemnozyc przez mu_static 

        F_drag = math.copysign(self.Cd0, wx.v_x) +\
            self.Cd1 * wx.v_x +\
            self.Cd2 * wx.v_x * wx.v_x
        

        v_x_dot = 1.0 / self.m * (Fx_r + Fx_f * math.cos(wx.delta) -
                               Fy_f * math.sin(wx.delta) - F_drag + self.m * wx.v_y * wx.r)

        v_y_dot = 1.0 / self.m * (Fx_f * math.sin(wx.delta) +
                               Fy_r + Fy_f * math.cos(wx.delta) - self.m * wx.v_x * wx.r)

        r_dot = 1.0 / self.I_z * \
            ((Fx_f * math.sin(wx.delta) + Fy_f *
             math.cos(wx.delta)) * (self.L - self.lr) - Fy_r * self.lr)

        omega_wheels_dot = (omega_wheels_ref - wx.omega_wheels) / self.tau_omega

        delta_dot = (delta_ref - wx.delta) / self.tau_delta

        x_dot = (wx.v_x * math.cos(wx.yaw) - wx.v_y * math.sin(wx.yaw))
        y_dot = (wx.v_x * math.sin(wx.yaw) + wx.v_y * math.cos(wx.yaw))
        yaw_dot = wx.r



        result[0] = x_dot
        result[1] = y_dot
        result[2] = yaw_dot
        result[3] = v_x_dot
        result[4] = v_y_dot
        result[5] = r_dot
        result[6] = omega_wheels_dot
        result[7] = delta_dot



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
    def __init__(self, si, goal_state, pos_threshold=0.5,velocity_threshold=0.5,bounds=(10,10),max_velocity=15.0):
        '''pos_threshold and velocity threshold absolute (not scaled by bounds and max_vel)'''
        super().__init__(si, goal_state, pos_threshold)
        self.vel_threshold = velocity_threshold 
        # self.velocity_weight =   pos_threshold /  (bounds[1]**2 + bounds[0]**2) * max_velocity / velocity_threshold    * (bounds[1]**2 + bounds[0]**2) / max_velocity  # Normalize weight based on  max velocity and scale to postion bounds 
        self.velocity_weight = pos_threshold / velocity_threshold
        #bounds are added here to normalize the position 

    #TODO is not sqrt the pos and vel dist workign properl y

    def isSatisfied(self, state):
 
        pos_dist = (state[0][0] - self.goal_state_[0])**2 + (state[0][1] - self.goal_state_[1])**2   # Normalize by max possible distance in bounds

   
        v_actual = state[2][0]
        v_goal = self.goal_state_[2]
        v_dist = abs(v_actual - v_goal)
        return pos_dist <= self.pos_threshold**2   and v_dist <= self.vel_threshold**2 # Must be stopped and close
    
    def distanceGoal(self, state):
        pos_dist = (state[0][0] - self.goal_state_[0])**2 + (state[0][1] - self.goal_state_[1])**2
        return pos_dist  + abs(state[2][0] - self.goal_state_[2]) * self.velocity_weight
