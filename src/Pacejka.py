import ompl.base as ob
import ompl.control as oc
import ompl.util as ou
import math
from functools import partial
from base_pathfind_classes import KinematicGoalRegionWithVelocity, RectangleRobot, Robot, KinematicGoalRegion,BasePathfinding
import numpy as np

import ompl.util as ou

ou.setLogLevel(ou.LOG_NONE) 
# ou.setLogLevel(ou.LOG_DEBUG) 


import io 


    



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
    def __init__(self, width: float, length: float) -> None:
        super().__init__(width, length)

        self.front_tire = PacejkaTireModel(  Sx_p= 0.1117,
  Alpha_p= 7.5586,
  By= 0.0569,
  Cy= 0.5797,
  Dy= 0.8745,
  Ey= 0.2432,
  Bx= 2.8452,
  Cx= 0.6443,
  Dx= 0.8216,
  Ex= 0.0051)
        self.rear_tire = PacejkaTireModel(  Sx_p= 0.1215,
  Alpha_p= 6.9388,
  By= 0.0682,
  Cy= 0.5788,
  Dy= 0.8723,
  Ey= 4.6495,
  Bx= 4.4566,
  Cx= 1.5544,
  Dx= 0.8968,
  Ex= 8.0218)
        
        self.m = 5.1
        self.g = 9.81
        # inerata of car over z axis 
        self.I_z = 0.1435
        # distance between front and rear 
        self.L = 0.33
        # distance from ceneter of gravity to rear axis  lf calculated by L - lr 
        self.lr = 0.1703
        # air friction 
        self.Cd0 = 0.0008
        self.Cd1 =  0.0006
        self.Cd2 = 0.0003
        # mu_static: 0.8
        # I_e: 0.010634126141667366 #
        # K_fi: 0.06216941028833389 #
        # b0: 0.10694752633571625 #
        # b1: 0.24296045303344727 #
        self.R = 0.05000000074505806 # radius of wheels 
        self.tau_omega = 0.022 #? delay in roation steering
        self.tau_delta = 0.022 #? delay in speed of wheels rotation 


        self.eps = 0.1

        self.i = 0

    def forward(self, state, control, result):
        """
        control: [omega_wheels_ref, delta_ref]
        state: [x,y, yaw, v_x, v_y, r, omega_wheels, delta]
        """

        omega_wheels_ref = control[0]
        delta_ref = control[1]



        wx = {
            'x': state[0],
            'y': state[1],
            'yaw': state[2],
            'v_x': state[3],
            'v_y': state[4],
            'r': state[5],
            'omega_wheels': state[6],
            'delta': state[7],
        }

        wx  = State(**wx)


        d = 100000
        if self.i % d == 0: 
            print(self.i , '        '      ,wx)
        self.i += 1

        Fx_f, Fy_f, Fx_r, Fy_r = self.front_tire.forward_front(self, wx) + self.rear_tire.forward_rear(self, wx)


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



   

class Pacejka_pathfinding(BasePathfinding):
    def __init__(self,robot:PacejkaRectangleRobot=PacejkaRectangleRobot(.5,1),map=None,start=(1.0,1.0, 0.0),goal=(8.0,1.0,0.0), bounds=(10,10),max_runtime=30.0, propagate_step_size=0.02, control_duration=(1,10), selection_radius=None, pruning_radius=None, velocity_weight=0.0, vel_threshold=4.0, pos_treshold=0.5):
        """
        set the velocity weight to 0 to ignore velocity in goal region
        """
        super().__init__(robot, map, start, goal,bounds,max_runtime,goal_threshold=pos_treshold) 
        self.propagate_step_size = propagate_step_size
        self.control_duration = control_duration  # (min_steps, max_steps)
        self.pruning_radius = pruning_radius
        self.selection_radius = selection_radius
        self.velocity_weight = velocity_weight
        self.vel_threshold = vel_threshold
        self.pos_treshold = pos_treshold


        robot.eps = propagate_step_size


        #TODO implement this for pacajka car model 
        if selection_radius is None:    
            self.selection_radius = 300 * 0.5 * 0.2 *  2
        if pruning_radius is None:
            self.pruning_radius = 300 * 0.5 * 0.2 * 0.5





    def propagate(self,state, control, result):
        """
        control: [omega_wheels_ref, delta_ref]
        state: [x,y, yaw, v_x, v_y, r, omega_wheels, delta]
        """
        result = self.robot.forward(state, control, result)

    def solve(self):

        pos_bounds = ob.RealVectorBounds(2)
        pos_bounds.setLow(0.0)
        pos_bounds.setHigh(self.bounds[1])
        r2 = ob.RealVectorStateSpace(2)
        r2.setBounds(pos_bounds)

        so2 = ob.SO2StateSpace()


       #TODO set bounds for paramas 
        v_bounds = ob.RealVectorBounds(2)
        v_bounds.setLow (0)
        v_bounds.setHigh (15.0)
        v_state = ob.RealVectorStateSpace(2)
        v_state.setBounds(v_bounds)



        o_bounds = ob.RealVectorBounds(3)
        #TODO set bounds for paramas 
        o_bounds.setLow(-300)
        o_bounds.setHigh(300)
        other_params = ob.RealVectorStateSpace(3)
        other_params.setBounds(o_bounds)
   
   
   
        space = ob.CompoundStateSpace()
        
        space.addSubspace(r2, 1.0)      # x, y
        space.addSubspace(so2, 1.0)     # theta
        space.addSubspace(v_state,1.0)
        space.addSubspace(other_params, 1.0)    # vx, vy

        space.lock() # Finalize the state space structure

        """
        control: [omega_wheels_ref, delta_ref]
        state: [x,y, yaw, v_x, v_y, r, omega_wheels, delta]
        """

        cspace = oc.RealVectorControlSpace(space, 2)
        c_bounds = ob.RealVectorBounds(2)
        #omega wheels speed refrence
        c_bounds.setLow(0, 0.05) 
        c_bounds.setHigh(0, 300)

        #delta bounds
        c_bounds.setLow(1, -math.pi/4)
        c_bounds.setHigh(1, math.pi/4)
        cspace.setBounds(c_bounds)


        si = oc.SpaceInformation(space, cspace)
        si.setPropagationStepSize(self.propagate_step_size)
        si.setMinMaxControlDuration(self.control_duration[0], self.control_duration[1])  # Min 0.02s, Max 0.3s per control


        validity_checker = ob.StateValidityCheckerFn(partial(self.is_state_valid, si))
        si.setStateValidityChecker(validity_checker)
        si.setStateValidityCheckingResolution(0.3)

        #TODO add adaptive ode solver
        ode = oc.ODE(self.propagate)
        odeSolver = oc.ODEBasicSolver(si, ode)
        # odeSolver = oc.ODEAdaptiveSolver(si, ode,0.01)

        propagator = oc.ODESolver.getStatePropagator(odeSolver)
        si.setStatePropagator(propagator)
    
        si.setup()


        pdef = ob.ProblemDefinition(si)


        start = ob.State(si)
        start()[0][0], start()[0][1] = self.start[:2]
        start()[1].value,start()[2][0] = (self.start[2],0.0) #(math.pi / 2.0,0.0)
        start()[2][0], start()[2][1] = (0.1,0) #intial v
        start()[3][0], start()[3][1], start()[3][2] = (0.0,0.1 / 0.5,0.0) #intial r, omega wheels, delta
        goal = ob.State(si)
        goal()[0][0], goal()[0][1] = self.goal[:2]
        goal()[1].value, goal()[2][0] = (self.goal[2], 0.0)

        goal_region = KinematicGoalRegion(si, goal, pos_threshold=self.pos_treshold)
        if self.velocity_weight > 0:
            goal_region = KinematicGoalRegionWithVelocity(si, goal, pos_threshold=self.pos_treshold,velocity_threshold=self.vel_threshold,velocity_weight=self.velocity_weight,bounds=self.bounds,max_velocity=self.robot.max_velocity) 
        pdef.setStartAndGoalStates (start,goal)
        pdef.setGoal(goal_region)

        pdef.setOptimizationObjective(ob.StateCostIntegralObjective(si, True))

        planner = oc.SST(si)
        planner = oc.RRT(si)
       
        planner.setProblemDefinition(pdef)
        # planner.setPruningRadius(self.pruning_radius)
        # planner.setSelectionRadius(self.selection_radius)


        planner.setup()


        solved = planner.solve(self.max_runtime)
        if solved:
            pdef.getSolutionPath().interpolate()
            # pdef.getSolutionPath().setResolution(0.001)
            # pdef.getSolutionPath().interpolate()
            self.solved_path = pdef.getSolutionPath().printAsMatrix()
            return pdef.hasExactSolution()
        return None
    
    def visualize(self, ax=None, path_data_str=None,point_iteration=9,path_iteration=1,velocity_scale =0.2):
        data = None 
        path_data_str = self.solved_path 
        if isinstance(path_data_str, list):
            data = np.array(path_data_str)
        else:
            data = np.loadtxt(io.StringIO(path_data_str))   

        #TODO split visualiztion of velocity into separete function now its normalized 
        if data is not None: 
            velocities = np.linalg.norm(data[:, 3:5], axis=1)
            data = np.hstack((data[:, :3], velocities.reshape(-1, 1)))
        super().visualize(ax, path_data_str,point_iteration,path_iteration,velocity_scale)


        return 0

if __name__ == "__main__": 

    r = PacejkaRectangleRobot(.5,1)
    print(r.print_info())
    
    pass


    ou.setLogLevel(ou.LOG_DEBUG) 
    map = np.ones((100,100))
    map[0,0] = 0 
    # map[40:,20:50] = 0    
    car_planner = Pacejka_pathfinding(max_runtime=60, map=map,robot =PacejkaRectangleRobot(0.1,0.1),vel_threshold=0.5,velocity_weight=0)
    print('solved', car_planner.solve())
    car_planner.visualize()















