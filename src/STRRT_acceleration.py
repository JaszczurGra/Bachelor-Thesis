import ompl.base as ob
import ompl.control as oc
import ompl.util as ou
import math
from functools import partial
from base_pathfind_classes import KinematicGoalRegionWithVelocity, RectangleRobot, Robot, KinematicGoalRegion,BasePathfinding
import numpy as np

import ompl.util as ou
ou.setLogLevel(ou.LOG_NONE) 

import random


# class SteeringZeroControlSampler(oc.ControlSampler):
#     """Custom control sampler that zeros steering with some probability."""
    
#     def __init__(self, control_space, zero_prob=0.2):
#         super().__init__(control_space)
#         self.zero_prob = zero_prob
#         self.default_sampler = control_space.allocDefaultControlSampler()
    
#     def sample(self, control):
#         # Sample normally
#         self.default_sampler.sample(control)
    
#         control[0] = -15 if control[0] < -7.5 else 15 if control[0] > 7.5 else 0  
#         # # 20% chance to zero steering
#         # if random.random() < self.zero_prob:
#         #     control[1] = 0.0

#TODO add time cost to the result 
class SSTCarOMPL_acceleration(BasePathfinding):
    def __init__(self,robot=RectangleRobot(),map=None,start=(1.0,1.0, 0.0),goal=(9.0,9.0,0.0), bounds=(0,10,0,10),max_runtime=30.0, propagate_step_size=0.01, control_duration=None, selection_radius=None, pruning_radius=None, vel_threshold=None, pos_treshold=0.5):
        """
        set the velocity weight to 0 to ignore velocity in goal region
        min_steering angle will be set by calcuations of later force limit 
        """
        super().__init__(robot, map, start, goal,bounds,max_runtime,goal_threshold=pos_treshold) 
        self.propagate_step_size = propagate_step_size
        self.vel_threshold = vel_threshold
        self.pos_treshold = pos_treshold
        self.robot.max_steering_at_max_v = min( math.atan(self.robot.wheelbase * self.robot.mu_static * 9.81 / self.robot.max_velocity**2), self.robot.max_steering_at_zero_v) 
        self._lateral_force_min_v = math.sqrt(self.robot.wheelbase * self.robot.mu_static * 9.81 / math.tan(self.robot.max_steering_at_zero_v) ) 
    
        distance = math.sqrt(bounds[0]**2 + bounds[1]**2)
        t_manouver = 2 * (math.sqrt(self.robot.acceleration * distance) / self.robot.acceleration) if robot.max_velocity**2 / robot.acceleration > distance else (distance / self.robot.max_velocity) + (self.robot.max_velocity / self.robot.acceleration)
        therotical_max_v = min(t_manouver * self.robot.acceleration /2, self.robot.max_velocity)
        self.control_duration = control_duration if control_duration is not None else (1,int(  np.clip(t_manouver / self.propagate_step_size * 0.05,5, 40 )))
        self._debug_counter = 0 
        self.selection_radius = selection_radius if selection_radius is not None else therotical_max_v * self.propagate_step_size * self.control_duration[1] * 2.5
        self.pruning_radius = pruning_radius if pruning_radius is not None else therotical_max_v * self.propagate_step_size * self.control_duration[1] * 0.5

        print('Theoretical shortest time in straight line stopping at the end: ', t_manouver)
        print("Theoretical max velocity", min(t_manouver * self.robot.acceleration /2, self.robot.max_velocity) )
        print('Using control duration steps:', self.control_duration)




    def propagate(self,state, control, result):
     
        """
        State: [x, y, theta, v]
        Control: [acceleration, steering_angle]
        F_l= mvv/r < mu * g 
        """
        angle = math.copysign(self.robot.wheelbase * self.robot.mu_static * 9.81 / state[3]**2, control[1])  if state[3] >= self._lateral_force_min_v else math.tan(np.clip(control[1], -self.robot.max_steering_at_zero_v, self.robot.max_steering_at_zero_v))
        # self._debug_counter += 1
        # if self._debug_counter % 100 == 0:
        #     print(angle, math.atan(angle) * 180/math.pi)
            # print(self._debug_counter / 1000000 , 'MIL propagation steps')
        result[0] =  state[3] * math.cos(state[2])  
        result[1] = state[3] * math.sin(state[2])  
        # result[2] = (state[3] / self.robot.wheelbase) * math.tan(np.clip(control[1], -MAX_DELTA, MAX_DELTA)) 
        result[2] = (state[3] / self.robot.wheelbase) *  angle 
        result[3] = control[0]

    def solve(self):
        space = ob.CompoundStateSpace()

        pos_bounds = ob.RealVectorBounds(2)
        pos_bounds.setLow(0.0)
        pos_bounds.setHigh(0, self.bounds[0])
        pos_bounds.setHigh(1, self.bounds[1])
        r2 = ob.RealVectorStateSpace(2)
        r2.setBounds(pos_bounds)

        so2 = ob.SO2StateSpace()
        
        v_bounds = ob.RealVectorBounds(1)
        v_bounds.setLow(0.0)  
        v_bounds.setHigh(self.robot.max_velocity) # Max speed 10 m/s
        r1_v = ob.RealVectorStateSpace(1)
        r1_v.setBounds(v_bounds)
        
        space.addSubspace(r2, 1.0)      # x, y
        space.addSubspace(so2, 1.0)     # theta
        space.addSubspace(r1_v, 1.0)    # v
        space.lock() 

        cspace = oc.RealVectorControlSpace(space, 2)
        c_bounds = ob.RealVectorBounds(2)
        c_bounds.setLow(0, - self.robot.acceleration) 
        c_bounds.setHigh(0, self.robot.acceleration)
        c_bounds.setLow(1, -self.robot.max_steering_at_zero_v)
        c_bounds.setHigh(1, self.robot.max_steering_at_zero_v)
        
        cspace.setBounds(c_bounds)

        # self.steering_zero_prob = 0.2
        # if self.steering_zero_prob > 0:
        #     cspace.setControlSamplerAllocator(oc.ControlSamplerAllocator(   lambda cs: SteeringZeroControlSampler(cs, self.steering_zero_prob)))

        si = oc.SpaceInformation(space, cspace)
        si.setPropagationStepSize(self.propagate_step_size)
        si.setMinMaxControlDuration(self.control_duration[0], self.control_duration[1])  # Min 0.02s, Max 0.3s per control


        validity_checker = ob.StateValidityCheckerFn(partial(self.is_state_valid, si))
        si.setStateValidityChecker(validity_checker)

        si.setStateValidityCheckingResolution(min (self.robot.width, self.robot.length) * 0.2 / max ( self.bounds[0] , self.bounds[1])  if isinstance(self.robot, RectangleRobot) else self.robot.radius * 0.2 / max ( self.bounds[0] , self.bounds[1]) )
        ode = oc.ODE(self.propagate)
        # odeSolver = oc.ODEBasicSolver(si, ode)
        odeSolver = oc.ODEAdaptiveSolver(si, ode,self.propagate_step_size * 0.2 )

        propagator = oc.ODESolver.getStatePropagator(odeSolver)
        si.setStatePropagator(propagator)
    
        si.setup()


        pdef = ob.ProblemDefinition(si)


        start = ob.State(si)
        start()[0][0], start()[0][1] = self.start[:2]
        start()[1].value,start()[2][0] = (self.start[2],0.0) #(math.pi / 2.0,0.0)
        goal = ob.State(si)
        goal()[0][0], goal()[0][1] = self.goal[:2]
        goal()[1].value, goal()[2][0] = (self.goal[2], 0.0)

        goal_region = KinematicGoalRegion(si, goal, pos_threshold=self.pos_treshold)
        if self.vel_threshold is not None :
            goal_region = KinematicGoalRegionWithVelocity(si, goal, pos_threshold=self.pos_treshold,velocity_threshold=self.vel_threshold,bounds=self.bounds,max_velocity=self.robot.max_velocity) 
        pdef.setStartAndGoalStates (start,goal)
        pdef.setGoal(goal_region)

        pdef.setOptimizationObjective(ob.StateCostIntegralObjective(si, True))

        planner = oc.SST(si)
        planner.setProblemDefinition(pdef)
        planner.setPruningRadius(self.pruning_radius)
        planner.setSelectionRadius(self.selection_radius)


        planner.setup()


        solved = planner.solve(self.max_runtime)
        if solved:
            path = pdef.getSolutionPath()
            path.interpolate()
            self.solved_path = path.printAsMatrix()
            self.solved_time = path.getControlDuration(0) * path.getControlCount()
            return pdef.hasExactSolution()
        return None



if __name__ == "__main__": 
    ou.setLogLevel(ou.LOG_DEBUG) 
    map = np.ones((50,50))
    map[0,0] = 0
    map[15:,10:34] = 0    
    # map[10:13,:40] = 0
    np.set_printoptions(threshold=np.inf,linewidth=200)
    robot =RectangleRobot(1,0.5,max_velocity=15,acceleration=15,mu_static=0.5,collision_check_angle_res=30,max_steering_at_zero_v=math.pi/8)

    car_planner = SSTCarOMPL_acceleration(max_runtime=60, map=map,robot =robot,vel_threshold=200,pos_treshold=0.2,bounds=(15,15),goal=(13,10,0),start=(2,2,0))
    print('Solved', car_planner.solve())
    print('Max velocity:', car_planner.visualize() , '/', car_planner.robot.max_velocity)
    print('Time taken by path:', car_planner.solved_time)

