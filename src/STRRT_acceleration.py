import ompl.base as ob
import ompl.control as oc
import ompl.util as ou
import math
from functools import partial
from base_pathfind_classes import KinematicGoalRegionWithVelocity, RectangleRobot, Robot, KinematicGoalRegion,BasePathfinding
import numpy as np

import ompl.util as ou
ou.setLogLevel(ou.LOG_NONE) 


#TODO 
#robot as squere 
#penelty for velocity in goal region
#different planners


#TODO add time cost to the result 
class SSTCarOMPL_acceleration(BasePathfinding):
    def __init__(self,robot=RectangleRobot(),map=None,start=(1.0,1.0, 0.0),goal=(9.0,9.0,0.0), bounds=(0,10,0,10),max_runtime=30.0, propagate_step_size=0.02, control_duration=(1,10), selection_radius=None, pruning_radius=None, vel_threshold=None, pos_treshold=0.5):
        """
        set the velocity weight to 0 to ignore velocity in goal region
        min_steering angle will be set by calcuations of later force limit 
        """
        super().__init__(robot, map, start, goal,bounds,max_runtime,goal_threshold=pos_treshold) 
        self.propagate_step_size = propagate_step_size
        self.control_duration = control_duration  # (min_steps, max_steps)
        self.pruning_radius = pruning_radius
        self.selection_radius = selection_radius
        self.vel_threshold = vel_threshold
        self.pos_treshold = pos_treshold
        self.robot.max_steering_at_max_v = min( math.atan(self.robot.wheelbase * self.robot.mu_static * 9.81 / self.robot.max_velocity**2), self.robot.max_steering_at_zero_v) 

        if selection_radius is None:    
            self.selection_radius = self.robot.max_velocity * self.propagate_step_size * self.control_duration[1] * 2
        if pruning_radius is None:
            self.pruning_radius = self.robot.max_velocity * self.propagate_step_size * self.control_duration[1] * 0.5
        #TODO pruning and selection radius adaptive to robot max velocity * propagate step size * control_duration 

        self._debug_counter = 0 




    def propagate(self,state, control, result):
     
        """
        State Propagator: Defines the dynamics of the car model.
        This function implements the Kinematic Car (Bicycle) Model ODEs,
        now including a velocity-dependent steering constraint.
        
        State: [x, y, theta, v]
        Control: [acceleration, steering_angle]
        
        The differential equations (ODE) are integrated over the 'duration' (dt).
        """

        # F  = mvv/r < C 

        # MAX_DELTA = self.robot.max_steering_at_zero_v -  np.clip(state[3] / self.robot.max_velocity, 0.0, 1.0) * (self.robot.max_steering_at_zero_v - self.robot.max_steering_at_max_v)
        
        # R_min = state[3]**2 / (self.robot.mu_static * 9.81) # min turning radius based on lateral friction limit          
    
    
        #TODO optimaze tan and atan computation 
        #TODO g is constant 
        MAX_DELTA = self.robot.max_steering_at_zero_v

        if state [3] >= 0.1:
            MAX_DELTA = min( math.atan(self.robot.wheelbase * self.robot.mu_static * 9.81 / state[3]**2), self.robot.max_steering_at_zero_v) 

        # self._debug_counter += 1
        # if self._debug_counter % 100000 == 0:
        #     print(self._debug_counter / 1000000 , 'MIL propagation steps')
        result[0] =  state[3] * math.cos(state[2])  
        result[1] = state[3] * math.sin(state[2])  
        result[2] = (state[3] / self.robot.wheelbase) * math.tan(np.clip(control[1], -MAX_DELTA, MAX_DELTA)) 
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
        space.lock() # Finalize the state space structure

        cspace = oc.RealVectorControlSpace(space, 2)
        c_bounds = ob.RealVectorBounds(2)
        #acceleration bounds
        c_bounds.setLow(0, - self.robot.acceleration) 
        c_bounds.setHigh(0, self.robot.acceleration)

        #steering angle bounds
        c_bounds.setLow(1, -self.robot.max_steering_at_zero_v)
        c_bounds.setHigh(1, self.robot.max_steering_at_zero_v)
        
        cspace.setBounds(c_bounds)


        si = oc.SpaceInformation(space, cspace)
        si.setPropagationStepSize(self.propagate_step_size)
        si.setMinMaxControlDuration(self.control_duration[0], self.control_duration[1])  # Min 0.02s, Max 0.3s per control


        validity_checker = ob.StateValidityCheckerFn(partial(self.is_state_valid, si))
        si.setStateValidityChecker(validity_checker)
        #TODO tune resolution  5 checks per length of body of robot implemt this for circular robot 
        #TODO is this per bounds as it's has to be between 0 and 1 
        #TODO maybe base this on the size of the map as it wont get better when the map is small 
        si.setStateValidityCheckingResolution(min (self.robot.width, self.robot.length) * 0.005 / max ( self.bounds[0] , self.bounds[1]) )
        # si.setStateValidityCheckingResolution(0.001 )

        #TODO tune adaptive ode solver
        ode = oc.ODE(self.propagate)
        odeSolver = oc.ODEBasicSolver(si, ode)
        #changes the step size adaptively based on estimated error 
        # odeSolver = oc.ODEAdaptiveSolver(si, ode,0.01)

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
    map = np.ones((400,400))
    map[0,0] = 0
    map[80:,80:240] = 0    
    robot =RectangleRobot(0.5,1.0,max_velocity=10,mu_static=2,collision_check_angle_res=30)
    robot = Robot()
    car_planner = SSTCarOMPL_acceleration(max_runtime=15, map=map,robot =robot,vel_threshold=200,bounds=(15,15))
    print('Solved', car_planner.solve())
    print('Max velocity:', car_planner.visualize() , '/', car_planner.robot.max_velocity)
    print('Time taken by path:', car_planner.solved_time)

