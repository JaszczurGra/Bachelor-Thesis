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




class SSTCarOMPL_acceleration(BasePathfinding):
    def __init__(self,robot=Robot(),map=None,start=(1.0,1.0, 0.0),goal=(9.0,9.0,0.0), bounds=(0,10,0,10),max_runtime=30.0, propagate_step_size=0.02, control_duration=(1,10), selection_radius=None, pruning_radius=None, velocity_weight=0.0, vel_threshold=4.0, pos_treshold=0.5):
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

        if selection_radius is None:    
            self.selection_radius = self.robot.max_velocity * self.propagate_step_size * self.control_duration[1] * 2
        if pruning_radius is None:
            self.pruning_radius = self.robot.max_velocity * self.propagate_step_size * self.control_duration[1] * 0.5
        #TODO pruning and selection radius adaptive to robot max velocity * propagate step size * control_duration 





    def propagate(self,state, control, result):
     
        """
        State Propagator: Defines the dynamics of the car model.
        This function implements the Kinematic Car (Bicycle) Model ODEs,
        now including a velocity-dependent steering constraint.
        
        State: [x, y, theta, v]
        Control: [acceleration, steering_angle]
        
        The differential equations (ODE) are integrated over the 'duration' (dt).
        """
        #TODO cap the steering angle based * by dt as this is theta dot 
        MAX_DELTA = self.robot.max_steering_at_zero_v -  np.clip(state[3] / self.robot.max_velocity, 0.0, 1.0) * (self.robot.max_steering_at_zero_v - self.robot.max_steering_at_max_v)
        delta = np.clip(control[1], -MAX_DELTA, MAX_DELTA)
   
        result[0] =  state[3] * math.cos(state[2])  
        result[1] = state[3] * math.sin(state[2])  
        result[2] = (state[3] / self.robot.wheelbase) * math.tan(delta) 
        # result[3] = np.clip(control[0], 0.0, self.robot.max_velocity) 
        result[3] = control[0]

    def solve(self):
        space = ob.CompoundStateSpace()

        pos_bounds = ob.RealVectorBounds(2)
        pos_bounds.setLow(0.0)
        pos_bounds.setHigh(10.0)
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
        si.setStateValidityCheckingResolution(0.3)

        #TODO add adaptive ode solver
        ode = oc.ODE(self.propagate)
        odeSolver = oc.ODEBasicSolver(si, ode)
        odeSolver = oc.ODEAdaptiveSolver(si, ode,0.01)

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
        if self.velocity_weight > 0:
            goal_region = KinematicGoalRegionWithVelocity(si, goal, pos_threshold=self.pos_treshold,velocity_threshold=self.vel_threshold,velocity_weight=self.velocity_weight,bounds=self.bounds,max_velocity=self.robot.max_velocity) 
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
            pdef.getSolutionPath().interpolate()
            # pdef.getSolutionPath().setResolution(0.001)
            # pdef.getSolutionPath().interpolate()
            self.solved_path = pdef.getSolutionPath().printAsMatrix()
            return pdef.hasExactSolution()
        return None


if __name__ == "__main__": 
    ou.setLogLevel(ou.LOG_DEBUG) 
    car_planner = SSTCarOMPL_acceleration(max_runtime=3, map=np.ones((100,100)),robot =RectangleRobot(0.5,1.0))
    print(car_planner.solve())
    car_planner.visualize()

