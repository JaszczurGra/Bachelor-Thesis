import ompl.base as ob
import ompl.control as oc
import ompl.util as ou
import math
from functools import partial
from base_pathfind_classes import Robot, KinematicGoalRegion,BasePathfinding
import numpy as np

import ompl.util as ou
ou.setLogLevel(ou.LOG_NONE) 


#TODO 
#robot as squere 
#penelty for velocity in goal region
#different planners




class SSTCarOMPL_acceleration(BasePathfinding):
    def __init__(self,robot=Robot(),Obstacles=[],start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5, bounds=(0,10,0,10),max_runtime=30.0, propagate_step_size=0.02, control_duration=(1,10), selection_radius=1.0, pruning_radius=0.005):
        super().__init__(robot, Obstacles, start, goal,bounds,max_runtime,goal_treshold=goal_treshold) 
        self.propagate_step_size = propagate_step_size
        self.control_duration = control_duration  # (min_steps, max_steps)
        self.selection_radius = selection_radius
        self.pruning_radius = pruning_radius
        #TODO pruning and selection radius adaptive to robot max velocity * propagate step size * control_duration 



    def is_state_valid(self,si, state):
        return self.robot.check_bounds(state,self.bounds) and not self.robot.check_collision(state,self.obstacles)
    

    def propagate(self,state, control, result):
     
        """
        State Propagator: Defines the dynamics of the car model.
        This function implements the Kinematic Car (Bicycle) Model ODEs,
        now including a velocity-dependent steering constraint.
        
        State: [x, y, theta, v]
        Control: [acceleration, steering_angle]
        
        The differential equations (ODE) are integrated over the 'duration' (dt).
        """

        MAX_DELTA = self.robot.max_steering_at_zero_v -  np.clip(state[3] / self.robot.max_velocity, 0.0, 1.0) * (self.robot.max_steering_at_zero_v - self.robot.max_steering_at_max_v)
        delta = np.clip(control[1], -MAX_DELTA, MAX_DELTA)
   
        result[0] = state[0] + state[3] * math.cos(state[2])  
        result[1] = state[1] + state[3] * math.sin(state[2])  
        result[2] = state[2] + (state[3] / self.robot.wheelbase) * math.tan(delta) 
        result[3] = np.clip(state[3] + control[0], 0.0, self.robot.max_velocity) 


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
        propagator = oc.ODESolver.getStatePropagator(odeSolver)
        si.setStatePropagator(propagator)
    
        si.setup()


        pdef = ob.ProblemDefinition(si)


        start = ob.State(si)
        start()[0][0], start()[0][1] = self.start_point
        start()[1].value,start()[2][0] =  (math.pi / 2.0,0.0)
        goal = ob.State(si)
        goal()[0][0], goal()[0][1] = self.goal_point
        goal()[1].value, goal()[2][0] = (0.0, 0.0)


        goal_region = KinematicGoalRegion(si, goal, threshold=0.5)
        pdef.setStartAndGoalStates (start,goal)
        pdef.setGoal(goal_region)


        # class MinimizeTimeObjective(ob.StateCostIntegralObjective):
        #     def __init__(self, si):
        #         super().__init__(si, True) 

        #     def motionCost(self, s1, s2):
        #         """
        #         Returns the cost of the motion between s1 and s2.
        #         For a minimum time objective, the cost is the duration (time).
                
        #         The difference between s2 and s1 is the total time elapsed. 
        #         OMPL tracks the duration of the control segment that generated this motion.
        #         """

        #         return ob.Cost(1.0) 

        # pdef.setOptimizationObjective(MinimizeTimeObjective(si))
        pdef.setOptimizationObjective(ob.StateCostIntegralObjective(si, True))

        planner = oc.SST(si)
        planner.setProblemDefinition(pdef)
        planner.setPruningRadius(self.pruning_radius)
        planner.setSelectionRadius(self.selection_radius)


        planner.setup()


        solved = planner.solve(self.max_runtime)

        if solved:
            self.solved_path = pdef.getSolutionPath().printAsMatrix()
            return pdef.hasExactSolution()
        return None



if __name__ == "__main__": 
    ou.setLogLevel(ou.LOG_DEBUG) 
    car_planner = SSTCarOMPL_acceleration(max_runtime=15)
    print(car_planner.solve())
    car_planner.visualize()

