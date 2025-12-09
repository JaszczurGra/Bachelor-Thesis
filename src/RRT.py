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




class RRT_Planer(BasePathfinding):
    def __init__(self,robot=Robot(),Obstacles=[],start=(1.0,1.0),goal=(9.0,9.0),goal_treshold=0.5, bounds=(0,10,0,10),max_runtime=30.0, propagate_step_size=0.02, control_duration=(1,10), selection_radius=1.0, pruning_radius=0.005):
        super().__init__(robot, Obstacles, start, goal,bounds,max_runtime,goal_threshold=goal_treshold) 
        self.propagate_step_size = propagate_step_size
        self.control_duration = control_duration  # (min_steps, max_steps)

    def is_state_valid(self,si, state):
        return self.robot.check_bounds(state,self.bounds) and not self.robot.check_collision(state,self.obstacles)
    

    def propagate(self,state, control, result):
        result[0] =   math.cos(control[0]) * self.robot.max_velocity  
        result[1] =  math.sin(control[0]) * self.robot.max_velocity  
        result[2] = control[0]


    def solve(self):

        space = ob.SE2StateSpace()

        bounds = ob.RealVectorBounds(2)
        bounds.low[0], bounds.high[0], bounds.low[1], bounds.high[1] = self.bounds
        space.setBounds(bounds)


        cspace = oc.RealVectorControlSpace(space, 1)
        c_bounds = ob.RealVectorBounds(1)

        c_bounds.setLow( -self.robot.max_steering_at_zero_v)
        c_bounds.setHigh( self.robot.max_steering_at_zero_v)

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

        start()[0][0], start()[0][1] = self.start
        start()[1].value=  math.pi / 2.0
        goal = ob.State(si)
        goal()[0][0], goal()[0][1] = self.goal
        goal()[1].value = 0.0

        

        goal_region = KinematicGoalRegion(si, goal, pos_threshold=0.5)
        pdef.setStartAndGoalStates (start,goal)
        pdef.setGoal(goal_region)


        pdef.setOptimizationObjective(ob.StateCostIntegralObjective(si, True))

        planner = oc.RRT(si)
        planner.setProblemDefinition(pdef)


        planner.setup()


        solved = planner.solve(self.max_runtime)

        if solved:
            self.solved_path = pdef.getSolutionPath().printAsMatrix()
            return pdef.hasExactSolution()
        return None



if __name__ == "__main__": 
    ou.setLogLevel(ou.LOG_DEBUG) 
    car_planner = RRT_Planer(max_runtime=15)
    print(car_planner.solve())
    car_planner.visualize()

