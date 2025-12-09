import ompl.base as ob
import ompl.control as oc
import ompl.util as ou
import ompl.geometric as og
import numpy as np
import math
from functools import partial
from ompl import geometric as og # needed for asGeometric()
from base_pathfind_classes import Robot,BasePathfinding

import time 

import matplotlib.pyplot as plt
import numpy as np
import io


import ompl.util as ou
ou.setLogLevel(ou.LOG_NONE) 






class Dubins_pathfinding(BasePathfinding):
    def __init__(self,robot=Robot(),map=None,start=(1.0,1.0),goal=(9.0,9.0),max_runtime=30.0,bounds=(0,10,0,10),interpolate_steps=50):
        super().__init__(robot, map, start, goal, bounds, max_runtime) 
        self.interpolate_steps = interpolate_steps

        #TODO add robot max_steering as dubins wheelbase 

    def solve(self):
        space = ob.DubinsStateSpace(self.robot.wheelbase / math.tan(self.robot.max_steering_at_zero_v), False)

        bounds = ob.RealVectorBounds(2)
        bounds.low[0], bounds.high[0], bounds.low[1], bounds.high[1] = self.bounds
        space.setBounds(bounds)


        ss = og.SimpleSetup(space)
        si = ss.getSpaceInformation()

        start = ob.State(si)
        start()[0][0], start()[0][1] = self.start

        goal = ob.State(si)
        goal()[0][0], goal()[0][1] = self.goal
 

        si.setup()


        ss.setStartAndGoalStates (start,goal)

        # ignore_theta_goal = True
        # if ignore_theta_goal:
        #     goal_state = ob.State(space)
        #     goal_state().setX(self.goal_point[0])
        #     goal_state().setY(self.goal_point[1])
        #     goal_state().setYaw(0.0) 
            
        #     def goal_distance(state):
        #         dx = state[0][0] - self.goal_point[0]
        #         dy = state[0][1] - self.goal_point[1]
        #         return math.sqrt(dx * dx + dy * dy)
        #     ss.setGoal(ob.GoalState(si, goal_state, goal_distance))

        ss.setStateValidityChecker(ob.StateValidityCheckerFn(partial(self.is_state_valid, si)))
        ss.getSpaceInformation().setStateValidityCheckingResolution(0.005)

        planer = og.RRTstar(ss.getSpaceInformation())
        ss.setPlanner(planer)

        solved = ss.solve(self.max_runtime)


        if solved:
            ss.getSolutionPath().interpolate(self.interpolate_steps)
            self.solved_path = ss.getSolutionPath().printAsMatrix()
            return ss.haveExactSolutionPath()
        return None


  

if __name__ == "__main__": 
    ou.setLogLevel(ou.LOG_DEBUG) 
    car_planner = Dubins_pathfinding(max_runtime=5)
    print(car_planner.solve())
    car_planner.visualize()
    plt.show()

