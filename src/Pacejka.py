import ompl.base as ob
import ompl.control as oc
import ompl.util as ou
import math
from functools import partial
from base_pathfind_classes import KinematicGoalRegionWithVelocity, RectangleRobot, Robot, KinematicGoalRegion,BasePathfinding,PacejkaRectangleRobot,PacejkaTireModel
import numpy as np

import ompl.util as ou

ou.setLogLevel(ou.LOG_NONE) 
# ou.setLogLevel(ou.LOG_DEBUG) 


import io 


    

#TODO desarializing for vis 


  


   

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

        robot.acceleration = 0 
        robot.max_steering_at_max_v = 0 

        robot.eps = propagate_step_size



        #TODO how to calculate the max movemnt in one control step  
        if selection_radius is None:    
            self.selection_radius = robot.max_velocity * propagate_step_size * control_duration[1] *  2
        if pruning_radius is None:
            self.pruning_radius = robot.max_velocity * propagate_step_size * control_duration[1] * 0.5





    def propagate(self,state, control, result):
        """
        control: [omega_wheels_ref, delta_ref]
        state: [x,y, yaw, v_x, v_y, r, omega_wheels, delta]
        """
        result = self.robot.forward(state, control, result)

    def solve(self):

        pos_bounds = ob.RealVectorBounds(2)
        pos_bounds.setLow(0.0)
        pos_bounds.setHigh(0, self.bounds[0])
        pos_bounds.setHigh(1, self.bounds[1])

        r2 = ob.RealVectorStateSpace(2)
        r2.setBounds(pos_bounds)

        so2 = ob.SO2StateSpace()



        # TODO shouldn't be bound ? 
        v_bounds = ob.RealVectorBounds(2)
        v_bounds.setLow (-self.robot.max_velocity)
        v_bounds.setHigh (self.robot.max_velocity)
        v_state = ob.RealVectorStateSpace(2)
        v_state.setBounds(v_bounds)



        o_bounds = ob.RealVectorBounds(3)
        #TODO this is refrence omega should the min val be 0 or -max omega for faster slowing down

        o_bounds.setLow(0,-10)
        o_bounds.setHigh(0,10)
        o_bounds.setLow(1,-self.robot.max_velocity / self.robot.R)
        o_bounds.setHigh(1,self.robot.max_velocity / self.robot.R)
        o_bounds.setLow(2,-10)
        o_bounds.setHigh(2,10)

        other_params = ob.RealVectorStateSpace(3)
        other_params.setBounds(o_bounds)
   
   
   
        space = ob.CompoundStateSpace()
        
        space.addSubspace(r2, 1.0)      # x, y
        space.addSubspace(so2, 1.0)     # theta
        space.addSubspace(v_state,1.0)   # vx, vy
        space.addSubspace(other_params, 1.0) #r omega_w, delta

        space.lock() # Finalize the state space structure

        cspace = oc.RealVectorControlSpace(space, 2)
        c_bounds = ob.RealVectorBounds(2)
        #omega wheels speed refrence
        c_bounds.setLow(0, 0.05) 
        c_bounds.setHigh(0, self.robot.max_velocity / self.robot.R)

        #delta bounds
        #TODO is this acutal max steering agngle? 
        c_bounds.setLow(1, -self.robot.max_steering_at_zero_v)
        c_bounds.setHigh(1, self.robot.max_steering_at_zero_v)
        cspace.setBounds(c_bounds)


        si = oc.SpaceInformation(space, cspace)
        si.setPropagationStepSize(self.propagate_step_size)
        si.setMinMaxControlDuration(self.control_duration[0], self.control_duration[1])  # Min 0.02s, Max 0.3s per control


        validity_checker = ob.StateValidityCheckerFn(partial(self.is_state_valid, si))
        si.setStateValidityChecker(validity_checker)
        si.setStateValidityCheckingResolution(0.2)

        #TODO add adaptive ode solver
        ode = oc.ODE(self.propagate)
        # odeSolver = oc.ODEBasicSolver(si, ode)
        odeSolver = oc.ODEAdaptiveSolver(si, ode,0.01)

        propagator = oc.ODESolver.getStatePropagator(odeSolver)
        si.setStatePropagator(propagator)
    
        si.setup()


        pdef = ob.ProblemDefinition(si)


        intital_v = 0.001

        start = ob.State(si)
        start()[0][0], start()[0][1] = self.start[:2]
        start()[1].value,start()[2][0] = (self.start[2],0.0) #(math.pi / 2.0,0.0)
        start()[2][0], start()[2][1] = (intital_v,0) #intial v
        start()[3][0], start()[3][1], start()[3][2] = (0.0,intital_v/ self.robot.R,0.0) #intial r, omega wheels, delta
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
        # planner = oc.RRT(si)
       
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
    
    def visualize(self, ax=None, path_data_str=None,point_iteration=3,path_iteration=1,velocity_scale =0.2):
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
    # 
    # 
    # TODO for goal region normalize the velocity from x,y not only take x into consideration      
    car_planner = Pacejka_pathfinding(max_runtime=60, map=map,robot =PacejkaRectangleRobot(0.5,1,max_velocity=200),vel_threshold=0.5,velocity_weight=0.5)
    print('solved', car_planner.solve())
    car_planner.visualize()















