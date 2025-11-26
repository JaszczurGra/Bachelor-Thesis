#!/usr/bin/env python3
"""
OMPL kinematic car planning with velocity-dependent steering and acceleration control.
Uses control-based planning (RRT with state propagation) on SE(2) × velocity state space.
"""

import math
try:
    from ompl import base as ob
    from ompl import control as oc
except ImportError:
    raise ImportError("Install OMPL: conda install -c conda-forge ompl (or pip install ompl)")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


# --- Environment Setup ---
SPACE_MIN_X, SPACE_MAX_X = 0.0, 100.0
SPACE_MIN_Y, SPACE_MAX_Y = 0.0, 100.0
MIN_VELOCITY, MAX_VELOCITY = -5.0, 15.0  # m/s (negative = reverse)

# Circular obstacles: (cx, cy, radius)
OBSTACLES = [
    (30.0, 30.0, 10.0),
    (60.0, 50.0, 12.0),
    (50.0, 80.0, 8.0),
    (20.0, 70.0, 7.0),
]

START = (10.0, 10.0, 0.0, 0.0)  # (x, y, theta, velocity)
GOAL  = (90.0, 90.0, 0.0, 0.0)

# Car parameters
WHEELBASE = 2.5        # distance between front and rear axle (m)
MAX_STEERING = 0.6     # max steering angle (radians) ~34 degrees
MIN_ACCELERATION = -3.0  # m/s^2
MAX_ACCELERATION = 2.0   # m/s^2

# Velocity-dependent steering limits (Ackermann-like)
def get_max_steering_for_velocity(v):
    """Reduce max steering angle at high speeds for realism."""
    abs_v = abs(v)
    if abs_v < 1.0:
        return MAX_STEERING
    # linear decay: full steering at v=0, reduced at high speed
    factor = max(0.3, 1.0 - (abs_v - 1.0) / 20.0)
    return MAX_STEERING * factor


def is_state_valid(state):
    """Check if state (x, y, theta, v) is collision-free."""
    x = state[0]
    y = state[1]
    # bounds check
    if not (SPACE_MIN_X <= x <= SPACE_MAX_X and SPACE_MIN_Y <= y <= SPACE_MAX_Y):
        return False
    # obstacle check
    for cx, cy, r in OBSTACLES:
        if math.hypot(x - cx, y - cy) <= r + 1.0:  # car safety margin
            return False
    return True


def car_ode(q, u, qdot):
    """
    Kinematic car ODE with acceleration and velocity-dependent steering.
    State q = [x, y, theta, v]
    Control u = [steering_angle, acceleration]
    qdot = derivative
    """
    x, y, theta, v = q[0], q[1], q[2], q[3]
    steering, accel = u[0], u[1]
    
    # velocity-dependent max steering (Ackermann constraint)
    max_steer = get_max_steering_for_velocity(v)
    steering = max(-max_steer, min(max_steer, steering))
    
    # kinematic bicycle model
    qdot[0] = v * math.cos(theta)                     # dx/dt
    qdot[1] = v * math.sin(theta)                     # dy/dt
    qdot[2] = (v / WHEELBASE) * math.tan(steering)    # dtheta/dt
    qdot[3] = accel                                    # dv/dt


def propagate(start, control, duration, result):
    """Simple Euler integration of car dynamics."""
    steps = max(1, int(duration / 0.02))  # 50Hz integration
    dt = duration / steps
    state = [start[0], start[1], start[2], start[3]]
    qdot = [0.0, 0.0, 0.0, 0.0]
    
    for _ in range(steps):
        car_ode(state, control, qdot)
        state[0] += qdot[0] * dt
        state[1] += qdot[1] * dt
        state[2] += qdot[2] * dt
        state[3] += qdot[3] * dt
        # clamp velocity
        state[3] = max(MIN_VELOCITY, min(MAX_VELOCITY, state[3]))
    
    result[0] = state[0]
    result[1] = state[1]
    result[2] = state[2]
    result[3] = state[3]


def plan_with_control():
    """Plan a path using RRT with control-based propagation."""
    # 1. Create compound state space: SE(2) × R (velocity)
    space = ob.CompoundStateSpace()
    se2 = ob.SE2StateSpace()
    bounds_se2 = ob.RealVectorBounds(2)
    bounds_se2.setLow(0, SPACE_MIN_X)
    bounds_se2.setHigh(0, SPACE_MAX_X)
    bounds_se2.setLow(1, SPACE_MIN_Y)
    bounds_se2.setHigh(1, SPACE_MAX_Y)
    se2.setBounds(bounds_se2)
    
    velocity_space = ob.RealVectorStateSpace(1)
    bounds_v = ob.RealVectorBounds(1)
    bounds_v.setLow(0, MIN_VELOCITY)
    bounds_v.setHigh(0, MAX_VELOCITY)
    velocity_space.setBounds(bounds_v)
    
    space.addSubspace(se2, 1.0)      # weight for SE(2)
    space.addSubspace(velocity_space, 0.3)  # lower weight for velocity

    # 2. Create control space: [steering, acceleration]
    cspace = oc.RealVectorControlSpace(space, 2)
    cbounds = ob.RealVectorBounds(2)
    cbounds.setLow(0, -MAX_STEERING)
    cbounds.setHigh(0, MAX_STEERING)
    cbounds.setLow(1, MIN_ACCELERATION)
    cbounds.setHigh(1, MAX_ACCELERATION)
    cspace.setBounds(cbounds)

    # 3. Setup SpaceInformation
    si = oc.SpaceInformation(space, cspace)
    
    # Validity checker
    def validity_fn(state):
        se2_part = state[0]
        v = state[1][0]
        return is_state_valid([se2_part.getX(), se2_part.getY(), se2_part.getYaw(), v])
    
    si.setStateValidityChecker(ob.StateValidityCheckerFn(validity_fn))
    
    # State propagator
    def propagator_fn(start, control, duration, result):
        se2_start = start[0]
        v_start = start[1][0]
        q_start = [se2_start.getX(), se2_start.getY(), se2_start.getYaw(), v_start]
        u = [control[0], control[1]]
        q_result = [0.0, 0.0, 0.0, 0.0]
        propagate(q_start, u, duration, q_result)
        
        se2_result = result[0]
        se2_result.setX(q_result[0])
        se2_result.setY(q_result[1])
        se2_result.setYaw(q_result[2])
        result[1][0] = q_result[3]
    
    si.setStatePropagator(oc.StatePropagatorFn(propagator_fn))
    si.setPropagationStepSize(0.1)  # control duration per step
    si.setup()

    # 4. Define start and goal
    start_state = ob.State(space)
    start_state().assign(0,START[0])
    start_state().assign(1,START[1])
    start_state().assign(2,START[2])
    start_state().assign(3,START[3])
    print("Start State:", START)
    goal_state = ob.State(space)
    goal_state().assign(0,GOAL[0])
    goal_state().assign(1,GOAL[1])
    goal_state().assign(2,GOAL[2])
    goal_state().assign(3,GOAL[3])

    # 5. Problem definition
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state, 3.0)  # goal tolerance

    # 6. Planner (RRT for control)
    planner = oc.RRT(si)
    planner.setProblemDefinition(pdef)
    planner.setup()

    # 7. Solve
    print("Planning...")
    solved = planner.solve(30.0)  # 30 sec timeout

    if solved:
        print("Solution found!")
        path = pdef.getSolutionPath()
        path.asGeometric().interpolate(300)  # densify
        return path
    else:
        print("No solution found.")
        return None


def extract_path_data(path):
    """Extract (x, y, theta, v) from control path."""
    coords = []
    states = path.getStates()
    for state in states:
        x = state[0].getX()
        y = state[0].getY()
        theta = state[0].getYaw()
        v = state[1][0]
        coords.append((x, y, theta, v))
    return coords


def visualize_solution(path_data):
    """Draw environment, path, and velocity profile."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left plot: spatial path ---
    ax1.set_xlim(SPACE_MIN_X, SPACE_MAX_X)
    ax1.set_ylim(SPACE_MIN_Y, SPACE_MAX_Y)
    ax1.set_aspect('equal')

    for cx, cy, r in OBSTACLES:
        circle = Circle((cx, cy), r, color='orange', alpha=0.5)
        ax1.add_patch(circle)

    ax1.plot(START[0], START[1], 'go', markersize=12, label='Start')
    ax1.plot(GOAL[0], GOAL[1], 'rx', markersize=12, label='Goal')

    if path_data:
        xs = [p[0] for p in path_data]
        ys = [p[1] for p in path_data]
        vs = [p[3] for p in path_data]
        
        # color by velocity
        norm = plt.Normalize(vmin=MIN_VELOCITY, vmax=MAX_VELOCITY)
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        from matplotlib.collections import LineCollection
        lc = LineCollection(segments, cmap='coolwarm', norm=norm, linewidth=3)
        lc.set_array(np.array(vs))
        ax1.add_collection(lc)
        cbar = fig.colorbar(lc, ax=ax1)
        cbar.set_label('Velocity (m/s)')

    ax1.legend()
    ax1.set_title("Control-Based RRT Path (Velocity-Colored)")
    ax1.grid(True, alpha=0.3)

    # --- Right plot: velocity profile ---
    if path_data:
        t = np.arange(len(path_data)) * 0.1  # approx time
        vs = [p[3] for p in path_data]
        ax2.plot(t, vs, 'b-', linewidth=2)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Velocity Profile')
        ax2.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = plan_with_control()
    if path:
        data = extract_path_data(path)
        visualize_solution(data)
    else:
        visualize_solution([])