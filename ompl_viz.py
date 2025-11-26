#!/usr/bin/env python3
# filepath: /home/jaszczurgra/Documents/Programing/Bachelor-Thesis/ompl_example_visual.py
"""
OMPL kinematic car planning example with Matplotlib visualization.
Uses RRT on SE(2) state space with circular obstacles.
"""

import math
try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    raise ImportError("Install OMPL: conda install -c conda-forge ompl (or pip install ompl)")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np


# --- Environment Setup ---
SPACE_MIN_X, SPACE_MAX_X = 0.0, 100.0
SPACE_MIN_Y, SPACE_MAX_Y = 0.0, 100.0

# Circular obstacles: (cx, cy, radius)
OBSTACLES = [
    (30.0, 30.0, 10.0),
    (60.0, 50.0, 12.0),
    (50.0, 80.0, 8.0),
    (20.0, 70.0, 7.0),
]

START = (10.0, 10.0, 0.0)  # (x, y, theta)
GOAL  = (90.0, 90.0, 0.0)


def is_state_valid(state):
    """Check if state (x, y, theta) is collision-free."""
    x = state.getX()
    y = state.getY()
    # bounds check
    if not (SPACE_MIN_X <= x <= SPACE_MAX_X and SPACE_MIN_Y <= y <= SPACE_MAX_Y):
        return False
    # obstacle check
    for cx, cy, r in OBSTACLES:
        if math.hypot(x - cx, y - cy) <= r:
            return False
    return True


def plan_with_rrt():
    """Plan a path using RRT on SE(2) and return the solution path."""
    # 1. Create SE(2) state space (x, y, theta)
    space = ob.SE2StateSpace()
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0, SPACE_MIN_X)
    bounds.setHigh(0, SPACE_MAX_X)
    bounds.setLow(1, SPACE_MIN_Y)
    bounds.setHigh(1, SPACE_MAX_Y)
    space.setBounds(bounds)

    # 2. Create SpaceInformation with validity checker
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ob.StateValidityCheckerFn(is_state_valid))
    si.setup()

    # 3. Define start and goal states
    start_state = ob.State(space)
    start_state().setX(START[0])
    start_state().setY(START[1])
    start_state().setYaw(START[2])

    goal_state = ob.State(space)
    goal_state().setX(GOAL[0])
    goal_state().setY(GOAL[1])
    goal_state().setYaw(GOAL[2])

    # 4. Create problem definition
    pdef = ob.ProblemDefinition(si)
    pdef.setStartAndGoalStates(start_state, goal_state, 2.0)  # goal tolerance 2.0

    # 5. Create planner (RRT)
    planner = og.RRT(si)

    # planner = og.STRRTstar(si)  # Using RRT*
    planner.setProblemDefinition(pdef)
    planner.setup()

    # 6. Solve
    solved = planner.solve(120.0)  # 10 second timeout

    if solved:
        print("Solution found!")
        path = pdef.getSolutionPath()
        path.interpolate(200)  # densify for smooth visualization
        return path
    else:
        print("No solution found.")
        return None


def extract_path_coords(path):
    """Extract (x, y) coordinates from OMPL geometric path."""
    coords = []
    states = path.getStates()
    for state in states:
        x = state.getX()
        y = state.getY()
        coords.append((x, y))
    return coords


def visualize_solution(path_coords):
    """Draw environment, obstacles, start, goal, and solution path."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # draw bounds
    ax.set_xlim(SPACE_MIN_X, SPACE_MAX_X)
    ax.set_ylim(SPACE_MIN_Y, SPACE_MAX_Y)
    ax.set_aspect('equal')

    # draw obstacles
    for cx, cy, r in OBSTACLES:
        circle = Circle((cx, cy), r, color='orange', alpha=0.5, label='Obstacle')
        ax.add_patch(circle)

    # draw start and goal
    ax.plot(START[0], START[1], 'go', markersize=12, label='Start')
    ax.plot(GOAL[0], GOAL[1], 'rx', markersize=12, label='Goal')

    # draw path
    if path_coords:
        xs, ys = zip(*path_coords)
        ax.plot(xs, ys, 'b-', linewidth=2, label='RRT Path')

    ax.legend()
    ax.set_title("OMPL RRT Path Planning (SE2)")
    ax.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    path = plan_with_rrt()
    if path:
        coords = extract_path_coords(path)
        visualize_solution(coords)
    else:
        # still show environment
        visualize_solution([])