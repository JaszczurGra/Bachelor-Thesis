from train import BSpline
import numpy as np
from scipy.interpolate import BSpline as SciPyBSpline
# spline = BSpline(n=3,d=7,num_T_pts=100)


path = np.array([
    [0, 0],
    [1, 2],
    [3, 1],
    [4, 4],
    [0, 0],
    [1, 2],
    [3, 1],
    [4, 4],


])  # shape (3, 2)

# Create the BSpline object
d = 7
print('starting')
spline = BSpline(n=d+1, d=d, num_T_pts=5000)
# scipyspline = SciPyBSpline()
print(spline.N[0])
# Generate the smooth curve
curve = spline.N[0] @ path[:d+1]  # shape (100, 2)

print('curve calculated')

control_points, *_ = np.linalg.lstsq(spline.N[0], curve, rcond=None)
print(control_points)
