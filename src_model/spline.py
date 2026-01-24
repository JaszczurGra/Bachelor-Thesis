from sys import path
from train import BSpline
from scipy.interpolate import BSpline as SciPyBSpline
# spline = BSpline(n=3,d=7,num_T_pts=100)


# path = np.array([
#     [0, 0],
#     [1, 2],
#     [3, 1],
#     [4, 4],
#     [0, 0],
#     [1, 2],
#     [3, 1],
#     [4, 4],


# ])  # shape (3, 2)

# # Create the BSpline object
# d = 7
# print('starting')
# spline = BSpline(n=d+1, d=d, num_T_pts=5000)
# # scipyspline = SciPyBSpline()
# print(spline.N[0])
# # Generate the smooth curve
# curve = spline.N[0] @ path[:d+1]  # shape (100, 2)

# print('curve calculated')

# control_points, *_ = np.linalg.lstsq(spline.N[0], curve, rcond=None)
# print(control_points)

# Python program to explain 
# cv2.polylines() method

# import cv2
import numpy as np

# # # path
# # img = np.zeros((200,200), np.uint8)

# # # Polygon corner points coordinates
# # pts = np.array([[[25, 70], [25, 145],
# #                 [75, 190], [150, 190],
# #                 [200, 145], [250, 70], 
# #                 [150, 25], [75, 25]]],
# #                np.int32)

# # # pts = pts.reshape((-1, 1, 2))
# # print(pts)


# # print(pts / 125 - 1)

# # exit(0)
# # isClosed = True

# # Green color in BGR
# color = 1

# # Line thickness of 8 px
# thickness = 11

# # Using cv2.polylines() method
# # Draw a Green polygon with 
# # thickness of 1 px
# image = cv2.polylines(img, [pts], 
#                       isClosed, color, 
#                       thickness)

# # Displaying the image
# while(1):
    
#     cv2.imshow('image', image * 255)
#     if cv2.waitKey(20) & 0xFF == 27:
        
#         break
# cv2.destroyAllWindows()

# def calculate_path_metrics(path):
#     """
#     Inputs:
#         path: A numpy array of shape (N, 2)
#     Returns:
#         radii: Turning radius at each point
#         curvatures: Curvature (1/R) at each point
#         straightness_score: The Mean Squared Curvature of the path
#     """
#     N = path.shape[0]
#     # Initialize arrays with 'inf' for radius and 0 for curvature
#     radii = np.full(N, np.inf)
#     curvatures = np.zeros(N)

#     # Sliding window approach: starts at index 1, ends at N-2
#     for i in range(1, N - 1):
#         p1 = path[i - 1]
#         p2 = path[i]
#         p3 = path[i + 1]

#         # 1. Calculate the lengths of the sides of the triangle
#         a = np.linalg.norm(p2 - p1)
#         b = np.linalg.norm(p3 - p2)
#         c = np.linalg.norm(p3 - p1)

#         # 2. Calculate the area of the triangle using the cross product
#         # Area = 0.5 * |x1(y2-y3) + x2(y3-y1) + x3(y1-y2)|
#         area = 0.5 * abs(p1[0]*(p2[1] - p3[1]) + 
#                          p2[0]*(p3[1] - p1[1]) + 
#                          p3[0]*(p1[1] - p2[1]))

#         # 3. Calculate Curvature and Radius
#         # Curvature k = 4*Area / (a*b*c)
#         if area > 1e-9:  # Avoid division by zero for straight lines
#             k = (4 * area) / (a * b * c)
#             r = 1 / k
#         else:
#             k = 0
#             r = np.inf

#         curvatures[i] = k
#         radii[i] = r

#     # 4. Calculate the Straightness Score (Mean Squared Curvature)
#     # Lower is straighter. We exclude the endpoints.
#     straightness_score = np.mean(curvatures[1:-1]**2)

#     return radii, curvatures, straightness_score

# # Example Usage:
# # Create a path: a straight line that then turns
# path = np.array([
#     [0, 0], [1, 0], [2, 0], [3, 0.1], [3.5, 1], [3.8, 2], [4, 3]
# ])

# r, k, score = calculate_path_metrics(path)

# print(f"Radii: {r}")
# print(f"Curvatures: {k}")

# print(f"Straightness Score (Lower is better): {score:.6f}")
import re 
def parse_run_url(url):
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/(?:sweeps/[^/]+/)?runs/([^/?]+)", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    else:
        raise ValueError("Invalid wandb run URL format.")

def parse_sweep_url(url):
    m = re.search(r"wandb\.ai/([^/]+)/([^/]+)/sweeps/([^/?#]+)", url)
    if m:
        return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
    else:
        raise ValueError("Invalid wandb sweep URL format.")

import wandb

sweep_id = "your-entity/your-project/abc123xyz"  # Replace with your sweep path
sweep_id = "https://wandb.ai/j-boro-poznan-university-of-technology/Bachelor-Thesis-src_model/sweeps/qwyctg7b/"
print(sweep_id)


api = wandb.Api()
sweep = api.sweep(parse_sweep_url(sweep_id))
for run in sweep.runs:

    if run.state != 'failed':
        print(run.name, run.state)
    # You can access run.config, run.summary, run.files(), etc.
    # For example, download a model checkpoint:
    # for file in run.files():
    #     if "model" in file.name:
    #         file.download(replace=True)