import numpy as np
from scipy.spatial import KDTree


def get_nearest_points(points,guesses,k=1):
    kd = KDTree(points)
    dist,idx = kd.query(guesses,k=k)
    # check that these make sense for non zero k.......
    match_points = points[idx]
    return (match_points,idx,dist)

def rotate_points(points,angle,center=(0,0)):
    center = np.array(center)
    rotated = points - center
    rotated[:,1] = np.cos(angle)*(points[:,1] - center[1]) - np.sin(angle)*(points[:,0] - center[0])
    rotated[:,0] = np.sin(angle)*(points[:,1] - center[1]) + np.cos(angle)*(points[:,0] - center[0])
    rotated = rotated+ center
    return rotated