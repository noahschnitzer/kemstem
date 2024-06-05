import numpy as np
from scipy.spatial import KDTree


def get_nearest_points(points,guesses,k=1):
    kd = KDTree(points)
    dist,idx = kd.query(guesses,k=k)
    # check that these make sense for non zero k.......
    match_points = points[idx]
    return (match_points,idx,dist)