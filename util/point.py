import numpy as np
from scipy.spatial import KDTree


def get_nearest_points(all_xs,all_ys,guess_xs,guess_ys,k=1):
    kd = KDTree(np.stack((all_xs,all_ys),axis=1))
    dist,idx = kd.query(np.stack((guess_xs,guess_ys),axis=1),k=k)
    # check that these make sense for non zero k.......
    match_xs = all_xs[idx]
    match_ys = all_ys[idx]
    return (dist,idx,match_xs,match_ys)