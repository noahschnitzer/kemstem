from .. import util
import numpy as np

def measure_displacements(reference_columns,distorted_columns,threshold=2):
    corresp_dist_cols,_,dists = util.point.get_nearest_points(distorted_columns,reference_columns,k=1)
    disps = corresp_dist_cols - reference_columns
    disps[dists > threshold] = np.nan
    return disps
