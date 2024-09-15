from .. import util
import numpy as np

def measure_displacements(reference_columns,distorted_columns,threshold=2):
    """
    Measure displacements between reference and distorted column positions.
    
    Parameters
    ----------
    reference_columns : ndarray, shape (n,2)
        Array of reference column positions.
    distorted_columns : ndarray, shape (n,2)
        Array of distorted column positions.
    threshold : float, optional
        Maximum distance to consider a match between reference and distorted columns (default is 2).
    
    Returns
    -------
    disps : ndarray, shape (n,2)
        Array of displacement vectors. NaN values indicate no match found within the threshold.
    
    Notes
    -----
    This function finds the nearest distorted column for each reference column
    and calculates the displacement vector. Displacements larger than the threshold
    are set to NaN.
    """

    corresp_dist_cols,_,dists = util.point.get_nearest_points(distorted_columns,reference_columns,k=1)
    disps = corresp_dist_cols - reference_columns
    disps[dists > threshold] = np.nan
    return disps
