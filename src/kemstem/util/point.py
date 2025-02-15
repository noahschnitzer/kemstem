import numpy as np
from scipy.spatial import KDTree
from skimage.transform import rotate

'''
def get_nearest_points(points,guesses,k=1):
    kd = KDTree(points)
    dist,idx = kd.query(guesses,k=k)
    match_points = points[idx]
    return (match_points,idx,dist)
'''

def get_nearest_points(points, guesses, k=1, handle_nan=False):
    """
    Find k nearest neighbors for each point in guesses using KDTree.
    Handles NaN values in guesses by returning NaN for those positions (still finnicky).
    
    Parameters:
    points: array-like, shape (n, d)
        The reference points to search within
    guesses: array-like, shape (m, d)
        The query points to find neighbors for
    k: int
        Number of nearest neighbors to find
        
    Returns:
    tuple of (match_points, idx, dist) where each has NaN values
    corresponding to NaN inputs in guesses
    """
    guesses = np.asarray(guesses)
    points = np.asarray(points)
    # typical case: no nans (or only a single point)
    if (not handle_nan) or (len(guesses.shape) == 1) or (np.isnan(guesses).sum() == 0):
        kd = KDTree(points)
        dist,idx = kd.query(guesses,k=k)
        match_points = points[idx]
        return (match_points,idx,dist)

    
    # Find rows with any NaN values
    nan_mask = np.any(np.isnan(guesses), axis=1)
    
    if np.all(nan_mask):
        # All guesses are NaN, return arrays of NaN
        if k > 1:
            match_shape = (len(guesses), k, points.shape[1])
            idx_shape = (len(guesses), k)
            dist_shape = (len(guesses), k)
        else:
            match_shape = (len(guesses), points.shape[1])
            idx_shape = (len(guesses),)
            dist_shape = (len(guesses),)
            
        return (
            np.full(match_shape, np.nan),
            np.full(idx_shape, -1, dtype=int),
            np.full(dist_shape, np.nan)
        )
    
    # Only query KDTree with non-NaN points
    valid_guesses = guesses[~nan_mask]
    
    # Create KDTree and query with valid points
    kd = KDTree(points)
    dist_valid, idx_valid = kd.query(valid_guesses, k=k)
    
    # Initialize output arrays with NaN/invalid values
    if k > 1:
        match_shape = (len(guesses), k, points.shape[1])
        idx_shape = (len(guesses), k)
        dist_shape = (len(guesses), k)
    else:
        match_shape = (len(guesses), points.shape[1])
        idx_shape = (len(guesses),)
        dist_shape = (len(guesses),)
    
    match_points = np.full(match_shape, np.nan)
    idx = np.full(idx_shape, -1, dtype=int)
    dist = np.full(dist_shape, np.nan)
    
    # Fill in results for valid queries
    match_points[~nan_mask] = points[idx_valid]
    idx[~nan_mask] = idx_valid
    dist[~nan_mask] = dist_valid
        
    return match_points, idx, dist


def rotate_points(points,angle,center=(0,0)):
    center = np.array(center)
    rotated = points - center
    rotated[:,1] = np.cos(angle)*(points[:,1] - center[1]) - np.sin(angle)*(points[:,0] - center[0])
    rotated[:,0] = np.sin(angle)*(points[:,1] - center[1]) + np.cos(angle)*(points[:,0] - center[0])
    rotated = rotated+ center
    return rotated

def rotate_image_and_calculate_transform(angle_rad, im):
    rot_im = rotate(im,-angle_rad*180/np.pi,resize=False,preserve_range=True) # rotation centered on (cols / 2 - 0.5, rows / 2 - 0.5)
    
    def applyToPoints(c):
        xs = c[:,1]
        ys = c[:,0]
        xs = xs - rot_im.shape[1]/2 - 0.5
        ys = ys - rot_im.shape[0]/2 - 0.5
        rot_xs = xs*np.cos(angle_rad) - ys*np.sin(angle_rad) +  rot_im.shape[1]/2  + 0.5
        rot_ys = xs*np.sin(angle_rad) + ys*np.cos(angle_rad) +  rot_im.shape[0]/2  + 0.5
        return np.array((rot_ys,rot_xs)).T
    def applyToImage(nim):
        return rotate(nim,-angle_rad*180/np.pi,resize=False,preserve_range=True) 

    return rot_im, applyToPoints,applyToImage


def extrapolate_point_pair(pt1,pt2,frac_pre,frac_post):
    d0 = pt2[0]-pt1[0]
    d1 = pt2[1]-pt1[1]
    
    adj1 = [pt1[0]-d0*frac_pre,pt1[1]-d1*frac_pre]
    adj2 = [pt2[0]+d0*frac_post,pt2[1]+d1*frac_post]

    return np.array((adj1,adj2))
