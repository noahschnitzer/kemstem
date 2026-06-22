import numpy as np
from scipy.spatial import KDTree, cKDTree
from scipy import interpolate
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
    
    Parameters
    ----------
    points: array-like, shape (n, d)
        The reference points to search within
    guesses: array-like, shape (m, d)
        The query points to find neighbors for
    k: int
        Number of nearest neighbors to find
        
    Returns
    -------
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
    """
    Rotate points around a center by a given angle.
    
    Parameters
    ----------
    points : array-like, shape (n, 2)
        List of (x,y) points to rotate.
    angle : float
        Rotation angle in radians.
    center : tuple, optional
        Center of rotation (x,y). Default is (0,0).
        
    Returns
    -------
    rotated : ndarray, shape (n, 2)
        The rotated points.
    """
    center = np.array(center)
    rotated = points - center
    rotated[:,1] = np.cos(angle)*(points[:,1] - center[1]) - np.sin(angle)*(points[:,0] - center[0])
    rotated[:,0] = np.sin(angle)*(points[:,1] - center[1]) + np.cos(angle)*(points[:,0] - center[0])
    rotated = rotated+ center
    return rotated

def rotate_image_and_calculate_transform(angle_rad, im):
    """
    Rotate an image and return functions to map points between original and rotated coordinates.
    
    Parameters
    ----------
    angle_rad : float
        Rotation angle in radians.
    im : ndarray
        Image to rotate.
        
    Returns
    -------
    rot_im : ndarray
        The rotated image.
    applyToPoints : function
        Function that takes points (y,x) in original image coordinates and returns
        points in the rotated image coordinates.
    applyToImage : function
        Function that takes an image and applies the same rotation.
    """
    rot_im = rotate(im,-angle_rad*180/np.pi,resize=False,preserve_range=True) # rotation centered on (cols / 2 - 0.5, rows / 2 - 0.5)
    
    def applyToPoints(c):
        """Map points from original image to rotated image coordinates."""
        xs = c[:,1]
        ys = c[:,0]
        xs = xs - rot_im.shape[1]/2 - 0.5
        ys = ys - rot_im.shape[0]/2 - 0.5
        rot_xs = xs*np.cos(angle_rad) - ys*np.sin(angle_rad) +  rot_im.shape[1]/2  + 0.5
        rot_ys = xs*np.sin(angle_rad) + ys*np.cos(angle_rad) +  rot_im.shape[0]/2  + 0.5
        return np.array((rot_ys,rot_xs)).T
    def applyToImage(nim):
        """Apply the calculated rotation to another image."""
        return rotate(nim,-angle_rad*180/np.pi,resize=False,preserve_range=True) 
    
    return rot_im, applyToPoints,applyToImage


def extrapolate_point_pair(pt1,pt2,frac_pre,frac_post):
    """
    Extrapolate a line segment defined by two points.
    
    Parameters
    ----------
    pt1 : array-like
        First point (y, x).
    pt2 : array-like
        Second point (y, x).
    frac_pre : float
        Fraction of the segment length to extend before pt1.
    frac_post : float
        Fraction of the segment length to extend after pt2.
        
    Returns
    -------
    extrapolated : ndarray, shape (2, 2)
        Array containing the two extrapolated points [adj1, adj2].
    """
    d0 = pt2[0]-pt1[0]
    d1 = pt2[1]-pt1[1]
    
    adj1 = [pt1[0]-d0*frac_pre,pt1[1]-d1*frac_pre]
    adj2 = [pt2[0]+d0*frac_post,pt2[1]+d1*frac_post]

    return np.array((adj1,adj2))


def interpolate_points(points, values, output_shape, method='nearest', fill_value=np.nan):
    """
    Interpolate scattered point data onto a regular pixel grid.

    Parameters
    ----------
    points : ndarray, shape (N, 2)
        Point positions in (y, x) format.
    values : ndarray, shape (N,)
        Scalar values at each point.
    output_shape : tuple of int
        (height, width) of the output grid.
    method : str, optional
        Interpolation method: 'nearest', 'linear', or 'cubic'. Default 'nearest'.
    fill_value : float, optional
        Value for points outside the convex hull of input points. Default np.nan.

    Returns
    -------
    interp : ndarray, shape output_shape
        Interpolated values on the regular grid.
    """
    xi = np.moveaxis(
        np.array(np.meshgrid(
            np.arange(output_shape[0]),
            np.arange(output_shape[1]),
            indexing='ij')),
        [0, 1, 2], [2, 0, 1])
    interp = interpolate.griddata(points, values.ravel(), xi,
                                  method=method, fill_value=fill_value)
    return interp


def aggregate_points(positions, values, sigma, query_points=None,
                     image_shape=None, agg_func=None, cutoff=3.0,
                     fill=np.nan):
    """
    Spatially aggregate values using a Gaussian-weighted window.

    Computes a weighted aggregation of nearby point values. When image_shape
    is provided, the result is rasterized onto a pixel grid; otherwise it is
    returned at the query points (or the input positions).

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Data point positions in (y, x) format.
    values : ndarray, shape (N,) or (N, 2)
        Scalar or vector values at each position.
    sigma : float
        Gaussian window standard deviation in pixels.
    query_points : ndarray, shape (M, 2), optional
        Points at which to evaluate the aggregate. Ignored when image_shape
        is provided. Defaults to positions.
    image_shape : tuple of int, optional
        If provided, rasterize the result onto a (height, width) grid.
    agg_func : callable, optional
        Function(values, weights) -> aggregated value. Defaults to
        Gaussian-weighted mean.
    cutoff : float, optional
        Window radius in units of sigma. Default 3.0.
    fill : float, optional
        Fill value for locations with no data in range. Default np.nan.

    Returns
    -------
    result : ndarray
        If image_shape is None: shape (M,) or (M, 2).
        If image_shape is provided: shape (H, W) or (H, W, 2).
    """
    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float)
    is_vector = values.ndim == 2

    if image_shape is not None and agg_func is None:
        return _aggregate_splat(positions, values, sigma, image_shape,
                                cutoff, fill, is_vector)

    if image_shape is not None:
        grid_y, grid_x = np.mgrid[0:image_shape[0], 0:image_shape[1]]
        query_points = np.column_stack([grid_y.ravel(), grid_x.ravel()])

    if query_points is None:
        query_points = positions

    result = _aggregate_gather(positions, values, sigma, query_points,
                               agg_func, cutoff, is_vector)

    if image_shape is not None:
        if is_vector:
            result = result.reshape(image_shape[0], image_shape[1],
                                    values.shape[1])
        else:
            result = result.reshape(image_shape)

    return result


def _aggregate_gather(positions, values, sigma, query_points, agg_func,
                      cutoff, is_vector):
    if agg_func is None:
        def agg_func(v, w):
            w_sum = w.sum()
            if w_sum == 0:
                return np.full(v.shape[1:], np.nan) if is_vector else np.nan
            if is_vector:
                return (v * w[:, np.newaxis]).sum(axis=0) / w_sum
            return (v * w).sum() / w_sum

    tree = cKDTree(positions)
    radius = cutoff * sigma
    result = np.empty((len(query_points),) + values.shape[1:])

    for i, qp in enumerate(query_points):
        idxs = tree.query_ball_point(qp, radius)
        if len(idxs) == 0:
            result[i] = np.full(values.shape[1:], np.nan) if is_vector else np.nan
            continue
        idxs = np.array(idxs)

        v = values[idxs]
        if is_vector:
            valid = ~np.any(np.isnan(v), axis=1)
        else:
            valid = ~np.isnan(v)

        if not np.any(valid):
            result[i] = np.full(values.shape[1:], np.nan) if is_vector else np.nan
            continue

        v = v[valid]
        dists = np.linalg.norm(positions[idxs[valid]] - qp, axis=1)
        weights = np.exp(-0.5 * (dists / sigma) ** 2)
        result[i] = agg_func(v, weights)

    return result


def _aggregate_splat(positions, values, sigma, image_shape, cutoff, fill,
                     is_vector):
    H, W = image_shape
    if is_vector:
        numer = np.zeros((H, W, values.shape[1]))
    else:
        numer = np.zeros((H, W))
    denom = np.zeros((H, W))
    radius = cutoff * sigma

    for i, (py, px) in enumerate(positions):
        if is_vector:
            if np.any(np.isnan(values[i])):
                continue
        else:
            if np.isnan(values[i]):
                continue

        y_lo = max(int(np.floor(py - radius)), 0)
        y_hi = min(int(np.ceil(py + radius)), H - 1)
        x_lo = max(int(np.floor(px - radius)), 0)
        x_hi = min(int(np.ceil(px + radius)), W - 1)

        if y_lo > y_hi or x_lo > x_hi:
            continue

        yy = np.arange(y_lo, y_hi + 1)
        xx = np.arange(x_lo, x_hi + 1)
        wy = np.exp(-0.5 * ((yy - py) / sigma) ** 2)
        wx = np.exp(-0.5 * ((xx - px) / sigma) ** 2)
        w_patch = np.outer(wy, wx)

        denom[y_lo:y_hi+1, x_lo:x_hi+1] += w_patch
        if is_vector:
            numer[y_lo:y_hi+1, x_lo:x_hi+1] += w_patch[..., np.newaxis] * values[i]
        else:
            numer[y_lo:y_hi+1, x_lo:x_hi+1] += w_patch * values[i]

    mask = denom > 0
    if is_vector:
        result = np.full((H, W, values.shape[1]), fill)
        result[mask] = numer[mask] / denom[mask, np.newaxis]
    else:
        result = np.full((H, W), fill)
        result[mask] = numer[mask] / denom[mask]

    return result
