from .. import util
import numpy as np
from sklearn.cluster import KMeans


def get_neighborhood(columns, centerpoints,k=20):
    """
    Get the neighborhoods of columns around specified center points.

    Parameters
    ----------
    columns : ndarray, shape (n,2)
        Array of all column positions (y,x).
    centerpoints : ndarray, shape (m,2)
        Array of center point positions (y,x).
    k : int, optional
        Number of nearest neighbors to consider (default is 20).

    Returns
    -------
    ndarray
        Array of vectors from neighbors to center points (m*k, 2).
    """


    surr_cols = util.point.get_nearest_points(columns,centerpoints,k=k)[0]
    rel = (np.swapaxes(surr_cols,0,1) - centerpoints).reshape(-1,2)
    
    return rel

def cluster_neighbors(neighborhood, n_clusters=40):
    """
    Cluster the neighborhood points using sklearn.cluster.KMeans.

    Parameters
    ----------
    neighborhood : ndarray, shape (n,2)
        Vectors to local neighborhoods (pair correlation function), (y,x).
    n_clusters : int, optional
        Number of clusters to form (default is 40).

    Returns
    -------
    ndarray
        Array of cluster centers (n_clusters, 2).
    """

    kmeans = KMeans(n_clusters=n_clusters,copy_x=True,random_state=1337).fit(neighborhood) # (n,2)
    return kmeans.cluster_centers_

def get_vector_to_neighbors(columns,origins,guess_vector, threshold=5):
    """
    Find the vectors from origin points to the column closest to the guess_vector.

    E.g., for an origin o, column c and guess_vector g, will return c-o for 
    the c nearest to o+g.

    Parameters
    ----------
    columns : ndarray, shape (n,2)
        Coordinates of columns of interest (y,x)
    origins : ndarray, shape (n,2)
        Coordinates of origin points (y,x)
    guess_vector : ndarray
        Guess vector from which the nearest column is found (2,).
    threshold : float, optional
        Maximum norm of found vs. guessed vector (default is 5).

    Returns
    -------
    ndarray
        Array of vectors from origins to their nearest neighbors (M, 2).
        NaN values indicate no neighbor found within the threshold.
    """

    match_col,match_idx,dist = util.point.get_nearest_points(columns,origins+guess_vector,k=1)
    threshold_mask = dist > threshold
    match_col[threshold_mask,:] = np.nan
    return match_col - origins



def get_connected_columns(columns, origin, vector, tolerance,bidirectional=True):
    err = 0
    current_c = origin
    chain = []
    while err < tolerance:
        chain.append(current_c)
        current_c,idx,err = util.get_nearest_points(columns,current_c + vector, k=1)
    
    origin_idx = 0
    if bidirectional:
        err = 0
        current_c = origin
        rev_chain = []
        while err < tolerance:
            rev_chain.append(current_c)
            current_c,idx,err = util.get_nearest_points(columns,current_c - vector, k=1)
        chain = rev_chain[:0:-1] + chain
        origin_idx = len(rev_chain[:0:-1])
    return np.array(chain), origin_idx


def index_lattice(columns,origin,v1,v2,tolerance=1,max_n1=20,max_n2=20,cumulative_adjust=False):
    indices = np.array(np.meshgrid(np.arange(-max_n1,max_n1+1),np.arange(-max_n2,max_n2+1),indexing='ij'))
    indices = np.moveaxis(indices,0,-1)
    if cumulative_adjust:
        chain1,origin_idx = get_connected_columns(columns,origin,v1,tolerance,bidirectional=True)
        chain1 = chain1[origin_idx-max_n1:origin_idx+max_n1+1]
        
        chains2 = []
        for c2start in tqdm(chain1):
            chain2,origin_idx = get_connected_columns(columns,c2start,v2,tolerance,bidirectional=True)
            chain2 = chain2[origin_idx-max_n2:origin_idx+max_n2+1]
            chains2.append(chain2)
        indexed_cols = np.array(chains2)
        return indices,indexed_cols
    else:
        compare_points = np.zeros(indices.shape)
        compare_points[:,:,0] = origin[0] + indices[:,:,0]*v1[0] + indices[:,:,1]*v2[0]
        compare_points[:,:,1] = origin[1] + indices[:,:,0]*v1[1] + indices[:,:,1]*v2[1]
        
        indexed_cols,_,dists = util.get_nearest_points(columns,compare_points.reshape(-1,2), k=1)
        indexed_cols[dists > tolerance,:] = np.nan
        indexed_cols = indexed_cols.reshape(compare_points.shape)

        return indices,indexed_cols,compare_points