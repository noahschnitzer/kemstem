from .. import util
import numpy as np
from sklearn.cluster import KMeans


def get_neighborhood(columns, centerpoints,k=20):
    surr_cols = util.point.get_nearest_points(columns,centerpoints,k=k)[0]
    rel = (np.swapaxes(surr_cols,0,1) - surr_cols[:,0,:]).reshape(-1,2)
    return rel

def cluster_neighbors(neighborhood, n_clusters=40):
    kmeans = KMeans(n_clusters=n_clusters,copy_x=True,random_state=1337,n_init='auto').fit(neighborhood) # (n,2)
    return kmeans.cluster_centers_

def get_vector_to_neighbors(columns,origins,guess_vector, threshold=5):
    match_col,match_idx,dist = util.point.get_nearest_points(columns,origins+guess_vector,k=1)
    threshold_mask = dist > threshold
    match_col[threshold_mask,:] = np.nan
    return match_col - origins