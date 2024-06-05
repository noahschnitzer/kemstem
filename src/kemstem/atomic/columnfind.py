from .. import util
import numpy as np
import skimage
#from scipy.ndimage import gaussian_filter, label,center_of_mass
import matplotlib.pyplot as plt

# based off stemtool
def find_columns(image, distance = 10, threshold = 0.1, deduplicate = False):
    image_norm = image - image.min()
    image_norm = image_norm / image_norm.max()
    thresh_mask = image_norm > threshold
    masked = (image_norm * thresh_mask) - threshold # max 1-threshold, min 0
    masked = masked / masked.max()
    peaks = skimage.feature.peak_local_max(masked,min_distance = distance)
    #peaks_mask = np.zeros_like(image,dtype=bool)
    #peaks_mask[tuple(peaks.T)] = True
    #peak_labels = label(peaks_mask)[0]
    #merged_peaks = center_of_mass(peaks_mask,peak_labels,range(1,np.max(peak_labels)+1))
    
    #peaks = np.array(merged_peaks)
    return peaks


def refine_columns(image, columns0, window_dimension=5,store_fits=True, remove_unfit = True):
    return util.general.gaussian_fit_peaks(image, columns0, window_dimension=window_dimension,store_fits=store_fits, remove_unfit = remove_unfit)