import numpy as np
from scipy.ndimage import gaussian_filter

def faux_highpass_filter(image,lowpass_sigma):
	return image - gaussian_filter(image,lowpass_sigma)