import numpy as np
from .. import util

# TODO: test with non-square
def create_gaussian_filter(shape,p,sigma):
	p = np.array(p)
	filt = np.zeros(shape)
	YY,XX = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))
	if len(p.shape)==2:
		for pt in np.array(p).T:
			filt = filt + np.reshape(util.func.gaussian_2d((YY,XX),1,pt[0],pt[1],sigma,sigma,0,0),shape)
	else:
		filt = np.reshape(util.func.gaussian_2d((YY,XX),1,p[0],p[1],sigma,sigma,0,0),shape)
	return filt

def fourier_filter(pattern_c,p,sigma):
	filt = create_gaussian_filter(pattern_c.shape,p,sigma)
	filtered_pattern_c = pattern_c * filt
	filtered_im_c = np.fft.ifft2(np.fft.ifftshift(filtered_pattern_c))

	return filtered_im_c, filtered_pattern_c, filt

def coarsening_length(size,sigma,c=4):
	return size / (2*np.pi*sigma) * np.sqrt(np.log(c))

def damp_complex_amplitude(array_c,level):
    return array_c * level / np.abs(array_c)

def mask_peaks_circle(pattern_c,p,levels,radius):
    masked_pattern_c = pattern_c.copy()
    II,JJ = np.meshgrid(np.arange(pattern_c.shape[0]),np.arange(pattern_c.shape[1]))
    for it,pt in enumerate(p):
        circle_mask = ((JJ-pt[0])**2 + (II-pt[1])**2 )< radius**2
        masked_pattern_c[circle_mask] = damp_complex_amplitude(masked_pattern_c[circle_mask], levels[it])
    masked_image = np.abs(np.fft.ifft2(np.fft.ifftshift(masked_pattern_c)))
    return masked_pattern_c, masked_image
