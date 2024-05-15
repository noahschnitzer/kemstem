import numpy as np
from .. import util

# TODO: test with non-square
def create_gaussian_filter(shape,p,sigma):
	p = np.array(p)
	filt = np.zeros(shape)
	YY,XX = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]))
	if len(p.shape)==2:
		for pt in np.array(p).T:
			filt = filt + np.reshape(util.func.gaussian_2d((YY,XX),1,pt[1],pt[0],sigma,sigma,0,0),shape)
	else:
		filt = np.reshape(util.func.gaussian_2d((YY,XX),1,p[1],p[0],sigma,sigma,0,0),shape)
	return filt

def fourier_filter(pattern_c,p,sigma):
	filt = create_gaussian_filter(pattern_c.shape,p,sigma)
	filtered_pattern_c = pattern_c * filt
	filtered_im_c = np.fft.ifft2(np.fft.ifftshift(filtered_pattern_c))

	return filtered_im_c, filtered_pattern_c, filt

def coarsening_length(size,sigma,c=4):
	return size / (2*np.pi*sigma) * np.sqrt(np.log(c))