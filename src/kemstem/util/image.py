import numpy as np
from scipy.ndimage import gaussian_filter

def faux_highpass_filter(image,lowpass_sigma):
	return image - gaussian_filter(image,lowpass_sigma)


def hann_filter(image):
    nx,ny = image.shape
    rx,ry = np.meshgrid(np.arange(nx),np.arange(ny))
    rx = rx.T
    ry = ry.T
    mask_realspace = (np.sin(np.pi*rx/nx)*np.sin(np.pi*(ry/ny)))**2
    fim = image*mask_realspace
    fft = np.fft.fft2(fim)
    return (fim, fft)

# www.roberthovden.com/tutorial/2015_fftartifacts.html
def periodic_plus_smooth_decomposition(im):
    [rows,cols] = np.shape(im) 
    #Compute boundary conditions 
    s = np.zeros( np.shape(im) ) 
    s[0,0:] = im[0,0:] - im[rows-1,0:] 
    s[rows-1,0:] = -s[0,0:] 
    s[0:,0] = s[0:,0] + im[0:,0] - im[:,cols-1] 
    s[0:,cols-1] = s[0:,cols-1] - im[0:,0] + im[:,cols-1] 
    #Create grid for computing Poisson solution 
    [cx, cy] = np.meshgrid(2*np.pi*np.arange(0,cols)/cols, 2*np.pi*np.arange(0,rows)/rows) 

    #Generate smooth component from Poisson Eq with boundary condition 
    D = (2*(2 - np.cos(cx) - np.cos(cy))) 
    D[0,0] = np.inf # Enforce 0 mean & handle div by zero # -?
    S = np.fft.fft2(s)/D 
    #S[0,0] = 0 # +?

    P = np.fft.fft2(im) - S # FFT of periodic component 


    return(P,S)
