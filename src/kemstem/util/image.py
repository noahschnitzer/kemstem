import numpy as np
from scipy.ndimage import gaussian_filter

def faux_highpass_filter(image,lowpass_sigma):
    """
    Apply a pseudo high-pass filter to an image.
    
    Creates a high-pass filtered version of the input image by subtracting
    a Gaussian-blurred version from the original image.
    
    Parameters
    ----------
    image : ndarray
        Input image to be filtered.
    lowpass_sigma : float or sequence of floats
        Standard deviation for Gaussian kernel.
    
    Returns
    -------
    ndarray
        High-pass filtered image (original minus low-pass filtered version).
    """
	return image - gaussian_filter(image,lowpass_sigma)


def hann_filter(image):
    """
    Apply a 2D Hann window filter to an image and compute its FFT.
    
    Multiplies the image by a 2D Hann window (sin²) to reduce edge effects
    in the Fourier transform.
    
    Parameters
    ----------
    image : ndarray
        Input image to be filtered, 2D array.
    
    Returns
    -------
    fim : ndarray
        Image after applying the Hann window.
    fft : ndarray
        2D Fourier transform of the windowed image.
    
    Notes
    -----
    The Hann window is created using sin²(πx/nx)sin²(πy/ny) where nx, ny
    are the image dimensions. This helps reduce spectral leakage in the
    Fourier transform.
    """

    nx,ny = image.shape
    rx,ry = np.meshgrid(np.arange(nx),np.arange(ny))
    rx = rx.T
    ry = ry.T
    mask_realspace = (np.sin(np.pi*rx/nx)*np.sin(np.pi*(ry/ny)))**2
    fim = image*mask_realspace
    fft = np.fft.fft2(fim)
    return (fim, fft)

def periodic_plus_smooth_decomposition(im):
    """
    Decompose an image into periodic and smooth components.
    
    Implements the periodic-plus-smooth decomposition as implemented at
    www.roberthovden.com/tutorial/2015_fftartifacts.html. This decomposition
    helps reduce artifacts in Fourier transforms of non-periodic images by
    separating the image into a periodic component and a smooth component.
    See: Hovden, R., Jiang, Y., Xin, H. L., & Kourkoutis, L. F. (2015). 
    Periodic artifact reduction in Fourier transforms of full field atomic 
    resolution images. Microscopy and Microanalysis, 21(2), 436-441.
    https://doi.org/10.1017/S1431927614014639
    
    Parameters
    ----------
    im : ndarray
        Input image to be decomposed, 2D array.
    
    Returns
    -------
    P : ndarray
        FFT of the periodic component.
    S : ndarray
        FFT of the smooth component.
    
    Notes
    -----
    The decomposition solves a Poisson equation with boundary conditions
    determined by the discontinuities at the image edges. The smooth
    component captures these discontinuities, leaving the periodic
    component better suited for Fourier analysis.
    
    The implementation enforces zero mean in the frequency domain by
    setting the DC component (0,0) appropriately.
    """
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

    P = np.fft.fft2(im) - S # FFT of periodic component 


    return(P,S)
