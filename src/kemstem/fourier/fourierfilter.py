import numpy as np
from tqdm import tqdm
from .. import util

def create_gaussian_filter(shape,p,sigma):
    """
    Generate a Gaussian filter.
    
    Creates a real image of 2D Gaussians with standard deviation sigma at the points p.
    
    Parameters
    ----------
    shape : tuple
        Shape of the filter (height, width).
    p : ndarray, shape (2,) or (n,2)
        Peak position(s) (y, x) in the Fourier transform.
    sigma : float
        Standard deviation of the Gaussian filter.
    
    Returns
    -------
    filt : ndarray
        Image with 2D Gaussians
    """
    
    p = np.array(p)
    filt = np.zeros(shape)
    YY,XX = np.meshgrid(np.arange(shape[0]),np.arange(shape[1]),indexing='ij')
    if len(p.shape)==2:
        for pt in np.array(p).T:
            filt = filt + np.reshape(util.func.gaussian_2d((YY,XX),1,pt[1],pt[0],sigma,sigma,0,0),shape)
    else:
        filt = np.reshape(util.func.gaussian_2d((YY,XX),1,p[1],p[0],sigma,sigma,0,0),shape)
    return filt

def fourier_filter(pattern_c,p,sigma):
    """
    Apply a Gaussian filter to a complex array.

    Parameters
    ----------
    pattern_c : ndarray
        Complex input pattern (Typically Fourier transform of an image).
    p : ndarray, shape (2,) or (n,2)
        Peak position(s) (y, x) in the Fourier transform.
    sigma : float
        Standard deviation of the Gaussian filter.

    Returns
    -------
    filtered_im_c : ndarray
        Complex inverse fourier transformed, filtered pattern.
    filtered_pattern_c : ndarray
        Complex filtered pattern
    filt : ndarray
        The filter used specified by p and sigma.
    """

    filt = create_gaussian_filter(pattern_c.shape,p,sigma)
    filtered_pattern_c = pattern_c * filt
    filtered_im_c = np.fft.ifft2(np.fft.ifftshift(filtered_pattern_c))

    return filtered_im_c, filtered_pattern_c, filt

def coarsening_length(size,sigma,c=4):
    """
    Calculate the real space coarsening length for a given 
    Gaussian filter applied in reciprocal space to an image of shape size
    with standard deviation sigma.
    
    Parameters
    ----------
    size : int
        Size of the image in pixels.
    sigma : float
        Standard deviation of the Gaussian filter in reciprocal space pixels.
    c : float, optional
        Constant factor (default is 4) setting the cutoff width. 
        c=4 corresponds to width at half maximum.
    
    Returns
    -------
    float
        Real space coarsening radius, in pixels.
    """
    
    return size / (2*np.pi*sigma) * np.sqrt(np.log(c))

def damp_complex_amplitude(array_c,level):
    """
    Damp the amplitude of a complex array to a specified level.
    
    Parameters
    ----------
    array_c : ndarray
        Complex input array.
    level : float
        Target amplitude level.
    
    Returns
    -------
    damped : ndarray
        Damped complex array.
    """
    
    return array_c * level / np.abs(array_c)

def mask_peaks_circle(pattern_c,p,levels,radius):
    """
    Mask peaks in a complex array by setting a circular area around each to a specified amplitude.
    
    Parameters
    ----------
    pattern_c : ndarray
        Complex input pattern (Fourier transform of an image).
    p : ndarray, shape (n,2)
        Peak positions (y, x) in the Fourier transform.
    levels : ndarray
        Target amplitude levels for each peak (typically the background level).
    radius : float
        Radius of the circular mask.
    
    Returns
    -------
    masked_pattern_c : ndarray
        Masked complex pattern
    masked_image : ndarray
        Absolute value of the inverse transform of the masked pattern. Real valued.
    """
    
    masked_pattern_c = pattern_c.copy()
    YY,XX = np.meshgrid(np.arange(pattern_c.shape[0]),np.arange(pattern_c.shape[1]),indexing='ij')
    for it,pt in tqdm(enumerate(p)):
        circle_mask = ((XX-pt[1])**2 + (YY-pt[0])**2 )< radius**2
        masked_pattern_c[circle_mask] = damp_complex_amplitude(masked_pattern_c[circle_mask], levels[it])
    masked_image = np.abs(np.fft.ifft2(np.fft.ifftshift(masked_pattern_c)))
    return masked_pattern_c, masked_image
