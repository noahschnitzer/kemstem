import numpy as np
from .. import util
from . import fourierfilter

def create_references(p,shape):
    """
    Create reference cosine and sine reference waves for phase lock-in.

    Parameters
    ----------
    p : tuple
        Peak position (y, x) in the Fourier transform.
    shape : tuple
        Shape of the image (height, width).

    Returns
    -------
    cosRef : ndarray
        Real valued sine reference wave
    sinRef : ndarray
        Real valued cosine reference wave
    p : tuple
        Reference wavevector (y, x).
    """

    xsize = shape[1]
    ysize = shape[0]
    x = p[1]
    y = p[0]
    qx=2*np.pi*(xsize/2.- x)/xsize
    qy=2*np.pi*(ysize/2.- y)/ysize

    YY,XX = np.meshgrid(np.arange(ysize),np.arange(xsize),indexing='ij')
    cosRef = np.cos(qx*XX+qy*YY)
    sinRef = np.sin(qx*XX+qy*YY)

    return cosRef, sinRef, qx, qy


def phaselock(filtered_im_c,p,sigma):
    """
    Perform phase lock in on a fourier filtered complex image.

    Parameters
    ----------
    filtered_im_c : ndarray
        Fourier filtered complex image.
    p : tuple
        Peak position (y, x) in the Fourier transform.
    sigma : float
        Standard deviation for the lock-in Gaussian low-pass filter.

    Returns
    -------
    phase : ndarray
        Real valued phase - range of 0 to 2π.
    """

    cosRef, sinRef, qx, qy =create_references(p, filtered_im_c.shape)
    
    #Multiply reference images with Fourier filtered images
    outA=cosRef*filtered_im_c
    outB=sinRef*filtered_im_c

    #Take fft of output channels
    fftoutA=np.fft.fftshift(np.fft.fft2((outA)))
    fftoutB=np.fft.fftshift(np.fft.fft2((outB)))

    #Low-pass filter the product images
    low_pass=fourierfilter.create_gaussian_filter(filtered_im_c.shape,(filtered_im_c.shape[0]/2,filtered_im_c.shape[1]/2),sigma)
    ifftA=np.real(np.fft.ifft2(np.fft.ifftshift(fftoutA*low_pass)))#get_ifft(fftoutA*low_pass)
    ifftB=np.real(np.fft.ifft2(np.fft.ifftshift(fftoutB*low_pass)))#get_ifft(fftoutB*low_pass)

    #Get the phase shift (distortions from reference images)
    theta_lf=np.arctan2(ifftB, ifftA)
    return theta_lf % (2*np.pi)


def gpa(filtered_im_c, p):
    """
    Calculate the Geometric Phase from a fourier filtered complex image.
    
    This function subtracts the carrier frequency corresponding to peak `p` 
    from the phase of the filtered image.

    Implemented based on Niels Cautaerts's TEMMETA package.

    Parameters
    ----------
    filtered_im_c : ndarray
        Fourier filtered complex image.
    p : tuple
        Peak position (y, x) in the Fourier transform used for filtering.

    Returns
    -------
    gpa_phase : ndarray
        Real valued phase image.
    """
    # Calculate carrier frequency components
    cosRef, sinRef, qx, qy = create_references(p, filtered_im_c.shape)
    
    # Create coordinate grids
    YY, XX = np.meshgrid(np.arange(filtered_im_c.shape[0]), 
                         np.arange(filtered_im_c.shape[1]), indexing='ij')
    
    # Calculate phase ramp 2*pi*g.r
    phase_ramp = qx*XX + qy*YY
    
    # Extract phase from filtered image
    raw_phase = np.angle(filtered_im_c)
    
    # Subtract carrier frequency phase
    gpa_phase = raw_phase + phase_ramp
    
    return gpa_phase