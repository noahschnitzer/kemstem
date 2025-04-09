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
    qx : float
        Reference wavevector x component
    qy : float
        Reference wavevector y component
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