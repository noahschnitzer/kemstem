import numpy as np
from scipy.signal import medfilt2d


def phase_to_strain(phase, qref, mask_threshold=2.):
    """
    Calculate longitudinal and transverse strain components from spatially varying phase.
    
    Projections of the gradient of the phase along and transverse to a reference
    wavevector are assigned as the longitudinal and transverse components.

    Typically, the appropriate reference wavevector is the vector from the filtered 
    peak (p) in the Fourier transform to its center, e.g.:
    `qref = p - np.array(image.shape)/2`
    
    Parameters
    ----------
    phase : ndarray
        Real valued 2D phase image.
    qref : tuple
        Reference wavevector (y, x).
    mask_threshold : float, optional
        Threshold for masking strain values (default is 2.0).

    Returns
    -------
    eps_longitudinal : ndarray
        longitudinal/parallel component of the strain
    eps_transverse : ndarray
        transvserse / orthogonal component of the strain

    Notes
    -----
    This function calculates compression and shear strain components from
    a real valued phase. A median filter is applied to remove artifacts from
    phase wrapping. 
    """
    qx = qref[1]
    qy = qref[0]

    grad_phi = np.gradient(phase)

    # Normalized such that a shift of 2pi over 1 wavelength gives eps=1
    eps_longitudinal = -(grad_phi[0]*qy + grad_phi[1]*qx)/(qx**2+qy**2)*phase.shape[0]/(2*np.pi)
    eps_transverse = -(0.5)*(-grad_phi[0]*qx + grad_phi[1]*qy)/(qx**2+qy**2)*phase.shape[0]/(2*np.pi)

    mask = np.abs(eps_transverse)>mask_threshold
    eps_temp = np.where(mask, 0, eps_transverse)
    for i in range(3):
        eps_temp = medfilt2d(eps_temp, kernel_size=5)
        eps_transverse = np.where(mask, eps_temp, eps_transverse)


    mask = np.abs(eps_longitudinal)>mask_threshold
    eps_temp = np.where(mask, 0, eps_longitudinal)
    for i in range(3):
        eps_temp = medfilt2d(eps_temp, kernel_size=5)
        eps_longitudinal = np.where(mask, eps_temp, eps_longitudinal)
    return eps_longitudinal, eps_transverse