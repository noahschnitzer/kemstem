import numpy as np
from scipy.signal import medfilt2d


def phase_to_strain(phase, qx, qy, mask_threshold=2.):
    grad_phi = np.gradient(phase)

    # Normalized such that a shift of 2pi over 1 wavelength gives eps=1
    eps_compression = -(grad_phi[0]*qy + grad_phi[1]*qx)/(qx**2+qy**2)*phase.shape[0]/(2*np.pi)
    eps_shear = -(0.5)*(-grad_phi[0]*qx + grad_phi[1]*qy)/(qx**2+qy**2)*phase.shape[0]/(2*np.pi)

    mask = np.abs(eps_shear)>mask_threshold
    eps_temp = np.where(mask, 0, eps_shear)
    for i in range(3):
        eps_temp = medfilt2d(eps_temp, kernel_size=5)
        eps_shear = np.where(mask, eps_temp, eps_shear)


    mask = np.abs(eps_compression)>mask_threshold
    eps_temp = np.where(mask, 0, eps_compression)
    for i in range(3):
        eps_temp = medfilt2d(eps_temp, kernel_size=5)
        eps_compression = np.where(mask, eps_temp, eps_compression)
    return eps_compression, eps_shear