import numpy as np
from scipy.signal import medfilt2d


def phase_to_strain(phase, p, mask_threshold=None):
    """
    Calculate longitudinal and transverse strain components from spatially varying phase.
    
    Projections of the gradient of the phase along and transverse to a reference
    wavevector are assigned as the longitudinal and transverse components.

    Parameters
    ----------
    phase : ndarray
        Real valued 2D phase image.
    p : tuple
        Peak position (y, x) in the Fourier transform.
    mask_threshold : float, optional
        Threshold for masking strain values. If None, no masking/filtering is applied (default is None).

    Returns
    -------
    eps_longitudinal : ndarray
        longitudinal/parallel component of the strain
    eps_transverse : ndarray
        transvserse / orthogonal component of the strain

    Notes
    -----
    This function calculates compression and shear strain components from
    a real valued phase. Uses improved complex exponential differentiation.
    """
    # Calculate qref from p
    # qref is the vector from the center to the peak
    # p is (y, x), shape is (h, w)
    qref = np.array(p) - np.array(phase.shape)/2.0
    
    qx = qref[1]
    qy = qref[0]

    # Use robust gradient calculation
    # grad_phi[0] is y-derivative, grad_phi[1] is x-derivative
    grad_phi_y = _calculate_phase_gradient_complex(phase, 0)
    grad_phi_x = _calculate_phase_gradient_complex(phase, 1)

    # Normalized such that a shift of 2pi over 1 wavelength gives eps=1
    eps_longitudinal = -(grad_phi_y*qy + grad_phi_x*qx)/(qx**2+qy**2)*phase.shape[0]/(2*np.pi)
    eps_transverse = -(0.5)*(-grad_phi_y*qx + grad_phi_x*qy)/(qx**2+qy**2)*phase.shape[0]/(2*np.pi)

    if mask_threshold is not None:
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


def _calculate_phase_gradient_complex(phase, axis):
    """
    Calculate the derivative of a phase image using complex exponentials to avoid wrapping issues.

    Implemented based on Niels Cautaerts's TEMMETA package.
    
    Parameters
    ----------
    phase : ndarray
        Phase image.
    axis : int
        Axis along which to calculate the derivative (0 or 1).
        
    Returns
    -------
    grad : ndarray
        The derivative of the phase.
    """
    s = np.exp(1j * phase)
    # Calculate difference along axis
    d = np.diff(s, axis=axis)
    
    # Pad to maintain shape (simple padding at the end)
    if axis == 0:
        d = np.pad(d, ((0, 1), (0, 0)), mode='edge')
    else:
        d = np.pad(d, ((0, 0), (0, 1)), mode='edge')
        
    # The derivative is Imag(conjugate(s) * d)
    # This is equivalent to angle difference for small steps
    grad = (np.conj(s) * d).imag
    return grad


def calculate_strain_tensor(phase1, phase2, p1, p2, mask_threshold=None):
    """
    Calculate the full 2D strain tensor from two phase images.

    Implemented based on Niels Cautaerts's TEMMETA package.
    
    Parameters
    ----------
    phase1 : ndarray
        Phase image corresponding to reflection 1.
    phase2 : ndarray
        Phase image corresponding to reflection 2.
    p1 : tuple
        Peak position (y, x) for reflection 1 in the Fourier transform.
    p2 : tuple
        Peak position (y, x) for reflection 2 in the Fourier transform.
    mask_threshold : float, optional
         Threshold for masking out outlier strain values. If None, no masking/filtering (default is None).

    Returns
    -------
    exx : ndarray
        Epsilon_xx strain component.
    eyy : ndarray
        Epsilon_yy strain component.
    exy : ndarray
        Epsilon_xy shear strain component (symmetric part).
    oxy : ndarray
        Omega_xy rotation component (antisymmetric part).
    """
    shape = phase1.shape
    h, w = shape
    
    # Calculate g vectors in units of 1/pixel    
    gx1 = (p1[1] - w/2) / w
    gy1 = (p1[0] - h/2) / h
    gx2 = (p2[1] - w/2) / w
    gy2 = (p2[0] - h/2) / h
    
    # Calculate derivatives using robust complex exponential method
    # dP1/dx is derivative along axis 1 (cols)
    # dP1/dy is derivative along axis 0 (rows)
    dP1dx = _calculate_phase_gradient_complex(phase1, 1)
    dP1dy = _calculate_phase_gradient_complex(phase1, 0)
    dP2dx = _calculate_phase_gradient_complex(phase2, 1)
    dP2dy = _calculate_phase_gradient_complex(phase2, 0)
    
    # Solve for lattice basis vectors
    # G = [[gx1, gx2], [gy1, gy2]]
    # A = inv(G^T)
    G = np.array([[gx1, gx2],
                  [gy1, gy2]])
    
    try:
        G_inv_T = np.linalg.inv(G.T)
        a1x, a2x = G_inv_T[0]
        a1y, a2y = G_inv_T[1]
    except np.linalg.LinAlgError:
        print("Warning: g-vectors are collinear or singular matrix.")
        return np.zeros_like(phase1), np.zeros_like(phase1), np.zeros_like(phase1), np.zeros_like(phase1)

    # Calculate displacement gradient tensor components
    # The formula from Hytch et al. (1998):
    # e_ij = -1/(2*pi) * sum(a_ki * dP_k/dx_j)
    
    exx_raw = -1/(2*np.pi) * (a1x * dP1dx + a2x * dP2dx) # du/dx
    eyx_raw = -1/(2*np.pi) * (a1y * dP1dx + a2y * dP2dx) # dv/dx
    exy_raw = -1/(2*np.pi) * (a1x * dP1dy + a2x * dP2dy) # du/dy
    eyy_raw = -1/(2*np.pi) * (a1y * dP1dy + a2y * dP2dy) # dv/dy
    
    # Calculate strain and rotation
    exx = exx_raw
    eyy = eyy_raw
    exy = 0.5 * (exy_raw + eyx_raw) # Symmetric shear strain
    oxy = 0.5 * (exy_raw - eyx_raw) # Rotation (antisymmetric)
    
    # Apply median filtering to remove outliers
    if mask_threshold is not None:
        for strain_map in [exx, eyy, exy, oxy]:
            mask = np.abs(strain_map) > mask_threshold
            # Initial fill for outliers
            strain_temp = np.where(mask, 0, strain_map)
            # Iterative median filtering
            for _ in range(3):
                strain_filt = medfilt2d(strain_temp, kernel_size=5)
                strain_temp = np.where(mask, strain_filt, strain_temp)
            np.copyto(strain_map, strain_temp)

    return exx, eyy, exy, oxy