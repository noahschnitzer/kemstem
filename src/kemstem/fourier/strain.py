import numpy as np
from scipy.signal import medfilt2d


def phase_to_strain(phase, p, mask_threshold=None):
    """
    Calculate longitudinal and transverse strain components from spatially varying phase.
    
    Projections of the gradient of the phase along and transverse to a reference
    wavevector g are assigned as the longitudinal and transverse components.

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
        Normal strain along g (du_g/dg), i.e. the relative change in lattice
        plane spacing.
    eps_transverse : ndarray
        Gradient of the displacement along g, taken transverse to g (du_g/dt).
        To first order this is the local rotation of the g lattice planes in
        radians; positive is clockwise as displayed (y down). It combines shear
        and rotation (eps_gt + omega) and is not the normal strain
        perpendicular to g, which cannot be measured from a single reflection.

    Notes
    -----
    A single reflection only measures the displacement component along g, u_g.
    The transverse direction t is g rotated 90 degrees counterclockwise as
    displayed, t = (gy, -gx) in (x, y) components. Separating shear from
    rotation requires a second reflection, see `calculate_strain_tensor`.
    Uses improved complex exponential differentiation.
    """
    # Reference g vector in cycles/pixel, from the center to the peak
    # p is (y, x), shape is (h, w)
    h, w = phase.shape
    gy = (p[0] - h//2) / h
    gx = (p[1] - w//2) / w

    # Use robust gradient calculation
    # grad_phi[0] is y-derivative, grad_phi[1] is x-derivative
    grad_phi_y = _calculate_phase_gradient_complex(phase, 0)
    grad_phi_x = _calculate_phase_gradient_complex(phase, 1)

    # Normalized such that a shift of 2pi over 1 wavelength gives eps=1
    eps_longitudinal = -(grad_phi_y*gy + grad_phi_x*gx)/(gx**2+gy**2)/(2*np.pi)
    eps_transverse = -(-grad_phi_y*gx + grad_phi_x*gy)/(gx**2+gy**2)/(2*np.pi)

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


def calculate_strain_tensor(phase1, phase2, p1, p2, mask_threshold=None, basis='g1', angle=0.0):
    """
    Calculate the full 2D strain tensor from two phase images.

    The tensor is expressed in an orthonormal frame (x', y') selected with
    `basis` and `angle`. The x' axis points along the reference direction set
    by `basis`, rotated by `angle`, and y' is x' rotated by +90 degrees. Angles
    are measured from +x toward +y, i.e. clockwise as displayed (y down).

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
    basis : {'g1', 'g2', 'image'}, optional
        Reference direction for the x' axis (default is 'g1'):
        'g1' -- along the reflection 1 wavevector, i.e. normal to the g1 lattice planes.
        'g2' -- along the reflection 2 wavevector, i.e. normal to the g2 lattice planes.
        'image' -- along the image +x axis (columns), so that y' is along +y (rows, down).
    angle : float, optional
        Rotation of the x' axis away from the `basis` reference direction, in
        radians, measured from +x toward +y (default is 0).

    Returns
    -------
    exx : ndarray
        Epsilon_x'x' normal strain component.
    eyy : ndarray
        Epsilon_y'y' normal strain component.
    exy : ndarray
        Epsilon_x'y' shear strain component (symmetric part, tensor shear).
    oxy : ndarray
        Omega_x'y' rotation component (antisymmetric part), independent of the frame.

    Notes
    -----
    With basis='g1' and angle=0, exx is the relative change in the g1 lattice
    plane spacing (equal to the longitudinal component from `phase_to_strain`
    for reflection 1) and eyy is the strain along the g1 planes, i.e. along the
    lattice vector a2. basis='g2' is the same with the reflections exchanged.
    The trace exx + eyy and the rotation oxy do not depend on the frame, and the
    result is the same for either of a pair of conjugate peaks.
    """
    shape = phase1.shape
    h, w = shape
    
    # Calculate g vectors in units of 1/pixel
    gx1 = (p1[1] - w//2) / w
    gy1 = (p1[0] - h//2) / h
    gx2 = (p2[1] - w//2) / w
    gy2 = (p2[0] - h//2) / h

    # Angle of the x' axis of the output frame
    if basis == 'g1':
        phi = np.arctan2(gy1, gx1) + angle
    elif basis == 'g2':
        phi = np.arctan2(gy2, gx2) + angle
    elif basis == 'image':
        phi = angle
    else:
        raise ValueError(f"Invalid argument: {basis!r}. Expected 'g1', 'g2', or 'image'.")
    
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

    # Rotate strain into the output frame, eps' = R eps R^T with x' = (c, s), y' = (-s, c)
    # oxy is unchanged by rotating the frame
    c, s = np.cos(phi), np.sin(phi)
    exx, eyy, exy = (c**2*exx + 2*c*s*exy + s**2*eyy,
                     s**2*exx - 2*c*s*exy + c**2*eyy,
                     c*s*(eyy - exx) + (c**2 - s**2)*exy)
    
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