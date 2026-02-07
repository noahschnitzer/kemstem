
import numpy as np
from skimage import color
import matplotlib.colors as mcolors

def vector_to_color(angles, magnitudes, method='lab', center='white', max_mag=None):
    """
    Convert vector angles and magnitudes to RGB colors.

    Parameters
    ----------
    angles : array-like
        Angles of vectors in radians.
    magnitudes : array-like
        Magnitudes of vectors.
    method : str, optional
        Color space method: 'lab' or 'hsv'. Default is 'lab'.
    center : str, optional
        Color at zero magnitude: 'white' or 'black'. Default is 'white'.
    max_mag : float, optional
        Maximum magnitude for normalization. If None, max(magnitudes) is used.

    Returns
    -------
    ndarray
        Array of RGB colors (N, 3).
    """
    angles = np.array(angles)
    magnitudes = np.array(magnitudes)
    
    if max_mag is None:
        max_mag = np.max(magnitudes) if len(magnitudes) > 0 else 1.0
        
    # Normalize magnitude [0, 1]
    norm_mag = np.clip(magnitudes / max_mag, 0, 1)

    if method == 'lab':
        return _lab_vector_colors(angles, norm_mag, center)
    elif method == 'hsv':
        return _hsv_vector_colors(angles, norm_mag, center)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'lab' or 'hsv'.")

def _lab_vector_colors(angles, norm_mag, center):
    """
    Generate colors using CIELAB color space.
    Hue is determined by angle.
    Chroma/Lightness depend on magnitude and center color.
    """
    # Lch (Lightness, Chroma, Hue) -> Lab -> RGB
    
    # Hue in radians
    # skimage.color.lch2lab expects hue in RADIANS, not degrees.
    hue_rad = angles
    
    # Chroma: 
    # Max chroma in Lab is around 100-140 depending on Hue/Lightness.
    # We'll target a reasonable max chroma for vibrant colors.
    max_chroma = 100
    
    if center == 'white':
        # White center: Low mag -> High Lightness, Low Chroma
        # High mag -> Balanced Lightness (50-60), High Chroma
        
        # Linear interpolation for Lightness: 100 (white) -> 55 (vibrant color)
        L = 100 - (45 * norm_mag)
        
        # Linear interpolation for Chroma: 0 (white/grey) -> max_chroma
        C = max_chroma * norm_mag
        
    elif center == 'black':
        # Black center: Low mag -> Low Lightness, Low Chroma
        # High mag -> Balanced Lightness (50-60), High Chroma
        
        # Linear interpolation for Lightness: 0 (black) -> 55 (vibrant color)
        L = 55 * norm_mag
        
        # Linear interpolation for Chroma: 0 (black/grey) -> max_chroma
        C = max_chroma * norm_mag
        
    else:
        raise ValueError(f"Unknown center '{center}'. Use 'white' or 'black'.")

    # Construct discrete Lch array
    # Lch shape: (N, 3) -> L, c, h
    lch = np.stack([L, C, hue_rad], axis=-1)
    
    # Convert to Lab
    # skimage.color.lch2lab takes Lch (L, C, h) and returns Lab
    lab_cart = color.lch2lab(lch)
    
    # Convert Lab to RGB
    rgb = color.lab2rgb(lab_cart)
    
    # Clip to [0, 1] just in case of out-of-gamut colors
    rgb = np.clip(rgb, 0, 1)
    
    return rgb

def _hsv_vector_colors(angles, norm_mag, center):
    """
    Generate colors using HSV color space.
    """
    # Hue: 0 to 1
    H = (angles % (2 * np.pi)) / (2 * np.pi)
    
    if center == 'white':
        # White center: 
        # Low mag -> White (Saturation 0, Value 1)
        # High mag -> Full Color (Saturation 1, Value 1)
        
        S = norm_mag
        V = np.ones_like(norm_mag)
        
    elif center == 'black':
        # Black center:
        # Low mag -> Black (Value 0)
        # High mag -> Full Color (Saturation 1, Value 1)
        
        S = np.ones_like(norm_mag) # Or maybe scale S too? Usually full saturation is nice.
        V = norm_mag
        
    else:
        raise ValueError(f"Unknown center '{center}'. Use 'white' or 'black'.")
        
    hsv = np.stack([H, S, V], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    
    return rgb
