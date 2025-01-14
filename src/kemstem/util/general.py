import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy import interpolate
from . import func
def normalize(data):
    """
    Normalize an array between 0 and 1 (inclusive).

    Subtract array minimum and divide by its maximum. No type conversions made.

    Parameters
    ----------
    data : ndarray
        Array to normalize. 

    Returns
    -------
    normalized : ndarray
        The normalized array.

    """

    data = data - np.min(data)
    data = data / np.max(data)
    return data


def normalize_sum(data):
    """
    Normalize an array by dividing by its sum.

    Parameters
    ----------
    data : ndarray
        Array to normalize.

    Returns
    -------
    normalized : ndarray
        The normalized array where the sum of all elements equals 1.
    """
    return data/data.sum()


def normalize_max(data):
    """
    Normalize an array by dividing by its maximum value.

    Parameters
    ----------
    data : ndarray
        Array to normalize.

    Returns
    -------
    normalized : ndarray
        The normalized array where the maximum value is 1.
    """

    return data/data.max()

def normalize_mean(data):
    """
    Normalize an array by centering the mean and scaling by the standard deviation

    Parameters
    ----------
    data: ndarray
        Array to normalize

    Returns
    -------
    normalize: ndarray
        The normalized array with mean of 0 and one standard deviation at +-1
    """

    return (data - data.mean()) / np.std(data)



def gaussian_fit_peaks(image, peaks0, window_dimension=5,store_fits=True, remove_unfit = True, verbose = True):
    """
    Fit 2D Gaussian functions to peaks in an image.

    This function refines the positions of n initially detected peaks by fitting
    a 2D Gaussian function to the region around each peak.

    Parameters
    ----------
    image : ndarray
        Real valued 2D array representing the image.
    peaks0 : ndarray, shape (n,2)
        Initial peak positions as (y, x) coordinates.
    window_dimension : int, optional
        Full width of the window around each peak for Gaussian fitting (default is 5).
        Must be an odd number.
    store_fits : bool, optional
        Not implemented. If True, store the Gaussian fit parameters and fitted data (default is True).
    remove_unfit : bool, optional
        If True, remove peaks that fail to fit properly (default is True).

    Returns
    -------
    p_ref : ndarray, shape (n,2)
        Refined peak positions as (y, x) coordinates.
    errors : ndarray, shape (n,)
        Boolean array indicating fitting errors.
    fit_params : ndarray, shape (n,7)
        Gaussian fit parameters for each peak:
        (amplitude, x-position, y-position, stdev1, stdev2, angle, offset).
    data_fits : ndarray, shape (window_dimension, window_dimension, n, 2)
        Original and fitted data for each peak with shape:
        The last dimension contains the original data (index 0) and the fitted data (index 1).
    Raises
    ------
    ValueError
        If window_dimension is not an odd number.

    Notes
    -----
    This function uses scipy.optimize.curve_fit to perform the Gaussian fitting.
    """

    if window_dimension % 2 == 0:
        raise ValueError('window_dimension must be odd.')

    if len(peaks0.shape) == 1:
        peaks0 = np.expand_dims(peaks0,axis=0)

    winrad = window_dimension // 2
    
    x0 = peaks0[:,1]
    y0 = peaks0[:,0]
    n_sites = x0.shape[0]
    xf = np.zeros(x0.shape)
    yf = np.zeros(y0.shape)
    errors = np.zeros(x0.shape,dtype=bool)
    opts = np.zeros((n_sites,7))
    data_fits = np.zeros((window_dimension,window_dimension,n_sites,2))
    
    YY,XX = np.meshgrid(np.arange(-winrad,winrad+1),np.arange(-winrad,winrad+1),indexing='ij')
    for it in tqdm(range(n_sites)):
        x0_i = int(x0[it])
        y0_i = int(y0[it])
        
        ydata = image[y0_i - winrad : y0_i + winrad + 1, x0_i - winrad : x0_i + winrad + 1]
        if ydata.shape != (window_dimension, window_dimension):
            errors[it] = True
            continue
        
        bounds = [ (0,-winrad,-winrad,0,0,0,-np.inf),
                   (np.inf,winrad,winrad,window_dimension,window_dimension,2*np.pi,np.inf)]
        initial_guess = (ydata[winrad, winrad], 0, 0, winrad*.8, winrad*.8, 0, 0)

        try:
            popt,pcov = curve_fit(func.gaussian_2d,(YY,XX), ydata.flatten(),p0=initial_guess,bounds=bounds)
            xf[it] = popt[1]+float(x0_i)
            yf[it] = popt[2]+float(y0_i)
            opts[it,:] = popt
            data_fits[:,:,it,0] = ydata
            data_fits[:,:,it,1] = func.gaussian_2d((YY,XX),*popt).reshape(XX.shape)
            
        except (RuntimeError,ValueError):
            errors[it] = True
            
    if errors.sum() > 0:
        if remove_unfit:
            if verbose:
                print(f'Errors with {np.sum(errors)} fits, removed')
            xf = np.delete(xf,errors)
            yf = np.delete(yf,errors)
            opts = np.delete(opts,errors,axis=0)
            data_fits = np.delete(data_fits,errors,axis=2)
        else:
            if verbose:
                print(f'Errors with {np.sum(errors)} fits, set to NaN')
            xf[errors] = x0[errors]
            yf[errors] = y0[errors]

    return np.array((yf,xf)).T, errors, opts, data_fits


def rasterize_from_points(points,values,output_shape,method='nearest',fill_value=np.nan):
    """
    Interpolate scattered data onto a regular grid.

    Parameters
    ----------
    points : ndarray, shape (n,2)
        Array of point coordinate pairs (y,x).
    values : ndarray, shape (n,)
        Array of values at the given points.
    output_shape : tuple
        Shape of the output grid (H, W).
    method : str, optional
        Interpolation method: 'nearest', 'linear', or 'cubic' (default is 'nearest').
    fill_value : float, optional
        Value used to fill points outside of the convex hull of input points (default is np.nan).

    Returns
    -------
    interp : ndarray
        Interpolated values on a regular grid with the specified output_shape.

    Notes
    -----
    This function uses scipy.interpolate.griddata for interpolation.
    """
    xi = np.array(np.meshgrid(np.arange(output_shape[0]),np.arange(output_shape[1]),indexing='ij')).T
    interp = interpolate.griddata(points,values.ravel(),xi,method=method,fill_value=fill_value)
    return interp    


