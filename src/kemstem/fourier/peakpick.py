import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from .. import util


def prepare_fourier_pattern(image,log=False, log_offset = 1e0, edge_reduction=None):
    """
    Calculate 0-frequency centered FFT of a 2D image.

    Returned FFT is complex unless log transformed for visualization.

    Parameters
    ----------
    image : ndarray
        2D array for transformation
    log : boolean
        Whether to take absolute value and log transform the result for visualization
    log_offset : float
        Value added to FFT magnitude prior to log transform to enhance contrast.
    edge_reduction : string
        Optionally apply a transform to the image to reduce edge artifacts. Options are:
        'hann' -- applies hann window, see util.image.hann_filter
        'pplus' -- applies periodic-plus-smooth decomposition, see util.image.periodic_plus_smooth_decomposition
    Returns
    -------
    pattern : ndarray
        The Fourier pattern, with the same shape as the input image

    """
    if edge_reduction is not None:
        if edge_reduction == 'hann':
            ft = util.image.hann_filter(image)[1]
        elif edge_reduction == 'pplus':
            ft = util.image.periodic_plus_smooth_decomposition(image)[0]
        else:
            raise ValueError(f"Invalid argument: {edge_reduction!r}. Expected 'hann' or 'pplus'.")
    else:
        ft = np.fft.fft2(image)
    ft = np.fft.fftshift(ft)
    if log:
        ft = np.log(log_offset+np.abs(ft))
    return ft

def select_peaks(pattern,preselected=None, cmap='gray',vmin=None,vmax=None,zoom=None,figsize=None,select_conjugates=False,delete_within=None):
    """
    Interactive peak selection tool.

    This function displays the input pattern and allows the user to select peaks
    by clicking on the image. It provides options for zooming, selecting conjugate
    peaks, and deleting nearby peaks.

    Parameters
    ----------
    pattern : ndarray
        2D array representing the Fourier pattern.
    preselected : ndarray, shape (n,2)
        Optional initial set of (y,x) points which will be preselected.
    cmap : str, optional
        Colormap for displaying the pattern (default is 'gray').
    vmin : float, optional
        Minimum value for color scaling (default is None).
    vmax : float, optional
        Maximum value for color scaling (default is None).
    zoom : float, optional
        Zoom factor for the display (default is None). 
        A box in the center of the pattern with width and height 2*zoom will be shown.
    figsize : tuple, optional
        Figure size in inches (width, height) (default is None).
    select_conjugates : bool, optional
        If True, automatically select the conjugate peak for each clicked point (default is False).
    delete_within : float, optional
        Distance threshold for deleting nearby peaks instead of adding new ones (default is None).
        When set, clicking within this many pixels of a previously selected point will delete the 
        original point.

    Returns
    -------
    peaks_selected : ndarray, shape
        A tuple containing two lists (y, x) of the selected peak coordinates. 

    Notes
    -----
    Left-click to add peaks. If delete_within is set, clicking near an existing peak
    will delete it instead of adding a new one.

    The result is returned as a tuple of mutable lists to allow peaks to be conveniently 
    added and removed in the midst of analysis, but this data structure does not conform 
    to the kemstem conventions for points. To use the selected peaks for further analysis
    the result should be transformed as:
    p0 = np.array(peaks_selected).T
    to arrive at an array with shape (n,2).
    
    """
    if preselected is None:
        x = []
        y = []
    else:
        y = list(preselected[:,0])
        x = list(preselected[:,1])
    if figsize is None:
        fig,ax = plt.subplots(1,1,constrained_layout=True)
    else:
        fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=figsize)


    def replot():
        ax.matshow(np.real(pattern),vmin=vmin,vmax=vmax,cmap=cmap)
        if zoom is not None:
            ax.set_xlim(pattern.shape[1]/2 - zoom ,pattern.shape[1]/2 + zoom)
            ax.set_ylim(pattern.shape[0]/2 + zoom ,pattern.shape[0]/2 - zoom)
        ax.axis('off')

    def _onclick_event(event):
        new_x = event.xdata
        new_y = event.ydata
        added_point = False
        if delete_within is not None and len(x) > 0:
            _,idx,dist = util.point.get_nearest_points(np.array((y,x)).T,(new_y,new_x),k=1)
            if dist < delete_within:
                del x[idx]
                del y[idx]
                ax.clear()
                #ax.matshow(np.real(pattern),vmin=vmin,vmax=vmax,cmap=cmap)
                replot()
                ax.plot(x,y,'r.')
            else:
                x.append(new_x)
                y.append(new_y)
                ax.plot(new_x,new_y,'r.')
                added_point = True
        else:
            x.append(new_x)
            y.append(new_y)
            ax.plot(new_x,new_y,'r.')
            added_point = True
        # TODO: test works properly for non square
        # TODO: test off by one for odd / even with flooring...
        if select_conjugates and added_point:
            size_x = pattern.shape[1]
            size_y = pattern.shape[0]
            x_conj = -(new_x-(size_x//2.)) + (size_x//2.)
            y_conj = -(new_y-(size_y//2.)) + (size_y//2.)
            x.append(x_conj)
            y.append(y_conj)
            ax.plot(x_conj,y_conj,'r.')
    fig.canvas.mpl_connect('button_press_event',_onclick_event)
    replot()
    ax.plot(x,y,'r.')

    return (y,x)

def refine_peaks_com(pattern, p0, crop_window,iters=10):
    """
    Refine peak positions using center of mass calculations.

    This function iteratively refines the positions of n initially detected peaks
    using the center of mass method within a specified window around each peak.

    Parameters
    ----------
    pattern : ndarray
        Real valued 2D array representing the Fourier pattern.
    p0 : array-like, shape (n,2)
        Initial peak positions as (y, x) coordinates.
    crop_window : int
        Radius of the window around each peak for refinement.
    iters : int, optional
        Number of refinement iterations (default is 10).
    viz : bool, optional
        If True, visualize the refinement process for each peak (default is False).

    Returns
    -------
    p_ref : ndarray, shape (n,2)
        Array of refined peak positions.
    """

    p_ref = np.zeros_like(p0)
    for it,pt in enumerate(p0):
        y0,x0 = pt
        for refnum in range(iters):
            crop = util.general.normalize(pattern[int(y0) - crop_window: int(y0) + crop_window, int(x0) - crop_window: int(x0) + crop_window])
            yr,xr = center_of_mass(crop)
            x0 = xr - crop_window + x0
            y0 = yr - crop_window + y0

        p_ref[it,:] = (y0,x0)
    return p_ref

def refine_peaks_gf(pattern, p0, window_dimension=5, remove_unfit = True,verbose=True):
    """
    Refine peak positions with 2D Gaussian fits. Currently wraps util.general.gaussian_fit_peaks
    """

    return util.general.gaussian_fit_peaks(pattern, p0, window_dimension=window_dimension, remove_unfit = remove_unfit,verbose=verbose)