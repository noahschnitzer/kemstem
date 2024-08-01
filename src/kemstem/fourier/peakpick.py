import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
from .. import util


def prepare_fourier_pattern(image,log=False, log_offset = 1e0):
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

    Returns
    -------
    ndarray
        The Fourier pattern, with the same shape as the input image

    """
    if log:
        return np.log(log_offset+np.abs(np.fft.fftshift(np.fft.fft2(image))))
    else:
        return np.fft.fftshift(np.fft.fft2(image))

'''

'''
# TODO: delete_within -- arg where if clicked pt is within this of another pt, delete that pt instead of adding
def select_peaks(pattern,cmap='gray',vmin=None,vmax=None,zoom=None,figsize=None,select_conjugates=False,delete_within=None):

    x = []
    y = []
    if figsize is None:
        fig,ax = plt.subplots(1,1,constrained_layout=True)
    else:
        fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=figsize)


    def replot():
        ax.matshow(np.real(pattern),vmin=vmin,vmax=vmax,cmap=cmap)
        if zoom is not None:
            ax.set_xlim(pattern.shape[0]/2 - zoom ,pattern.shape[0]/2 + zoom)
            ax.set_ylim(pattern.shape[1]/2 + zoom ,pattern.shape[1]/2 - zoom)
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
    return (y,x)
    #return np.stack((x,y))

# TODO: lots. fit, better viz, better store, check impl.
def refine_peaks_com(pattern, p0, crop_window,iters=10,viz=False):
    p_ref = np.zeros_like(p0) #np.zeros((len(p0[0]),2)) # (n,2)
    for it,pt in enumerate(p0):
        y0,x0 = pt
        for refnum in range(iters):
            crop = util.general.normalize(pattern[int(y0) - crop_window: int(y0) + crop_window, int(x0) - crop_window: int(x0) + crop_window])
            yr,xr = center_of_mass(crop)
            x0 = xr - crop_window + x0
            y0 = yr - crop_window + y0

        p_ref[it,:] = (y0,x0)

        if viz:
            fig,ax = plt.subplots(1,1)
            ax.matshow(crop,cmap='gray')
            ax.plot(xr,yr,'rx')
            ax.plot(crop_window,crop_window,'bo')
    return p_ref
def refine_peaks_gf(pattern, p0, window_dimension=5,store_fits=True, remove_unfit = True):
    return util.general.gaussian_fit_peaks(pattern, p0, window_dimension=window_dimension,store_fits=store_fits, remove_unfit = remove_unfit)