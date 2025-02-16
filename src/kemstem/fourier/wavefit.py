import numpy as np
from .. import util
import scipy.optimize as opt
from concurrent.futures import ProcessPoolExecutor


def _create_patches(grating, patch_size, step_size):
    """
    Create patches from a grating image for analysis.
    
    Divides the input grating into overlapping square patches of specified size,
    stepping through the image at regular intervals.
    
    Parameters
    ----------
    grating : ndarray
        Input image to be divided into patches.
    patch_size : int
        Size of each square patch (must be odd).
    step_size : int
        Number of pixels to move between patch centers.
    
    Returns
    -------
    patches : ndarray
        Array of extracted patches.
    Ypoints : ndarray
        Y-coordinates of patch centers.
    Xpoints : ndarray
        X-coordinates of patch centers.
    shape : tuple
        Shape of the patch grid (rows, columns).
    """
    if patch_size % 2 == 0:
        raise ValueError('patch_size must be odd.')
    patch_radius = patch_size // 2
    patches = []
    Ypoints = np.arange(grating.shape[0])[patch_radius:-(patch_radius+1):step_size]
    Xpoints = np.arange(grating.shape[1])[patch_radius:-(patch_radius+1):step_size]
    for yy in Ypoints:
        for xx in Xpoints:
            #patches.append(grating[yy-patch_radius:yy+patch_radius+1,xx-patch_radius:xx+patch_radius+1])
            patches.append(util.get_patch(grating, [yy,xx], patch_size))
    patches = np.array(patches)
    shape = (len(Ypoints),len(Xpoints))
    return patches,Ypoints,Xpoints,shape

def _renormalize_each_patch(patches):
    """
    Helper function to normalize each patch in the input array.
    
    Applies mean normalization to each patch individually.
    
    Parameters
    ----------
    patches : ndarray
        Array of image patches to normalize.
    
    Returns
    -------
    ndarray
        Array of normalized patches.
    """
    return np.array([util.general.normalize_mean(patch) for patch in patches])


def _fit_patch(args):
    """
    Fit a 2D sinusoidal function to a single patch.
    
    Uses curve fitting to match a 2D sinusoid to the input patch data.
    
    Parameters
    ----------
    args : tuple
        Tuple containing:
        - patch (ndarray): The image patch to fit
        - guess (tuple): Initial parameter guess for the fit
        - yx (tuple): Meshgrid coordinates for the patch
    
    Returns
    -------
    popt : tuple
        Optimal values for the parameters
    perr : tuple
        Standard deviation errors on the parameters
    data_fits : ndarray
        Stack of original patch and fitted function
    """
    patch = args[0]
    guess = args[1]
    yx = args[2]


    try: 
        popt,pcov = opt.curve_fit(util.func.sin2D, yx,patch.ravel(),
            p0 = guess,  #bounds=bounds,
            ftol=1e-12, method='lm',
            xtol=1e-12,gtol=1e-12,maxfev=4000,)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError:
        popt = (np.nan,np.nan,np.nan,np.nan)#guess
        perr = (np.nan,np.nan,np.nan,np.nan)
    data_fits = np.stack((patch,util.func.sin2D(yx,*popt).reshape(patch.shape)),axis=2)
    return popt,perr,data_fits

def fit_grating(grating, patch_size, step_size, guess,
                chunksize=None,verbose=True,renormalize_patches=False, match_image_shape=True,store_full_results=False):
    """
    Fit 2D sinusoidal functions to patches of a grating image.
    
    Divides the input grating into patches and fits each patch with a 2D sinusoid
    using parallel processing.
    
    Parameters
    ----------
    grating : ndarray
        Input real valued grating image to analyze.
    patch_size : int
        Size of each square patch (must be odd).
    step_size : int
        Number of pixels to move between patch centers.
    guess : tuple
        Initial parameter guess for the sinusoidal fit. See peak_to_fit_guess
    chunksize : int, optional
        Size of chunks for parallel processing. Default is npatches//20.
    verbose : bool, optional
        Whether to print progress information. Default is True.
    renormalize_patches : bool, optional
        Whether to normalize patches before fitting. Default is False.
    match_image_shape : bool, optional
        Whether to interpolate results to match original image shape. Default is True. Assumes target shape is square.
    store_full_results : bool, optional
        Whether to return additional fitting results. Default is False.
    
    Returns
    -------
    amplitude : ndarray
        Fitted amplitude values.
    spacing : ndarray
        Fitted spacing values (pixels).
    rotation : ndarray
        Fitted rotation angles (radians).
    sampled_points : ndarray
        Coordinates of patch centers.
    If store_full_results is True, also returns:
        opts : ndarray
            All fitted parameters.
        errs : ndarray
            Parameter fitting covariances.
        patches : ndarray
            Original patches.
        mesh : tuple
            Coordinate meshgrid for patches.
    """
    patches,Ypoints,Xpoints,subsample_shape = _create_patches(grating,patch_size,step_size)
    #patches = patches / grating.max()

    if renormalize_patches:
        patches = _renormalize_each_patch(patches)
    # Ypoints, Xpoints order should be switched (bc of the transpose) (???)
    sampled_points = np.array(np.meshgrid(Ypoints,Xpoints,indexing='ij')).T.reshape(-1,2)
    mesh = np.meshgrid(np.arange(patches.shape[1]),np.arange(patches.shape[2]))
    npatches = patches.shape[0]
    if verbose:
        print(f'# of patches: {npatches}')
    if chunksize == None:
        chunksize = npatches//20
        if verbose:
            print(f'Using chunk size: {chunksize}')

    executor = ProcessPoolExecutor()
    futures = executor.map(_fit_patch,[(patch,guess,mesh) for patch in patches],chunksize=chunksize)
    # doing patch / grating.max() causes a huge hang before the multiprocess execution, so definitely don't do...
    executor.shutdown()

    opts = np.zeros((npatches,4))
    errs = np.zeros((npatches,4))
    for it,future in enumerate(futures):
        opts[it,:] = future[0]
        errs[it,:] = future[1]

    amplitude = np.abs(opts[:,0].reshape(subsample_shape))
    rotation = opts[:,2].reshape(subsample_shape)
    spacing = 2*np.pi / opts[:,1].reshape(subsample_shape)

    if match_image_shape:
        amplitude = util.general.rasterize_from_points(sampled_points,amplitude,grating.shape)
        rotation = util.general.rasterize_from_points(sampled_points,rotation,grating.shape)
        spacing = util.general.rasterize_from_points(sampled_points,spacing,grating.shape)

    if store_full_results:
        return amplitude,spacing,rotation,sampled_points, opts, errs, patches, mesh

    else:
        return amplitude,spacing,rotation,sampled_points


def peak_to_fit_guess(pt,imshape):
    """
    Convert a peak position to initial parameter guesses for sinusoidal fitting.
    
    Calculates initial guesses for amplitude, spacing, and rotation based on the
    position of a peak relative to the image center.
    
    Parameters
    ----------
    pt : ndarray
        Peak position coordinates (y, x).
    imshape : tuple
        Shape of the image (height, width).
    
    Returns
    -------
    tuple
        Initial parameter guesses (amplitude, radius, theta, phase) for sinusoidal fitting.
    """
    c = np.array(imshape)//2
    r = np.linalg.norm(pt-c)
    r= 2*np.pi *r / imshape[0]
    theta = np.arctan2((pt-c)[1],(pt-c)[0])
    return (1,r,theta,0)

def test_fit(grating,patch_size,step_size,guess,test_patch_idx = None,renormalize_patches=False):
    """
    Test sinusoidal fitting on a single patch from the grating.
    
    Extracts patches from the grating and performs fitting on a single selected patch
    for testing purposes.
    
    Parameters
    ----------
    grating : ndarray
        Input grating image.
    patch_size : int
        Size of each square patch (must be odd).
    step_size : int
        Number of pixels to move between patch centers.
    guess : tuple
        Initial parameter guess for the fit.
    test_patch_idx : int, optional
        Index of patch to test. Default is npatches//4.
    renormalize_patches : bool, optional
        Whether to normalize patches before fitting. Default is False.
    
    Returns
    -------
    tuple
        Results from fit_patch for the selected patch:
        - Optimal parameter values
        - Parameter errors
        - Original and fitted data
    """

    patches,Ypoints,Xpoints,subsample_shape = _create_patches(grating,patch_size,step_size)
    if renormalize_patches:
        patches = _renormalize_each_patch(patches)
    mesh = np.meshgrid(np.arange(patches.shape[1]),np.arange(patches.shape[2]))
    npatches = patches.shape[0]
    if test_patch_idx is None:
        test_patch_idx = npatches//4
    print(f'{npatches} total patches, testing on {test_patch_idx}')
    return _fit_patch((patches[test_patch_idx,:,:],guess,mesh))

