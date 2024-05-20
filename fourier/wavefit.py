import numpy as np
from .. import util
import scipy.optimize as opt
from concurrent.futures import ProcessPoolExecutor


# patch_size must be odd
def create_patches(grating, patch_size, step_size):
    patch_radius = patch_size // 2
    patches = []
    Ypoints = np.arange(grating.shape[0])[patch_radius:-(patch_radius+1):step_size]
    Xpoints = np.arange(grating.shape[1])[patch_radius:-(patch_radius+1):step_size]
    for yy in Ypoints:
        for xx in Xpoints:
            patches.append(grating[yy-patch_radius:yy+patch_radius+1,xx-patch_radius:xx+patch_radius+1])
    patches = np.array(patches)
    shape = (len(Ypoints),len(Xpoints))
    return patches,Ypoints,Xpoints,shape

'''
    args:
        0 patch
        1 guess
'''
def fit_patch(args):
    patch = args[0]
    guess = args[1]
    yx = np.meshgrid(np.arange(patch.shape[0]),np.arange(patch.shape[1]))

    try: 
        popt,pcov = opt.curve_fit(util.func.sin2D, yx,patch.ravel(),
            p0 = guess, method='lm', ftol=1e-12,
            xtol=1e-12,gtol=1e-12,maxfev=4000)
        perr = np.sqrt(np.diag(pcov))
    except RuntimeError:
        popt = initial_guess
        perr = (np.nan,np.nan,np.nan,np.nan)

    data_fits = np.stack((patch,util.func.sin2D(yx,*popt).reshape(patch.shape)),axis=2)

    return popt,perr,data_fits

def fit_grating(grating,patch_size,step_size,guess,chunksize=None):
    assert grating.shape[0] == grating.shape[1]
    assert len(grating.shape)==2

    patches,Ypoints,Xpoints,subsample_shape = create_patches(grating,patch_size,step_size)
    sampled_points = np.array(np.meshgrid(Ypoints,Xpoints,indexing='ij')).T.reshape(-1,2)
    npatches = patches.shape[0]

    if chunksize == None:
        chunksize = npatches//20

    executor = ProcessPoolExecutor()
    futures = executor.map(fit_patch,[(patch/grating.max(),guess) for patch in patches],chunksize=chunksize)
    executor.shutdown()

    opts = np.zeros((npatches,4))
    errs = np.zeros((npatches,4))
    for it,future in enumerate(futures):
        opts[it,:] = future[0]
        errs[it,:] = future[1]

    amplitude = np.abs(opts[:,0].reshape(subsample_shape))
    rotation = opts[:,2].reshape(subsample_shape)
    spacing = 2*np.pi / opts[:,1].reshape(subsample_shape)

    return amplitude,spacing,rotation,sampled_points


def peak_to_fit_guess(pt,imshape):
    c = np.array(imshape//2)
    r = np.linalg.norm(pt-c)
    r= 2*np.pi *r / imshape[0]
    theta = np.arctan2((pt-c)[1],(pt-c)[0])
    return (1,r,theta,0)

def test_fit(grating,patch_size,step_size,guess,test_patch_idx = None):
    patches,Ypoints,Xpoints,subsample_shape = create_patches(grating,patch_size,step_size)
    npatches = patches.shape[0]
    if test_patch_idx is None:
        test_patch_idx = npatches//4
    return fit_patch((patches[test_patch_idx,:,:],guess))

