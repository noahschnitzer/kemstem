import numpy as np
from .. import util
import scipy.optimize as opt
from concurrent.futures import ProcessPoolExecutor


def create_patches(grating, patch_size, step_size):
    '''
        path_size must be odd
    '''
    assert patch_size % 2 != 0
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

def renormalize_each_patch(patches):
    # return np.array([(util.normalize(patch)-0.5)*2 for patch in patches])
    return np.array([util.general.normalize_mean(patch) for patch in patches])


def fit_patch(args):
    '''
        args:
            0 patch
            1 guess
    '''
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

def fit_grating(grating,patch_size,step_size,guess,chunksize=None,verbose=True,renormalize_patches=False, match_image_shape=True):
    # seems to all work ok for non square, aside from final interpolation
    #assert grating.shape[0] == grating.shape[1]
    #assert len(grating.shape)==2

    patches,Ypoints,Xpoints,subsample_shape = create_patches(grating,patch_size,step_size)
    #patches = patches / grating.max()

    if renormalize_patches:
        patches = renormalize_each_patch(patches)
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
    futures = executor.map(fit_patch,[(patch,guess,mesh) for patch in patches],chunksize=chunksize)
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

    return amplitude,spacing,rotation,sampled_points


def peak_to_fit_guess(pt,imshape):
    c = np.array(imshape//2)
    r = np.linalg.norm(pt-c)
    r= 2*np.pi *r / imshape[0]
    theta = np.arctan2((pt-c)[1],(pt-c)[0])
    return (1,r,theta,0)

def test_fit(grating,patch_size,step_size,guess,test_patch_idx = None,renormalize_patches=False):
    patches,Ypoints,Xpoints,subsample_shape = create_patches(grating,patch_size,step_size)
    if renormalize_patches:
        patches = renormalize_each_patch(patches)
    mesh = np.meshgrid(np.arange(patches.shape[1]),np.arange(patches.shape[2]))
    npatches = patches.shape[0]
    if test_patch_idx is None:
        test_patch_idx = npatches//4
    return fit_patch((patches[test_patch_idx,:,:],guess,mesh))

