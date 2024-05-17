import numpy as np
from tqdm import tqdm
from scipy.optimize import curve_fit
from . import func
### normalization
#min=0,max=1, no type conversions
def normalize(data):
    data = data - np.min(data)
    data = data / np.max(data)
    return data
def normalize_sum(data):
    return data/data.sum()
def normalize_max(data):
    return data/data.max()



# window_dimension must be odd
def gaussian_fit_peaks(image, peaks0, window_dimension=5,store_fits=True, remove_unfit = True):

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
    
    YY,XX = np.meshgrid(np.arange(-winrad,winrad+1),np.arange(-winrad,winrad+1))
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
            xf[it] = popt[2]+float(x0_i)
            yf[it] = popt[1]+float(y0_i)
            opts[it,:] = popt
            data_fits[:,:,it,0] = ydata
            data_fits[:,:,it,1] = func.gaussian_2d((YY,XX),*popt).reshape(XX.shape)
            
        except RuntimeError:
            errors[it] = True
            
    
    if remove_unfit:
        print(f'Removing indices: {np.where(errors)[0]}')
        xf = np.delete(xf,errors)
        yf = np.delete(yf,errors)
    else:
        print(f'Set to NaN: {np.where(errors)[0]}')
        xf[errors] = x0[errors]
        yf[errors] = y0[errors]

    return np.array((yf,xf)).T, errors, opts, data_fits