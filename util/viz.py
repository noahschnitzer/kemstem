import matplotlib.pyplot as plt
import numpy as np

'''

'''
def plot_numbered_points(image,points,ax=None,delta=0,delta_step=0,verbose=0,zoom=100,vmin=None,vmax=None,color='r'):
    x = np.array(points)[0,:]
    y = np.array(points)[1,:]
    if ax is None:
    	fig,ax = plt.subplots(1,1,constrained_layout=True)
    if image is not None:
    	ax.matshow(image,cmap='gray',vmin=vmin,vmax=vmax)
    	ax.set_xlim(image.shape[0]/2 - zoom ,image.shape[0]/2 + zoom)
    	ax.set_ylim(image.shape[1]/2 + zoom ,image.shape[1]/2 - zoom)

    ax.plot(x,y,'.',color=color)
    for it in range(len(x)):
        if verbose:
            print(str(it)+':'+str(x[it])+','+str(y[it]))
        ax.text(x[it]+delta,y[it]+delta,str(it),color='w',fontsize=4)
        delta = delta+delta_step
    #ax.set_xlim(image.shape[0],0)


#def plot_fit_comparison(data_stack,fit_stack,montage_shape=None):
