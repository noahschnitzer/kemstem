import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import LineCollection,PatchCollection

import cmocean
from skimage.util import montage

def plot_numbered_points(image,points,ax=None,delta=0,delta_step=0,verbose=0,zoom=100,vmin=None,vmax=None,color='r'):
    x = np.array(points)[:,1]
    y = np.array(points)[:,0]
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


def plot_fit_comparison(data_fit_stack,figsize=(6,3),cmap='viridis'):
    if len(data_fit_stack.shape)==3:
        data_fit_stack = np.expand_dims(data_fit_stack,2)
    fig,ax = plt.subplots(1,2,constrained_layout=True,figsize=figsize)
    ax[0].matshow(montage(np.rollaxis(data_fit_stack[:,:,:,0],2,0),padding_width=10,fill=0),cmap=cmap)
    ax[0].set_title('Data')
    ax[0].axis('off')
    ax[1].matshow(montage(np.rollaxis(data_fit_stack[:,:,:,1],2,0),padding_width=10,fill=0),cmap=cmap)
    ax[1].set_title('Fit')
    ax[1].axis('off')
    return fig,ax

def neighborhood_scatter_plot(neighborhood,ax=None,clusters=None):
    if ax is None:
        fig,ax = plt.subplots(1,1,constrained_layout=True)
    ax.axhline(y=0,color='k')
    ax.axvline(x=0,color='k')
    ax.plot(neighborhood[:,1],neighborhood[:,0],'.',alpha=.5,markersize=1)
    if clusters is not None:
        ax.plot(clusters[:,1],clusters[:,0],'.')
        for it, ctr in enumerate(clusters):
            ax.text(ctr[1]+2,ctr[0],str(it),color='red')
    ax.invert_yaxis()
    return ax


def plot_scalar_sites(ax,scalar, sites,s=5,cmap='inferno',**scatterargs):
    mappable=ax.scatter(x = sites[:,1],y=sites[:,0],s=s,c=scalar,cmap=cmap,**scatterargs)
    return mappable

def plot_scalar_bonds(ax,scalar,sites_1,sites_2,cmap='inferno',linewidth=1,vmin=None,vmax=None,**lcargs):
    vnorm = Normalize(vmin=vmin,vmax=vmax,clip=False)
    segs = np.stack((sites_1[:,::-1],sites_2[:,::-1]),axis=1)
    line_segments = LineCollection(segs, array=scalar,norm = vnorm,
                                   linewidths=(linewidth),
                                   linestyles='solid',cmap=cmap,**lcargs)
    ax.add_collection(line_segments)
    return line_segments


def plot_phase(phase,ax=None):
    if ax is None:
        fig,ax = plt.subplots(1,1,constrained_layout=True)
    ctours = [-np.sqrt(3)/2,0,np.sqrt(3)/2]
    mappable = ax.matshow(phase,cmap=cmocean.cm.phase,alpha=1,vmin=0,vmax=2*np.pi)
    ctours = [-np.sqrt(3)/2,0,np.sqrt(3)/2]#np.arange(0,2*np.pi,np.pi/4)
    ax.contour(np.sin(phase),ctours,colors='black',alpha=1, linewidths=1,linestyles='solid')
    ax.axis('off')
    return ax, mappable

def plot_displaced_site(columns,displacements,scale,colors='angle',ax=None,cmap='hsv',linewidth=.2,shape=4,angleshift=0,disp_min=0,disp_max=np.inf,scale_power=0.5):
    mask_sites = (np.linalg.norm(displacements,axis=1) < disp_max) & (np.linalg.norm(displacements,axis=1) > disp_min)
    x0 = columns[mask_sites,1] 
    y0 = columns[mask_sites,0] 
    dx0 = displacements[mask_sites,1]* scale
    dy0 = displacements[mask_sites,0]* scale
    angles = np.arctan2(dy0,dx0)
    patches=[]
    for i in range(len(x0)):
        y,x=y0[i],x0[i]
        dy,dx=dy0[i],dx0[i]
        L=(dx**2.+dy**2.)**(.5)
        L = L ** scale_power
        #L=L**.5 
        L2=L/shape
        xy = np.array([[x,y],[x,y],[x,y]]) +np.array([[L*np.cos(angles[i]), L*np.sin(angles[i])], [L2*np.sin(angles[i]), -L2*np.cos(angles[i])], [-L2*np.sin(angles[i]), L2*np.cos(angles[i])]])
        triangle=Polygon(xy, closed=True)
        patches.append(triangle)
    p = PatchCollection(patches, cmap=cmap, alpha=1)
    
    if type(colors) is str and colors=='angle':
        p.set_array((angles+angleshift)%(2*np.pi))
    elif type(colors) is str and colors=='mag':
        p.set_array(np.linalg.norm(displacements,axis=1))
    elif type(colors) is np.ndarray and colors.shape == angles.shape:
        p.set_array(np.array(colors))
    else:
        print('No usable colors')
    p.set_edgecolor('k')
    p.set_linewidth(linewidth)
    if ax is None:
        fig,ax = plt.subplots(1,1,constrained_layout=True)
    cax=ax.add_collection(p)
    return cax


def coarsening_marker(axis,coarsening_radius,position_frac=(0.95,.95),**kwargs):
    posx = position_frac[0]*axis.get_xlim()[1]
    posy = position_frac[1]*axis.get_ylim()[0]
    coarsening_marker = Circle((posx,posy),radius=coarsening_radius,**kwargs)
    axis.add_patch(coarsening_marker)
    return coarsening_marker

