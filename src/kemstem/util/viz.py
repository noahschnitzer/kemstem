import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize, is_color_like, to_rgba
from matplotlib.patches import Polygon, Circle, FancyArrow
from matplotlib.collections import LineCollection,PatchCollection

import cmocean
from skimage.util import montage

from .colormaps import vector_to_color

def plot_numbered_points(image,points,ax=None,delta=0,delta_step=0,verbose=0,zoom=100,vmin=None,vmax=None,color='r', fontsize=8):
    """
    Plot numbered points on an image.

    Parameters
    ----------
    image : ndarray or None
        2D array representing the image. If None, only points are plotted.
    points : array-like, shape (n,2)
        Array of point coordinates as (y, x).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    delta : float, optional
        Initial offset for point labels (default is 0).
    delta_step : float, optional
        Step size for incrementing label offset (default is 0).
    verbose : int, optional
        If non-zero, print point coordinates (default is 0).
    zoom : float, optional
        Zoom factor for display (default is 100).
    vmin, vmax : float, optional
        Color scale range for image display.
    color : str, optional
        Color of the points (default is 'r').
    fontsize: float,optional
        Font size for point labels. Default is 8. 

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """

    x = np.array(points)[:,1]
    y = np.array(points)[:,0]
    if ax is None:
    	fig,ax = plt.subplots(1,1,constrained_layout=True)
    if image is not None:
    	ax.matshow(image,cmap='gray',vmin=vmin,vmax=vmax)
    	ax.set_xlim(image.shape[1]/2 - zoom ,image.shape[1]/2 + zoom)
    	ax.set_ylim(image.shape[0]/2 + zoom ,image.shape[0]/2 - zoom)

    ax.plot(x,y,'.',color=color)
    for it in range(len(x)):
        if verbose:
            print(str(it)+':'+str(x[it])+','+str(y[it]))
        ax.text(x[it]+delta,y[it]+delta,str(it),color='w',fontsize=fontsize)
        delta = delta+delta_step
    return ax


def plot_fit_comparison(data_fit_stack,figsize=(6,3),cmap='viridis'):
    """
    Plot a comparison of original data and fitted data.

    Parameters
    ----------
    data_fit_stack : ndarray, shape (fit_window, fit_window, n_fits, 2)
        Array where the last dimension contains the original data (index 0) and the fitted data (index 1).
    figsize : tuple, optional
        Figure size in inches (width, height) (default is (6, 3)).
    cmap : str, optional
        Colormap for the plots (default is 'viridis').

    Returns
    -------
    tuple
        Figure and axes objects (fig, ax).
    """

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

def neighborhood_scatter_plot(neighborhood,ax=None,clusters=None,fontsize=8):
    """
    Create a scatter plot of n pair correlation vectors with optional labeled cluster centers.

    Parameters
    ----------
    neighborhood : ndarray, shape (n,2)
        Array of point coordinates as (y, x).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    clusters : ndarray, shape (n,2) optional
        Array of cluster center coordinates as (y, x).

    Returns
    -------
    matplotlib.axes.Axes
        The axes containing the plot.
    """

    if ax is None:
        fig,ax = plt.subplots(1,1,constrained_layout=True)
    ax.axhline(y=0,color='k')
    ax.axvline(x=0,color='k')
    ax.plot(neighborhood[:,1],neighborhood[:,0],'.',alpha=.5,markersize=1)
    if clusters is not None:
        ax.plot(clusters[:,1],clusters[:,0],'.')
        for it, ctr in enumerate(clusters):
            ax.text(ctr[1]+2,ctr[0],str(it),color='red',fontsize=fontsize)
    ax.invert_yaxis()
    ax.axis('scaled')
    return ax


def plot_scalar_sites(ax,scalar, sites,s=5,cmap='inferno',**scatterargs):
    """
    Plot scalar values at a set of n points.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    scalar : array-like, shape (n)
        Scalar values to be plotted.
    sites : ndarray, shape (n,2)
        Array of point coordinates as (y, x).
    s : float, optional
        Marker size (default is 5).
    cmap : str, optional
        Colormap for scalar values (default is 'inferno').
    **scatterargs : dict
        Additional arguments to pass to ax.scatter().

    Returns
    -------
    matplotlib.collections.PathCollection
        The scatter plot collection.
    """

    mappable=ax.scatter(x = sites[:,1],y=sites[:,0],s=s,c=scalar,cmap=cmap,**scatterargs)
    return mappable

def plot_scalar_bonds(ax,scalar,sites_1,sites_2,cmap='inferno',linewidth=1,vmin=None,vmax=None,**lcargs):
    """
    Plot scalar values as colored bonds between sites.

    Note: to plot lines of all a single color, pass cmap=None and color=[your color of interest]

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    scalar : array-like, shape (n)
        Scalar values to be plotted.
    sites_1 : ndarray, shape (n,2)
        Arrays of coordinates for the start points of bonds as (y, x).
    sites_2 : ndarray, shape (n,2)
        Arrays of coordinates for the end points of bonds as (y, x).
    cmap : str, optional
        Colormap for scalar values (default is 'inferno').
    linewidth : float, optional
        Width of the bond lines (default is 1).
    vmin, vmax : float, optional
        Color scale range for scalar values.
    **lcargs : dict
        Additional arguments to pass to LineCollection.

    Returns
    -------
    matplotlib.collections.LineCollection
        The line collection representing the bonds.
    """

    vnorm = Normalize(vmin=vmin,vmax=vmax,clip=False)
    segs = np.stack((sites_1[:,::-1],sites_2[:,::-1]),axis=1)
    line_segments = LineCollection(segs, array=scalar,norm = vnorm,
                                   linewidths=(linewidth),
                                   linestyles='solid',cmap=cmap,**lcargs)
    ax.add_collection(line_segments)
    return line_segments


def plot_scalar_polygons(ax,sites,scalar,cmap='inferno',vmin=None,vmax=None,linewidth=0,alpha=1):
    """
    Plot n scalar values as colored polygons with k sides.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    sites : ndarray, shape (k,n,2)
        Coordinates (y,x) of k vertices of each n polygons to be plotted.
    scalar : ndarray, shape (n,)
        Scalar values to be plotted.
    cmap : str, optional
        Colormap for scalar values (default is 'inferno').
    linewidth : float, optional
        Width of the outer polygon lines (default is 0).
    vmin, vmax : float, optional
        Color scale range for scalar values.
    alpha : float, optional
        Transparency of the polygons, default 1.
    Returns
    -------
    matplotlib.collections.PatchCollection
        The Patch collection of the polygons.
    """
    n_points = [site.shape[0] for site in sites]
    n_points.append(scalar.shape[0])
    n_points = np.array(n_points)
    if np.all(n_points[0] == n_points):
        n_site = n_points[0]
    else:
        raise ValueError(f'All shapes must be the same: {n_points}')

    patches=[]
    for it in range(n_site):
        xy = np.array([site[it,::-1] for site in sites])
        #xy = np.array([site1[it,::-1],site2[it,::-1],site3[it,::-1]])
        poly = Polygon(xy, closed=True)
        patches.append(poly)
    #p = PatchCollection(patches, cmap='inferno', alpha=1,)
    #p.set_array(angles)
    vnorm = Normalize(vmin=vmin,vmax=vmax,clip=False)
    p = PatchCollection(patches, array=scalar,norm = vnorm, cmap=cmap, alpha=alpha,linewidths=(linewidth),edgecolor='k')
    
    ax.add_collection(p)
    return p



def plot_phase(phase,ax=None,contour_angles=[0,2*np.pi/3,4*np.pi/3],contour_phase_shift=0,linewidths=1,colors='black',**kwargs):
    """
    Plot a phase image with contour lines.

    Contours are drawn at 2*pi/3 steps.

    Parameters
    ----------
    phase : ndarray
        Real valued 2D array of phase values, in radians.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    contour_angles : iterable
        Angles in the range [0,2*pi] at which to draw contour lines
    contour_phase_shift : float
        phase offset at which to draw contours
    linewidths : float, optional
        Width of the contour lines (default is 1).
    colors : string or list
        Colors passed to call to `contour` to color contour lines
    **kwargs : dict
        Additional arguments to pass to ax.contour().

    Returns
    -------
    tuple
        Axes and colorbar mappable object (ax, mappable).
    """

    if ax is None:
        fig,ax = plt.subplots(1,1,constrained_layout=True)
    phasemap = ax.matshow(phase%(2*np.pi),cmap=cmocean.cm.phase,alpha=1,vmin=0,vmax=2*np.pi)
    for angle in contour_angles:
        adj_phase = np.sin(contour_phase_shift+phase-angle)
        adj_phase[np.cos(contour_phase_shift+phase-angle)>0] = np.nan
        ax.contour(adj_phase,levels = [0],colors=colors,alpha=1, linewidths=linewidths,linestyles='solid',**kwargs)
    return ax,phasemap

def plot_displaced_site(columns,displacements,scale,colors='angle',ax=None,cmap='hsv',linewidth=.2,shape=4,angleshift=0,disp_min=0,disp_max=np.inf,scale_power=0.5,style='triangle',arrow_width=None):
    """
    Plot n displaced sites as colored triangles or arrows.

    Parameters
    ----------
    columns : ndarray, shape (n,2)
        Array of original site coordinates as (y, x).
    displacements : ndarray, shape (n,2)
        Array of displacement vectors as (dy, dx).
    scale : float
        Scaling factor for displacements.
    colors : str, tuple, or ndarray, optional
        How the triangles are colored (default is 'angle'):
        'angle' -- displacement direction, mapped through cmap.
        'mag' -- displacement magnitude, mapped through cmap.
        'vector' -- direction and magnitude, see util.colormaps.vector_to_color.
        Any other matplotlib color string (e.g. 'red', 'C0', '#ff8800') or an
        RGB/RGBA tuple or list -- a single facecolor applied to every site.
        ndarray, shape (n,) -- one value per site, mapped through cmap.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    cmap : str, optional
        Colormap for the triangles (default is 'hsv').
    linewidth : float, optional
        Width of the triangle or arrow edges (default is 0.2).
    shape : float, optional
        Shape factor for triangles, the ratio of length to half the base
        width (default is 4). Not used for arrows.
    angleshift : float, optional
        Angle shift for color mapping (default is 0).
    disp_min, disp_max : float, optional
        Minimum and maximum displacement magnitudes to plot.
    scale_power : float, optional
        Power scaling factor for displacement magnitudes (default is 0.5).
    style : {'triangle', 'arrow'}, optional
        Marker drawn at each site (default is 'triangle'). Both start at the
        site and have length (scale*|displacement|)**scale_power.
    arrow_width : float, optional
        Shaft width of the arrows in data units (pixels). As in a quiver plot,
        all arrows share the same shaft width and head size (head width 3x and
        head length 5x the shaft width), and arrows shorter than the head are
        shrunk as a whole. If None, 0.1 times the median arrow length is used;
        pass a value to keep arrows consistent between plots (e.g. with a
        separately plotted scale). Not used for triangles.

    Returns
    -------
    matplotlib.collections.PatchCollection
        The collection of triangle or arrow patches.
    """
    if style not in ('triangle', 'arrow'):
        raise ValueError(f"Invalid argument: style={style!r}. Expected 'triangle' or 'arrow'.")


    mask_sites = (np.linalg.norm(displacements,axis=1) < disp_max) & (np.linalg.norm(displacements,axis=1) > disp_min)
    x0 = columns[mask_sites,1] 
    y0 = columns[mask_sites,0] 
    dx0 = displacements[mask_sites,1]* scale
    dy0 = displacements[mask_sites,0]* scale
    angles = np.arctan2(dy0,dx0)
    mags = np.linalg.norm(displacements[mask_sites,:],axis=1)
    lengths = np.hypot(dx0, dy0) ** scale_power
    if style == 'arrow' and arrow_width is None:
        arrow_width = 0.1*np.median(lengths) if len(lengths) > 0 else 1.0
    patches=[]
    for i in range(len(x0)):
        y,x=y0[i],x0[i]
        L = lengths[i]
        if style == 'triangle':
            L2=L/shape
            xy = np.array([[x,y],[x,y],[x,y]]) +np.array([[L*np.cos(angles[i]), L*np.sin(angles[i])], [L2*np.sin(angles[i]), -L2*np.cos(angles[i])], [-L2*np.sin(angles[i]), L2*np.cos(angles[i])]])
            patches.append(Polygon(xy, closed=True))
        else:
            # quiver-style proportions in units of shaft width, scaled down for arrows shorter than the head
            w = arrow_width * min(1, L/(5*arrow_width))
            patches.append(FancyArrow(x, y, L*np.cos(angles[i]), L*np.sin(angles[i]), width=w,
                                      head_width=3*w, head_length=5*w, overhang=0.1, length_includes_head=True))

    varray = None
    facecolors = None
    vnorm = None
    
    if type(colors) is str and colors=='angle':
        #p.set_array((angles+angleshift)%(2*np.pi))
        varray = (angles+angleshift)%(2*np.pi)
        #print(((angles+angleshift)%(2*np.pi)).min())
        #print(((angles+angleshift)%(2*np.pi)).max())
        #print(((angles+angleshift)%(2*np.pi)).mean())
        vnorm = Normalize(vmin=0,vmax=2*np.pi,clip=False)

    elif type(colors) is str and colors=='mag':
        #p.set_array(mags)
        varray = mags
        vnorm = Normalize(vmin=None,vmax=None,clip=False)
    
    elif type(colors) is str and colors == 'vector':
        # Parse cmap for method and center
        # Defaults
        method = 'lab'
        center = 'white'
        
        # Check if cmap string provides hints
        if isinstance(cmap, str):
            parts = cmap.split('_')
            # Check for method
            if parts[0] in ['lab', 'hsv']:
                method = parts[0]
            # Check for center
            if len(parts) > 1 and parts[1] in ['white', 'black']:
                center = parts[1]
            elif cmap in ['white', 'black']: # handle case like 'white' implying 'lab_white'
                center = cmap
        
        # Calculate max magnitude for normalization if not provided via disp_max
        # If disp_max is inf, use the max of the data
        max_mag = disp_max if disp_max != np.inf else np.max(mags) if len(mags) > 0 else 1.0

        facecolors = vector_to_color(angles + angleshift, mags, method=method, center=center, max_mag=max_mag)

    elif isinstance(colors, (str, tuple, list)) and is_color_like(colors):
        facecolors = to_rgba(colors)

    elif type(colors) is np.ndarray:
        if colors.shape != (columns.shape[0],):
            raise ValueError(f"colors array has shape {colors.shape}, expected ({columns.shape[0]},), one value per site.")
        ucolors = colors[mask_sites]
        #p.set_array(ucolors)
        varray = ucolors
        vnorm = Normalize(vmin=None,vmax=None,clip=False)

    else:
        raise ValueError(f"Invalid argument: colors={colors!r}. Expected 'angle', 'mag', 'vector', "
                         "a matplotlib color, or an array of one value per site.")
    
    if facecolors is not None:
         p = PatchCollection(patches, facecolors=facecolors)
    else:
         p = PatchCollection(patches, cmap=cmap, alpha=1, norm=vnorm)
         p.set_array(varray)

    p.set_edgecolor('k')
    p.set_linewidth(linewidth)
    if ax is None:
        fig,ax = plt.subplots(1,1,constrained_layout=True)
    cax=ax.add_collection(p)
    return cax


def coarsening_marker(axis,coarsening_radius,position_frac=(0.95,.95),facecolor='w',edgecolor='k',**kwargs):
    """
    Add a circular marker to indicate coarsening length scale.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axes to add the marker to.
    coarsening_radius : float
        Radius of the coarsening marker.
    position_frac : tuple, optional
        Fractional position of the marker center (x, y) (default is (0.95, 0.95)).
    facecolor : str, optional
        Fill color of the marker (default is 'w').
    edgecolor : str, optional
        Edge color of the marker (default is 'k').
    **kwargs : dict
        Additional arguments to pass to matplotlib.patches.Circle.

    Returns
    -------
    matplotlib.patches.Circle
        The circular marker patch.
    """

    posx = position_frac[0]*axis.get_xlim()[1]
    posy = position_frac[1]*axis.get_ylim()[0]
    coarsening_marker = Circle((posx,posy),radius=coarsening_radius,facecolor=facecolor,edgecolor=edgecolor,**kwargs)
    axis.add_patch(coarsening_marker)
    return coarsening_marker


def plot_color_wheel(cm,filename=None,dpi=150,transparent=True,n_points=256,r_inner=0.3,r_outer=0.8,s=1e0,angleshift=0):
    """
    Plots a color wheel for the provided colormap or vector coloring method.

    Parameters
    ----------
    cm : colormap or str
        Colormap to plot a colorwheel for. Can be a matplotlib colormap object, 
        a string name of a matplotlib colormap, or a vector coloring string 
        (e.g., 'lab_white', 'hsv_black').
    filename : string, optional
        Path to save color wheel image to.
    dpi : int, optional
        dots per inch for saved image. No effect if no filename provided. Default 150.
    transparent : boolean, optional
        Save with transparent background. Default True.
    n_points : int, optional
        Number of points used for plotting along one dimension (grid is n_points x n_points). Default 1024.
    r_inner : float, optional
        Inner radius of color wheel. Default 0.3
    r_outer : float, optional
        Outer radius of color wheel. Default 0.8
    s : float, optional
        Size of markers comprising wheel. Default 1.
    angleshift : float, optional
        Angle shift (radians) applied to the colors, matching the angleshift
        argument of plot_displaced_site. Default 0.

    Notes
    -----
    The wheel is drawn with the y axis pointing down, as for images, so that
    the color at each direction on the wheel matches the color of a
    displacement in that direction plotted with plot_displaced_site on an image.

    """ 
    YY,XX = np.meshgrid(np.linspace(-1,1,n_points),np.linspace(-1,1,n_points))
    XX =XX.ravel()
    YY = YY.ravel()
    r = np.sqrt(XX**2+YY**2)
    theta = np.arctan2(YY,XX)

    r_filter = (r < r_outer) & (r > r_inner)
    
    # Handle vector coloring string
    is_vector_color = False
    if isinstance(cm, str):
        parts = cm.split('_')
        if parts[0] in ['lab', 'hsv']:
            is_vector_color = True
            method = parts[0]
            center = parts[1] if len(parts) > 1 else 'white'
            
            # Normalize r for magnitude input [0, 1] relative to r_outer?
            # Or just pass r directly if we consider r_outer to be max_mag=1?
            # Let's say magnitude varies from 0 to 1 at r=1.
            # But here we are plotting r up to 1 (or close to it).
            # So magnitudes = r.
            # We want colors at r=r_outer to be full magnitude?
            # Actually, `vector_to_color` takes magnitudes and normalizing factor.
            # Let's use max_mag=r_outer so that the outer edge is "full saturation/magnitude".
            magnitudes = r[r_filter]
            angles = theta[r_filter] + angleshift
            
            c = vector_to_color(angles, magnitudes, method=method, center=center, max_mag=r_outer)
            
        else:
            # Try to get matplotlib colormap from string
            try:
                cm = plt.get_cmap(cm)
            except ValueError:
                print(f"Warning: '{cm}' is not a known colormap. Using default.")
                cm = plt.get_cmap('viridis')

    if not is_vector_color:
        # Standard matplotlib colormap behavior
        c = cm(((theta[r_filter] + angleshift) % (2*np.pi)) / (2*np.pi))

    fig,ax = plt.subplots(1,1,constrained_layout=True,figsize=(3,3))
    #ax.matshow(r)
    ax.scatter(XX[r_filter],YY[r_filter],s=s,c=c)
    ax.axis('equal')
    ax.invert_yaxis()
    ax.axis('off')
    if filename is not None:
        fig.savefig(filename, transparent=transparent,dpi=dpi)
