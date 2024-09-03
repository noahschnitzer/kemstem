import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import scipy as sp
from scipy.ndimage.filters import minimum_filter, median_filter



def find_RP(strain_map,alpha,x0,x1,y0,y1,threshold, width_peak, linewidth, distance, vis = False):
    x_end = int((y1-y0)*np.tan(alpha)+x0)
    z = sk.measure.profile_line(strain_map, src = [y0, x0], dst = [y1, x_end], linewidth = linewidth) # Takes integrated width line profile
    z[z< threshold] = 0 # thresholds the data
    
    peaks, _ = sp.signal.find_peaks(z, distance = distance, width = width_peak)
    
    RP = []
    for j in range(len(peaks)):
        RP.append(np.array([int((float(peaks[j]))*(np.cos(alpha))+(y0)),int(((float(peaks[j]))*(np.sin(alpha))+(x0)))]))
            
    if vis:
        fig,ax = plt.subplots(1,2,figsize = (9,4))
        ax[0].plot(z)
        ax[0].plot(peaks,np.zeros(peaks.shape[0]),'r.')
        ax[1].matshow(median_filter(strain_map,size=10), cmap='BrBG')
        ax[1].plot((x0,x_end),(y0,y1),'r.')
        ax[1].plot(np.array(RP)[:,1],np.array(RP)[:,0],'ro-',markersize = 3)
    return (RP)

def measure_distances(RP,px_scale):
    UC_dist = []
    bond_coordinates = []
    for i in range(len(RP)):
        for j in range(len(RP[i])-1):
            px_dist = np.sqrt((RP[i][j][0]-RP[i][j+1][0])**2+(RP[i][j][1]-RP[i][j+1][1])**2) 
            UC_dist.append(int(np.floor(px_dist * px_scale)))
            bond_coordinates.append([[RP[i][j][1],RP[i][j+1][1]],[RP[i][j][0],RP[i][j+1][0]]])
    return (UC_dist,bond_coordinates)
            
def vis_bond(ax,image,strain_map,UC_dist,bond_coordinates,cmap='plasma'):
    c_im00=ax.matshow(image, cmap='gray')
    c_im0 = ax.matshow(strain_map, cmap='BrBG',alpha=0.4)
    
    cm = plt.get_cmap('plasma')  
    count = len(set(UC_dist)) # Number of different n phases

    colors = []

    for i in range(len(bond_coordinates)):
        this_color = cm(int((UC_dist[i]-1)*256/count))
        if colors.count(this_color) == 0:
                colors.append(this_color)
        ax.plot(bond_coordinates[i][0], bond_coordinates[i][1],color = this_color, marker = 'o' ,linewidth = 1.5, markersize = 1, label ='n = ' + str(UC_dist[i]))

    hand, labl = ax.get_legend_handles_labels()
    handout=[]
    lablout=[]
    for h,l in zip(hand,labl):
        if l not in lablout:
            lablout.append(l)
            handout.append(h)
    ax.legend(handout, lablout, fontsize = 'medium', bbox_to_anchor=(1.2,0.3))
    return colors
    
def vis_hist(ax,UC_dist,colors):

    while UC_dist.count(0) > 0:
        UC_dist.remove(0)

    u, counts = np.unique(UC_dist, return_counts=True)
    print(u)
    print(counts) 

    plt.bar(np.arange(len(u)), counts, color = colors)
    plt.xticks(np.arange(len(u)), u)
    plt.show()
