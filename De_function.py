from pymms.data import fgm, edp, fpi, util
import datetime as dt
import xarray as xr
import numpy as np
from scipy import constants as c
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt, dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import time
import os

def curl(K, V):
        curl = 0
    
        for k_name, v_name in zip(K, V):
            k = K[k_name]
            v = V[v_name]
            curl += xr.concat([k[:,1]*v[:,2] - k[:,2]*v[:,1],
                           k[:,2]*v[:,0] - k[:,0]*v[:,2],
                           k[:,0]*v[:,1] - k[:,1]*v[:,0]], dim='component').transpose()
    
        curl = curl.assign_coords({'component': ['x', 'y', 'z']})
        return curl


def De_total(sc,mode,species,t0,t1):
 #   optdesc = 'd{0}s-moms'.format(species)
    #should it be brst or srvy?
    data = fpi.load_moms(sc=sc, mode=mode, optdesc=optdesc, start_date=t0, end_date=t1)
    data['velocity']
    b= fgm.load_data(sc=sc, mode=mode, start_date=t0, end_date=t1)

    b['B_GSE']
#brst of srvy for edp?
    e= edp.load_data(sc=sc, mode='srvy', start_date=t0, end_date=t1)
#e['E_GSE']

    def curl(K, V):
        curl = 0
    
        for k_name, v_name in zip(K, V):
            k = K[k_name]
            v = V[v_name]
            curl += xr.concat([k[:,1]*v[:,2] - k[:,2]*v[:,1],
                           k[:,2]*v[:,0] - k[:,0]*v[:,2],
                           k[:,0]*v[:,1] - k[:,1]*v[:,0]], dim='component').transpose()
    
        curl = curl.assign_coords({'component': ['x', 'y', 'z']})
        return curl
#should this one then be linear?
    b1 = b['B_GSE'].interp_like(e, method='linear')
    data1 =data.interp_like(e, method='nearest')
    B = xr.Dataset({'B1': b1})
    U = xr.Dataset({'U1': data1['velocity'].rename({'velocity_index': 'component'}).assign_coords({'component': ['x', 'y', 'z']})})
    B2 = 1e-9 * B
    U2 = 1e-3 * U
    curlB = curl(U2, B2)
#A = 1e-12 * curlB

    e1=e['E_GSE']
    #Error in here?!!
    E = xr.Dataset({'E1': e1.rename({''.join(sc)+'_edp_label1_fast_l2': 'component'}).assign_coords({'component': ['x', 'y', 'z']})})

    D= E['E1'] + curlB

    D #(just shows how the data looks like)

    fig, axes = plt.subplots(nrows=4, ncols=1, squeeze=False)


    ax = axes[0,0]
    D.loc[:,'x'].plot(ax=ax, label='x')
    D.loc[:,'y'].plot(ax=ax, label='y')
    D.loc[:,'z'].plot(ax=ax, label='z')
    ax.set_title('Electron Frame Dissipation Rate')
    ax.set_xlabel('')
    ax.set_xticklabels([''])
    ax.set_ylabel('De [$\\mu A/m^{2}$]')
    ax.legend()
    ax = axes[1,0]
    D.loc[:,'x'].plot(ax=ax, label='x')
    ax = axes[2,0]
    D.loc[:,'y'].plot(ax=ax, label='y',color='orange')
    ax = axes[3,0]
    D.loc[:,'z'].plot(ax=ax, label='z',color='green')

#D=D.to_array()

#D=D.T


# Create the plot
    nrows = 1
    ncols = 1
    figsize = (6.0, 2.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, squeeze=False)
    ax = axes[0, 0]

# Plot using pcolormesh
    mesh = ax.pcolormesh(D, shading='flat')
    fig.colorbar(mesh, ax=ax)

# Other plot customization if needed
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Pcolormesh Plot')

    plt.show()

    B['B1'] #(same as D)

#b_deleted = B['B1'].sel(b_index=slice(None, -1))
#n = B['B1'].shape[0]
    b_deleted = B['B1'].sel({'b_index' :['x','y','z']})
    b_deleted
    x=np.cross(U2['U1'], b_deleted)
    x= 1e-12 * x

    U['U1']


    D1= E['E1'] + x

    D1

#plot
    fig, axes = plt.subplots(nrows=4, ncols=2, squeeze=False)

# Current Density
    ax = axes[0,0]
    D1.loc[:,'x'].plot(ax=ax, label='x')
    D1.loc[:,'y'].plot(ax=ax, label='y')
    D1.loc[:,'z'].plot(ax=ax, label='z')
    ax.set_title('Electron Frame Dissipation Rate')
    ax.set_xlabel('')
    ax.set_xticklabels([''])
    ax.set_ylabel('De [$\\mu A/m^{2}$]')
    ax.legend()
    ax = axes[1,0]
    D1.loc[:,'x'].plot(ax=ax, label='x')
    ax = axes[2,0]
    D1.loc[:,'y'].plot(ax=ax, label='y',color='orange')
    ax = axes[3,0]
    D1.loc[:,'z'].plot(ax=ax, label='z',color='green')
    ax = axes[0,1]
    D.loc[:,'x'].plot(ax=ax, label='x')
    D.loc[:,'y'].plot(ax=ax, label='y')
    D.loc[:,'z'].plot(ax=ax, label='z')
    ax.set_title('Electron Frame Dissipation Rate')
    ax.set_xlabel('')
    ax.set_xticklabels([''])
    ax.set_ylabel('De [$\\mu A/m^{2}$]')
    ax.legend()
    ax = axes[1,1]
    D.loc[:,'x'].plot(ax=ax, label='x')
    ax = axes[2,1]
    D.loc[:,'y'].plot(ax=ax, label='y',color='orange')
    ax = axes[3,1]
    D.loc[:,'z'].plot(ax=ax, label='z',color='green')

#t0 = dt.datetime(2017, 7, 11, 22, 33, 30)
#t1 = dt.datetime(2017, 7, 11, 22, 34, 30)
#De('mms1','brst','e',t0,t1)



#for a single plot
def De_single(sc,mode,species,t0,t1):
 #   optdesc = 'd{0}s-moms'.format(species)
    #should it be brst or srvy?
        data = fpi.load_moms(sc=sc, mode=mode, optdesc=optdesc, start_date=t0, end_date=t1)
        data['velocity']
        b= fgm.load_data(sc=sc, mode=mode, start_date=t0, end_date=t1)

        b['B_GSE']
#brst of srvy for edp?
        e= edp.load_data(sc=sc, mode='srvy', start_date=t0, end_date=t1)
#e['E_GSE']

        def curl(K, V):
        curl = 0
    
                for k_name, v_name in zip(K, V):
                k = K[k_name]
                v = V[v_name]
                curl += xr.concat([k[:,1]*v[:,2] - k[:,2]*v[:,1],
                           k[:,2]*v[:,0] - k[:,0]*v[:,2],
                           k[:,0]*v[:,1] - k[:,1]*v[:,0]], dim='component').transpose()
    
                curl = curl.assign_coords({'component': ['x', 'y', 'z']})
                return curl
#should this one then be linear?
        b1 = b['B_GSE'].interp_like(e, method='linear')
        data1 =data.interp_like(e, method='nearest')
        B = xr.Dataset({'B1': b1})
        U = xr.Dataset({'U1': data1['velocity'].rename({'velocity_index': 'component'}).assign_coords({'component': ['x', 'y', 'z']})})
        B2 = 1e-9 * B
        U2 = 1e-3 * U
        curlB = curl(U2, B2)
#A = 1e-12 * curlB

        e1=e['E_GSE']
    #Error in here?!!
        E = xr.Dataset({'E1': e1.rename({''.join(sc)+'_edp_label1_fast_l2': 'component'}).assign_coords({'component': ['x', 'y', 'z']})})

        D= E['E1'] + curlB

        D #(just shows how the data looks like)

        fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)


        ax = axes[0,0]
        D.loc[:,'x'].plot(ax=ax, label='x')
        D.loc[:,'y'].plot(ax=ax, label='y')
        D.loc[:,'z'].plot(ax=ax, label='z')
        ax.set_title('Electron Frame Dissipation Rate')
        ax.set_xlabel('')
        ax.set_xticklabels([''])
        ax.set_ylabel('De [$\\mu A/m^{2}$]')
        ax.legend()
        ax = axes[1,0]
        D.loc[:,'x'].plot(ax=ax, label='x')
        ax = axes[2,0]
        D.loc[:,'y'].plot(ax=ax, label='y',color='orange')
        ax = axes[3,0]
        D.loc[:,'z'].plot(ax=ax, label='z',color='green')
       # plt.show()
#D=D.to_array()

#D=D.T



#t0 = dt.datetime(2017, 7, 11, 22, 33, 30)
#t1 = dt.datetime(2017, 7, 11, 22, 34, 30)
#De('mms1','brst','e',t0,t1)

