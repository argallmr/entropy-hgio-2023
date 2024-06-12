#unnecessary ones should be deleted
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt, dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import time
import os
#from pymms.data import util, edi, fpi, anc
import datetime as dt
from  pathlib import Path
from scipy import constants
from matplotlib import pyplot as plt, dates as mdates, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pymms import config

import datetime as dt
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt, dates as mdates
from scipy import constants
import re
from pymms.data import fpi, edp ,util
#import util

def non_max(f, g):
    # Integrate over phi and theta
    #   - Measurement bins with zero counts result in a
    #     phase space density of 0
    #   - Photo-electron correction can result in negative
    #     phase space density.
    #   - Log of value <= 0 is nan. Avoid be replacing
    #     with 1 so that log(1) = 0
    s = (f - g)**2
    try:
        # 1e12 converts s^3/cm^6 to s^3/m^6
        s = (1e12 * f * s).integrate('phi')
    except ValueError:
        # In burst mode, phi is time-dependent
        #   - Use trapz to explicitly specify which axis is being integrated
        #   - Expand dimensions of phi so that they are broadcastable
        s = np.trapz(1e12 * f * s,
                          f['phi'].expand_dims({'theta': 1, 'energy_index': 1}, axis=(2, 3)),
                          axis=f.get_axis_num('phi_index'))
        
        # trapz returns an ndarray. Convert it back to a DataArray
        s = xr.DataArray(s,
                              dims=('time', 'theta', 'energy_index'),
                              coords={'time': f['time'],
                                      'theta': f['theta'],
                                      'energy_index': f['energy_index'],
                                      'U': f['U']})
    
    # Integrate over theta
    s = (np.sin(s['theta']) * s).integrate('theta')

    # Integrate over Energy
    with np.errstate(invalid='ignore', divide='ignore'):
        y = np.sqrt(s['U']) / (1 - s['U'])**(5/2)
    y = y.where(np.isfinite(y.values), 0)

    coeff = np.sqrt(2) * (eV2J * E0 / mass)**(3/2)
    s = coeff * np.trapz(y * s, y['U'], axis=y.get_axis_num('energy_index'))
    s=np.sqrt (s)
    s = xr.DataArray(s, dims='time', coords={'time': f['time']})

    return s # J/K/m^3



def greco (sc, mode, species , t0 , t1 ):
    E0 = 100 # keV
    kB = constants.k # J/K
    eV2J = constants.eV
    eV2K = constants.value('electron volt-kelvin relationship')
    me = constants.m_e
    mp = constants.m_p
    mass = me if species == 'e' else mp
    instr = 'd{0}s'.format(species)
    level = 'l2'
    optdesc = 'd{0}s-dist'.format(species)
    fpi_dist = fpi.load_dist(sc=sc, mode=mode, optdesc=optdesc,
                         start_date=t0, end_date=t1)

    #Precondition the distributions
    fpi_kwargs = fpi.precond_params(sc, mode, level, 'des-dist', t0, t1,time=fpi_dist['time'])
    f = fpi.precondition(fpi_dist['dist'], **fpi_kwargs)
    n = fpi.density(f)
    V = fpi.velocity(f, n=n)
    T = fpi.temperature(f, n=n, V=V)
    P = fpi.pressure(f, n=n, T=T)
    t = ((T[:,0,0] + T[:,1,1] + T[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    g = fpi.maxwellian_distribution(f, N=n, bulkv=V, T=t)
    n_max = fpi.density(g)
    V_max = fpi.velocity(g, n=n_max)
    T_max = fpi.temperature(g, n=n_max, V=V_max)
    t_max = ((T_max[:,0,0] + T_max[:,1,1] + T_max[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    v_th=np.sqrt(kB*t/mass)**(3/2)*1/10
    
    def non_max(f, g):
    # Integrate over phi and theta
    #   - Measurement bins with zero counts result in a
    #     phase space density of 0
    #   - Photo-electron correction can result in negative
    #     phase space density.
    #   - Log of value <= 0 is nan. Avoid be replacing
    #     with 1 so that log(1) = 0
        s = (f - g)**2
        try:
        # 1e12 converts s^3/cm^6 to s^3/m^6
            s = (1e12 * f * s).integrate('phi')
        except ValueError:
        # In burst mode, phi is time-dependent
        #   - Use trapz to explicitly specify which axis is being integrated
        #   - Expand dimensions of phi so that they are broadcastable
            s = np.trapz(1e12 * f * s,
                          f['phi'].expand_dims({'theta': 1, 'energy_index': 1}, axis=(2, 3)),
                          axis=f.get_axis_num('phi_index'))
        
        # trapz returns an ndarray. Convert it back to a DataArray
            s = xr.DataArray(s,
                              dims=('time', 'theta', 'energy_index'),
                              coords={'time': f['time'],
                                      'theta': f['theta'],
                                      'energy_index': f['energy_index'],
                                      'U': f['U']})
    
    # Integrate over theta
        s = (np.sin(s['theta']) * s).integrate('theta')

    # Integrate over Energy
        with np.errstate(invalid='ignore', divide='ignore'):
            y = np.sqrt(s['U']) / (1 - s['U'])**(5/2)
        y = y.where(np.isfinite(y.values), 0)

        coeff = np.sqrt(2) * (eV2J * E0 / mass)**(3/2)
        s = coeff * np.trapz(y * s, y['U'], axis=y.get_axis_num('energy_index'))
        s=np.sqrt (s)
        s = xr.DataArray(s, dims='time', coords={'time': f['time']})

        return s # J/K/m^3
    
    greco2=v_th/n*non_max(f,g)
    
    
    
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False,figsize=(6.5, 2))

    # Current Density
    #ax = axes[0,0]
    #D1.loc[:,'x'].plot(ax=ax, label='x')
    #D1.loc[:,'y'].plot(ax=ax, label='y')
    #D1.loc[:,'z'].plot(ax=ax, label='z')
    #ax.set_title('Application of Reciprocal Vectors')
    #ax.set_xlabel('')
    #ax.set_xticklabels([''])
    #ax.set_ylabel('J [$\\mu A/m^{2}$]')
    #ax.legend()

    ax = axes[0,0]
    greco2.plot(ax=ax, label='$greco,  non-maxwelianity_{V,'+species+'}$')
  #  plt.ylim(0.05, 0.25)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    
    # util.format_axes(ax)
    ax.legend()

    plt.setp(axes, xlim=mdates.date2num([t0, t1]))
    return fig, axes
