import numpy as np
import xarray as xr
import datetime as dt
from  pathlib import Path
from scipy import constants
from matplotlib import pyplot as plt, dates as mdates, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pymms import config 
from pymms.data import fpi
import os
#os.chdir('/Users/argall/Documents/Python/pymms/examples/')
#os.chdir(r"D:\uni UNH\mms\pymms\examples\\")
#import util
os.chdir('/Users/krmhanieh/Documents/GitHub/pymms/examples')
    
import util
#import os
#os.chdir('/Users/argall/Documents/Python/pymms/examples/')
#import util

data_root = Path(config['dropbox_root'])

E0 = 100 # keV

kB = constants.k # J/K
eV2J = constants.eV
eV2K = constants.value('electron volt-kelvin relationship')
me = constants.m_e
mp = constants.m_p

def relative_entropy(f, f_M):
    # Integrate over phi and theta
    #   - Measurement bins with zero counts result in a
    #     phase space density of 0
    #   - Photo-electron correction can result in negative
    #     phase space density.
    #   - Log of value <= 0 is nan. Avoid be replacing
    #     with 1 so that log(1) = 0
    E0 = 100 # keV

    kB = constants.k # J/K
    eV2J = constants.eV
    eV2K = constants.value('electron volt-kelvin relationship')
    me = constants.m_e
    mp = constants.m_p
    mass = me if species == species else mp
    sv_rel = f / f_M
    sv_rel = sv_rel.where(sv_rel > 0, 1)
    try:
        # 1e12 converts s^3/cm^6 to s^3/m^6
        sv_rel = (1e12 * f * np.log(sv_rel)).integrate('phi')
    except ValueError:
        # In burst mode, phi is time-dependent
        #   - Use trapz to explicitly specify which axis is being integrated
        #   - Expand dimensions of phi so that they are broadcastable
        sv_rel = np.trapz(1e12 * f * np.log(sv_rel),
                          f['phi'].expand_dims({'theta': 1, 'energy_index': 1}, axis=(2, 3)),
                          axis=f.get_axis_num('phi_index'))
        
        # trapz returns an ndarray. Convert it back to a DataArray
        sv_rel = xr.DataArray(sv_rel,
                              dims=('time', 'theta', 'energy_index'),
                              coords={'time': f['time'],
                                      'theta': f['theta'],
                                      'energy_index': f['energy_index'],
                                      'U': f['U']})
    
    # Integrate over theta
    sv_rel = (np.sin(sv_rel['theta']) * sv_rel).integrate('theta')

    # Integrate over Energy
    with np.errstate(invalid='ignore', divide='ignore'):
        y = np.sqrt(sv_rel['U']) / (1 - sv_rel['U'])**(5/2)
    y = y.where(np.isfinite(y.values), 0)

    coeff = -kB * np.sqrt(2) * (eV2J * E0 / mass)**(3/2)
    sv_rel = coeff * np.trapz(y * sv_rel, y['U'], axis=y.get_axis_num('energy_index'))

    sv_rel = xr.DataArray(sv_rel, dims='time', coords={'time': f['time']})

    return sv_rel # J/K/m^3


  def plot_lut_ts(n, t, s, sv, s_max_moms,
                n_max, t_max, s_max, sv_max, sv_rel_max,
                n_lut, t_lut, s_lut, sv_lut, sv_rel_lut):

    fig, axes = plt.subplots(nrows=5, ncols=1, squeeze=False, figsize=(6.5, 9))
    plt.subplots_adjust(top=0.95)

    # Error in the adjusted look-up table
    dn_max = (n - n_max) / n * 100.0
    dn_lut = (n - n_lut) / n * 100.0

    ax = axes[0,0]
    l1 = dn_max.plot(ax=ax, label='$\Delta n_{'+species+',Max}/n_{'+species+',Max}$')
    l2 = dn_lut.plot(ax=ax, label='$\Delta n_{'+species+',lut}/n_{'+species+',lut}$')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('$\Delta n_{'+species+'}/n_{'+species+'}$ (%)')
    # util.format_axes(ax, xaxis='off')
    util.add_legend(ax, [l1[0], l2[0]], corner='SE', horizontal=True)
    
    # Deviation in temperature
    dt_max = (t - t_max) / t * 100.0
    dt_lut = (t - t_lut) / t * 100.0
    
    ax = axes[1,0]
    l1 = dt_max.plot(ax=ax, label='$\Delta T_{'+species+',Max}/T_{'+species+',Max}$')
    l2 = dt_lut.plot(ax=ax, label='$\Delta T_{'+species+',lut}/T_{'+species+',lut}$')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('$\Delta T_{'+species+'}/T_{'+species+'}$ (%)')
    ax.set_ylim(-1,2.5)
    # util.format_axes(ax, xaxis='off')
    util.add_legend(ax, [l1[0], l2[0]], corner='NE', horizontal=True)
    
    # Deviation in entropy
    ds_moms = (s - s_max_moms) / s * 100.0
    ds_max = (s - s_max) / s * 100.0
    ds_lut = (s - s_lut) / s * 100.0
    
    ax = axes[2,0]
    l1 = ds_max.plot(ax=ax, label='$\Delta s_{'+species+',Max}/s_{'+species+',Max}$')
    l2 = ds_lut.plot(ax=ax, label='$\Delta s_{'+species+',lut}/s_{'+species+',lut}$')
    l3 = ds_moms.plot(ax=ax, label='$\Delta s_{'+species+',moms}/s_{'+species+',moms}$')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('$\Delta s_{'+species+'}/s_{'+species+'}$ (%)')
    ax.set_ylim(-9,2.5)
    # util.format_axes(ax, xaxis='off')
    util.add_legend(ax, [l1[0], l2[0], l3[0]], corner='SE', horizontal=True)
    
    # Deviation in velocity-space entropy
    dsv_max = (sv - sv_max) / sv * 100.0
    dsv_lut = (sv - sv_lut) / sv * 100.0
    
    ax = axes[3,0]
    l1 = dsv_max.plot(ax=ax, label='$\Delta s_{V,'+species+',Max}/s_{V,'+species+',Max}$')
    l2 = dsv_lut.plot(ax=ax, label='$\Delta s_{V,'+species+',lut}/s_{V,'+species+',lut}$')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$\Delta s_{V,'+species+'}/s_{V,'+species+'}$ (%)')
    # util.format_axes(ax)
    util.add_legend(ax, [l1[0], l2[0]], corner='SE', horizontal=True)
    
    ax = axes[4,0]
    l1 = sv_rel_max.plot(ax=ax, label='$s_{V,'+species+',rel,Max}$')
    l2 = sv_rel_M.plot(ax=ax, label='$s_{V,'+species+',rel,lut}$')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('$s_{V,'+species+',rel}$\n[J/K/$m^{3}$]')
    # util.format_axes(ax)
    util.add_legend(ax, [l1[0], l2[0]], corner='SE', horizontal=True)

    fig.suptitle('Maxwellian Look-up Table')

    def rel_entropy(sc, mode,species, start_date, end_date):
#    sc = 'mms3'
#mode = 'brst'
#species = 'e'
#start_date = dt.datetime(2017, 7, 11, 22, 34, 0)
#end_date = dt.datetime(2017, 7, 11, 22, 34, 5)
    def relative_entropy(f, f_M):
    # Integrate over phi and theta
    #   - Measurement bins with zero counts result in a
    #     phase space density of 0
    #   - Photo-electron correction can result in negative
    #     phase space density.
    #   - Log of value <= 0 is nan. Avoid be replacing
    #     with 1 so that log(1) = 0
        E0 = 100 # keV

        kB = constants.k # J/K
        eV2J = constants.eV
        eV2K = constants.value('electron volt-kelvin relationship')
        me = constants.m_e
        mp = constants.m_p
        mass = me if species == species else mp
        sv_rel = f / f_M
        sv_rel = sv_rel.where(sv_rel > 0, 1)
        try:
        # 1e12 converts s^3/cm^6 to s^3/m^6
            sv_rel = (1e12 * f * np.log(sv_rel)).integrate('phi')
        except ValueError:
        # In burst mode, phi is time-dependent
        #   - Use trapz to explicitly specify which axis is being integrated
        #   - Expand dimensions of phi so that they are broadcastable
            sv_rel = np.trapz(1e12 * f * np.log(sv_rel),
                          f['phi'].expand_dims({'theta': 1, 'energy_index': 1}, axis=(2, 3)),
                          axis=f.get_axis_num('phi_index'))
        
            # trapz returns an ndarray. Convert it back to a DataArray
            sv_rel = xr.DataArray(sv_rel,
                              dims=('time', 'theta', 'energy_index'),
                              coords={'time': f['time'],
                                      'theta': f['theta'],
                                      'energy_index': f['energy_index'],
                                      'U': f['U']})
    
        # Integrate over theta
        sv_rel = (np.sin(sv_rel['theta']) * sv_rel).integrate('theta')

        # Integrate over Energy
        with np.errstate(invalid='ignore', divide='ignore'):
            y = np.sqrt(sv_rel['U']) / (1 - sv_rel['U'])**(5/2)
        y = y.where(np.isfinite(y.values), 0)

        coeff = -kB * np.sqrt(2) * (eV2J * E0 / mass)**(3/2)
        sv_rel = coeff * np.trapz(y * sv_rel, y['U'], axis=y.get_axis_num('energy_index'))

        sv_rel = xr.DataArray(sv_rel, dims='time', coords={'time': f['time']})

        return sv_rel # J/K/m^3
    E0 = 100 # keV

    kB = constants.k # J/K
    eV2J = constants.eV
    eV2K = constants.value('electron volt-kelvin relationship')
    me = constants.m_e
    mp = constants.m_p
    mass = me if species == species else mp
    instr = 'd{0}s'.format(species)
    level = 'l2'
    optdesc = 'd{0}s-dist'.format(species)

    # Create a file name for the look-up table
    lut_file = data_root / '_'.join((sc, instr, mode, level,
                                 optdesc+'-look-up-table',
                                 start_date.strftime('%Y%m%d_%H%M%S'),
                                 end_date.strftime('%Y%m%d_%H%M%S')))
    lut_file = lut_file.with_suffix('.ncdf')
    # Read the data
    fpi_dist = fpi.load_dist(sc=sc, mode=mode, optdesc=optdesc,
                         start_date=start_date, end_date=end_date)

    # Precondition the distributions
    fpi_kwargs = fpi.precond_params(sc, mode, level, optdesc,
                                start_date, end_date,
                                time=fpi_dist['time'])
    f = fpi.precondition(fpi_dist['dist'], **fpi_kwargs)
    
    # Moments and entropy parameters for the measured distribution
    n = fpi.density(f)
    V = fpi.velocity(f, N=n)
    T = fpi.temperature(f, N=n, V=V)
    P = fpi.pressure(f, N=n, T=T)
    t = ((T[:,0,0] + T[:,1,1] + T[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    p = ((P[:,0,0] + P[:,1,1] + P[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    s = fpi.entropy(f)
    
    # Create equivalent Maxwellian distributions and calculate moments
    f_max = fpi.maxwellian_distribution(f, N=n, bulkv=V, T=t)
    
    sV = fpi.vspace_entropy(f, N=n, s=s)
    s_max_moms = fpi.maxwellian_entropy(n, p)
    n_max = fpi.density(f_max)
    V_max = fpi.velocity(f_max, N=n_max)
    T_max = fpi.temperature(f_max, N=n_max, V=V_max)
    s_max = fpi.entropy(f_max)
    sv_max = fpi.vspace_entropy(f, N=n_max, s=s_max)
    t_max = ((T_max[:,0,0] + T_max[:,1,1] + T_max[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    # Relative entropy
    sv_rel_max = relative_entropy(f, f_max)
    # Get density and temperature ranges over which to create the LUT
    # and calculate the number of bins to use
    n_lut_range = (0.9*n.min().values, 1.1*n.max().values)
    t_lut_range = (0.9*t.min().values, 1.1*t.max().values)
    # dims = (int(10**max(np.floor(np.abs(np.log10(N_range[1] - N_range[0]))), 1)),
    #         int(10**max(np.floor(np.abs(np.log10(t_range[1] - t_range[0]))), 1)))
    dims = (100, 100)

    # Print the LUT ranges and size
    print('Density range: {0}'.format(n_lut_range))
    print('Temperature range: {0}'.format(t_lut_range))
    print('Dimensions of LUT: {0}'.format(dims))
    
    # If the look-up table does not already exist, create it
    if not lut_file.exists():
        fpi.maxwellian_lookup(f[[0], ...], n_lut_range, t_lut_range, dims=dims, fname=lut_file)

    # Load the look-up table
    lut = xr.load_dataset(lut_file)
    
    
    # Find the error in N and T between the Maxwellian and Measured
    # distribution
    n_lut, t_lut = np.meshgrid(lut['N_data'], lut['t_data'], indexing='ij')
    dn = (n_lut - lut['N']) / n_lut * 100.0
    dt = (t_lut - lut['t']) / t_lut * 100.0
    
    
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8.5, 4))
    plt.subplots_adjust(wspace=0.6)

    # 2D Density LUT
    ax = axes[0,0]
    img = dn.T.plot(ax=ax, cmap=cm.get_cmap('rainbow', 15), add_colorbar=False)
    ax.set_title('')
    ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
    ax.set_ylabel('$t_{'+species+'}$ (eV)')

    # Create a colorbar that is aware of the image's new position
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="5%", pad=0.1)
    fig.add_axes(cax)
    cb = fig.colorbar(img, cax=cax, orientation="vertical")
    cb.set_label('$\Delta n_{'+species+'}/n_{'+species+'}$ (%)')
    cb.ax.minorticks_on()

    # 2D Temperature LUT
    ax = axes[0,1]
    img = dt.T.plot(ax=ax, cmap=cm.get_cmap('rainbow', 15), add_colorbar=False)
    ax.set_title('')
    ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
    ax.set_ylabel('$T_{'+species+'}$ (eV)')
    util.format_axes(ax, time=False)

    # Create a colorbar that is aware of the image's new position
    divider = make_axes_locatable(ax)
    cax = divider.new_horizontal(size="5%", pad=0.1)
    fig.add_axes(cax)
    cb = fig.colorbar(img, cax=cax, orientation="vertical")
    cb.set_label('$\Delta t_{'+species+'}/t_{'+species+'}$ (%)')
    cb.ax.minorticks_on()
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.5, 6.5))
    plt.subplots_adjust(wspace=0.5, hspace=0.35)

    # Density LUT at two densities and all temperatures
    ax = axes[0,0]
    dn[0,:].plot(ax=ax, label='n={0:0.4f}'.format(dn['N_data'][0].item()))
    dn[-1,:].plot(ax=ax, label='n={0:0.4f}'.format(dn['N_data'][-1].item()))
    ax.set_title('n LUT at n=const')
    ax.set_xlabel('T (eV)')
    ax.set_ylabel('$\Delta n/n$ (%)')
    ax.legend()

    # Temperature LUT at two densities and all temperatures
    ax = axes[0,1]
    dt[0,:].plot(ax=ax, label='n={0:0.4f}'.format(dn['N_data'][0].item()))
    dt[-1,:].plot(ax=ax, label='n={0:0.4f}'.format(dn['N_data'][-1].item()))
    ax.set_title('T LUT at n=const')
    ax.set_xlabel('T (eV)')
    ax.set_ylabel('$\Delta t/t$ (%)')
    ax.legend()

    # Density LUT at two temperatures and all densities
    ax = axes[1,0]
    dn[:,0].plot(ax=ax, label='T={0:0.2f}'.format(dn['t_data'][0].item()))
    dn[:,-1].plot(ax=ax, label='T={0:0.2f}'.format(dn['t_data'][-1].item()))
    ax.set_title('n LUT at t=const')
    ax.set_xlabel('n $(cm^{-3})$')
    ax.set_ylabel('$\Delta n/n$ (%)')
    ax.legend()

    # Temperature LUT at two temperatures and all densities
    ax = axes[1,1]
    dt[:,0].plot(ax=ax, label='T={0:0.2f}'.format(dn['t_data'][0].item()))
    dt[:,-1].plot(ax=ax, label='T={0:0.2f}'.format(dn['t_data'][-1].item()))
    ax.set_title('t LUT at t=const')
    ax.set_xlabel('n $(cm^{-3})$')
    ax.set_ylabel('$\Delta t/t$ (%)')
    ax.legend()

    
    
    n_M = np.zeros_like(n)
    t_M = np.zeros_like(t)
    s_M = np.zeros_like(s)
    sV_M = np.zeros_like(sV)
    f_M = np.zeros_like(f)
    sv_rel_M = np.zeros_like(sv_rel_max)
    
    
    
    def plot_lut_ts(n, t, s, sv, s_max_moms,
                n_max, t_max, s_max, sv_max, sv_rel_max,
                n_lut, t_lut, s_lut, sv_lut, sv_rel_lut):

        fig, axes = plt.subplots(nrows=5, ncols=1, squeeze=False, figsize=(6.5, 9))
        plt.subplots_adjust(top=0.95)

        # Error in the adjusted look-up table
        dn_max = (n - n_max) / n * 100.0
        dn_lut = (n - n_lut) / n * 100.0

        ax = axes[0,0]
        l1 = dn_max.plot(ax=ax, label='$\Delta n_{'+species+',Max}/n_{'+species+',Max}$')
        l2 = dn_lut.plot(ax=ax, label='$\Delta n_{'+species+',lut}/n_{'+species+',lut}$')
        ax.set_title('')
        ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('$\Delta n_{'+species+'}/n_{'+species+'}$ (%)')
        # util.format_axes(ax, xaxis='off')
        util.add_legend(ax, [l1[0], l2[0]], corner='SE', horizontal=True)
    
        # Deviation in temperature
        dt_max = (t - t_max) / t * 100.0
        dt_lut = (t - t_lut) / t * 100.0
    
        ax = axes[1,0]
        l1 = dt_max.plot(ax=ax, label='$\Delta T_{'+species+',Max}/T_{'+species+',Max}$')
        l2 = dt_lut.plot(ax=ax, label='$\Delta T_{'+species+',lut}/T_{'+species+',lut}$')
        ax.set_title('')
        ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('$\Delta T_{'+species+'}/T_{'+species+'}$ (%)')
        ax.set_ylim(-1,2.5)
        # util.format_axes(ax, xaxis='off')
        util.add_legend(ax, [l1[0], l2[0]], corner='NE', horizontal=True)
    
        # Deviation in entropy
        ds_moms = (s - s_max_moms) / s * 100.0
        ds_max = (s - s_max) / s * 100.0
        ds_lut = (s - s_lut) / s * 100.0
    
        ax = axes[2,0]
        l1 = ds_max.plot(ax=ax, label='$\Delta s_{'+species+',Max}/s_{'+species+',Max}$')
        l2 = ds_lut.plot(ax=ax, label='$\Delta s_{'+species+',lut}/s_{'+species+',lut}$')
        l3 = ds_moms.plot(ax=ax, label='$\Delta s_{'+species+',moms}/s_{'+species+',moms}$')
        ax.set_title('')
        ax.set_xticklabels([])
        ax.set_xlabel('')
        ax.set_ylabel('$\Delta s_{'+species+'}/s_{'+species+'}$ (%)')
        ax.set_ylim(-9,2.5)
        # util.format_axes(ax, xaxis='off')
        util.add_legend(ax, [l1[0], l2[0], l3[0]], corner='SE', horizontal=True)
    
        # Deviation in velocity-space entropy
        dsv_max = (sv - sv_max) / sv * 100.0
        dsv_lut = (sv - sv_lut) / sv * 100.0
    
        ax = axes[3,0]
        l1 = dsv_max.plot(ax=ax, label='$\Delta s_{V,'+species+',Max}/s_{V,'+species+',Max}$')
        l2 = dsv_lut.plot(ax=ax, label='$\Delta s_{V,'+species+',lut}/s_{V,'+species+',lut}$')
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_xticklabels([])
        ax.set_ylabel('$\Delta s_{V,'+species+'}/s_{V,'+species+'}$ (%)')
        # util.format_axes(ax)
        util.add_legend(ax, [l1[0], l2[0]], corner='SE', horizontal=True)
    
        ax = axes[4,0]
        l1 = sv_rel_max.plot(ax=ax, label='$s_{V,'+species+',rel,Max}$')
        l2 = sv_rel_M.plot(ax=ax, label='$s_{V,'+species+',rel,lut}$')
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('$s_{V,'+species+',rel}$\n[J/K/$m^{3}$]')
        # util.format_axes(ax)
        util.add_legend(ax, [l1[0], l2[0]], corner='SE', horizontal=True)

        fig.suptitle('Maxwellian Look-up Table')
    
    # Minimize density error
        for idx, dens in enumerate(N):
            imin = np.argmin(np.abs(lut['N'].data - dens.item()))
            irow = imin // dims[1]
            icol = imin % dims[1]
            n_M[idx] = lut['N'][irow, icol]
            t_M[idx] = lut['t'][irow, icol]
            s_M[idx] = lut['s'][irow, icol]
            sV_M[idx] = lut['sv'][irow, icol]
            f_M[idx,...] = lut['f'][irow, icol, ...]

    sv_rel_M = relative_entropy(f, f_M)
    plot_lut_ts(n, t, s, sV, s_max_moms,
            n_max, t_max, s_max, sv_max, sv_rel_max,
            n_M, t_M, s_M, sV_M, sv_rel_M)
    
    # Minimize temperature error
    for idx, temp in enumerate(t):
            imin = np.argmin(np.abs(lut['t'].data - temp.item()))
            irow = imin // dims[1]
            icol = imin % dims[1]
            n_M[idx] = lut['N'][irow, icol]
            t_M[idx] = lut['t'][irow, icol]
            s_M[idx] = lut['s'][irow, icol]
            sV_M[idx] = lut['sv'][irow, icol]
            f_M[idx,...] = lut['f'][irow, icol, ...]

    sv_rel_M = relative_entropy(f, f_M)
    
    plot_lut_ts(n, t, s, sV, s_max_moms,
            n_max, t_max, s_max, sv_max, sv_rel_max,
            n_M, t_M, s_M, sV_M, sv_rel_M)
    
        # Minimize error in both density and temperature
    for idx, (dens, temp) in enumerate(zip(N, t)):
            imin = np.argmin(np.sqrt((lut['t'].data - temp.item())**2
                                + (lut['N'].data - dens.item())**2
                                ))
            irow = imin // dims[1]
            icol = imin % dims[1]
            n_M[idx] = lut['N'][irow, icol]
            t_M[idx] = lut['t'][irow, icol]
            s_M[idx] = lut['s'][irow, icol]
            sV_M[idx] = lut['sv'][irow, icol]
            f_M[idx,...] = lut['f'][irow, icol, ...]

    sv_rel_M = relative_entropy(f, f_M)
    
    
    plot_lut_ts(n, t, s, sV, s_max_moms,
            n_max, t_max, s_max, sv_max, sv_rel_max,
            n_M, t_M, s_M, sV_M, sv_rel_M)
    
    
        # Interpolate
    lut_interp = lut.interp({'N_data': N, 't_data': t}, method='linear')

    lut_interp['sv_rel_M'] = relative_entropy(f, lut_interp['f'])
    
    
    plot_lut_ts(n, t, s, sV, s_max_moms,
            n_max, t_max, s_max, sv_max, sv_rel_max,
            lut_interp['N'], lut_interp['t'], lut_interp['s'], lut_interp['sv'], lut_interp['sv_rel_M'])
    
    
    
        # Sample interval
    delta_t = 1.0 / (float(np.diff(sv_rel_M['time']).mean()) * 1e-9)

        # Gradiated of the relative entropy computed via centered difference
    d_sv_rel_M_dt = xr.DataArray(np.gradient(sv_rel_M / n_M, delta_t),
                             dims=('time',),
                             coords={'time': sv_rel_M['time']})

        # Increment of the relative energy per particle
    d_E_rel_dt = 1e-6 * eV2K * t_M * d_sv_rel_M_dt # J/s = W
    
    fig, axes = plt.subplots(nrows=3, ncols=1, squeeze=False, figsize=(6.5,5))

    ax = axes[0,0]
    sv_rel_M.plot(ax=ax)
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$s_{'+species+'V,rel}$\n[J/K/$m^{3}$]')

    ax = axes[1,0]
    d_sv_rel_M_dt.plot(ax=ax)
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$ds_{'+species+'V,rel}/dt$\n[J/K/$m^{3}$/s]')
    ax = axes[2,0]
    d_E_rel_dt.plot(ax=ax)
    ax.set_ylabel('$d\mathcal{E}_{'+species+'V,rel}/dt$\n[W]')
    
    
#start_date = dt.datetime(2017, 7, 11, 22, 34, 0)
#end_date = dt.datetime(2017, 7, 11, 22, 34, 5)
#rel_entropy('mms3','brst','e', start_date, end_date)
