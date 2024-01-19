import numpy as np
import xarray as xr
from scipy import constants as c
from pathlib import Path
from matplotlib import pyplot as plt, dates as mdates, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pymms.data import fpi
from pymms import config
data_root = Path(config['data_root'])

import os
os.chdir('/Users/argall/Documents/Python/projects/entropy-hgio-2023/')
import database, physics, tools

# For now, take the plotting tools from pymms
#   - Tools are not in the `pip install pymms` library
#   - Change directories so they are discoverable
<<<<<<< HEAD
os.chdir('/Users/argall/Documents/Python/pymms/examples/')
# os.chdir(r"D:\uni UNH\mms\pymms\examples\\")
# os.chdir('/Users/krmhanieh/Documents/GitHub/pymms/examples')
=======
import os
#os.chdir('/Users/argall/Documents/Python/pymms/examples/')
#os.chdir(r"D:\uni UNH\mms\pymms\examples\\")
os.chdir('/Users/krmhanieh/Documents/GitHub/pymms/examples')
>>>>>>> 6ac4346c547d96b9601958f9b8483d89d443e061
import util

eV2K = c.value('electron volt-kelvin relationship')


def overview(sc, t0, t1, mode='srvy'):

    # Create the file name
    fname = database.filename(mode, t0, t1)

    # Load the data
    data = xr.load_dataset(fname)

    RE = 6378 # km
    sc = sc[3]

    fig, axes = plt.subplots(nrows=7, ncols=1, squeeze=False, figsize=(8,7))

    # Magnetic Field
    ax = axes[0,0]
    data['B'+sc][:,0].plot(ax=ax, label='Bx', color='blue')
    data['B'+sc][:,1].plot(ax=ax, label='By', color='green')
    data['B'+sc][:,2].plot(ax=ax, label='Bz', color='red')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('B\n[nT]')
    ax.legend()

    # Electric Field
    ax = axes[1,0]
    data['E'+sc][:,0].plot(ax=ax, label='Ex', color='blue')
    data['E'+sc][:,1].plot(ax=ax, label='Ey', color='green')
    data['E'+sc][:,2].plot(ax=ax, label='Ez', color='red')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('E\n[mV/m]')
    ax.legend()

    # Plasma Density
    ax = axes[2,0]
    data['ni'+sc].plot(ax=ax, label='ni', color='blue')
    data['ne'+sc].plot(ax=ax, label='ne', color='red')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('n\n[$cm^{-3}$]')
    ax.legend()

    # Ion Velocity
    ax = axes[3,0]
    data['Vi'+sc][:,0].plot(ax=ax, label='Vx', color='blue')
    data['Vi'+sc][:,1].plot(ax=ax, label='Vy', color='green')
    data['Vi'+sc][:,2].plot(ax=ax, label='Vz', color='red')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('Vi\n[$km/s$]')
    ax.legend()

    # Electron Velocity
    ax = axes[4,0]
    data['Ve'+sc][:,0].plot(ax=ax, label='Vx', color='blue')
    data['Ve'+sc][:,1].plot(ax=ax, label='Vy', color='green')
    data['Ve'+sc][:,2].plot(ax=ax, label='Vz', color='red')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('Ve\n[$km/s$]')
    ax.legend()

    # Scalar Pressure
    ax = axes[5,0]
    data['pi'+sc].plot(ax=ax, label='pi', color='blue')
    data['pe'+sc].plot(ax=ax, label='pe', color='red')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('pe\n[nPa]')
    ax.legend()

    # Location
    ax = axes[6,0]
    (data['r'+sc][:,0]/RE).plot(ax=ax, label='x', color='blue')
    (data['r'+sc][:,1]/RE).plot(ax=ax, label='y', color='green')
    (data['r'+sc][:,2]/RE).plot(ax=ax, label='z', color='red')
    ax.set_title('')
    ax.set_ylabel('r\n[$R_{E}$]')
    ax.legend()

    plt.show()


def dissipation_measures(t0, t1, mode='srvy'):

    fname = database.filename(mode, t0, t1)

    # Load the data
    data = xr.load_dataset(fname)
    
    # Calculate electron frame dissipation measure
    De_moms = xr.Dataset()
    for idx in range(1, 5):
        sc = str(idx)
        De_moms['De'+sc] = physics.De_moms(data['E'+sc], data['B'+sc],
                                           data['ne'+sc], data['Vi'+sc],
                                           data['Ve'+sc])
    
    De_curl = physics.De_curl(data[['E1', 'E2', 'E3', 'E4']],
                              data[['B1', 'B2', 'B3', 'B4']],
                              data[['Ve1', 'Ve2', 'Ve3', 'Ve4']],
                              data[['r1', 'r2', 'r3', 'r4']])

    # Calculate p-theta
    ptheta = physics.pressure_dilatation(data[['r1', 'r2', 'r3', 'r4']],
                                         data[['Ve1', 'Ve2', 'Ve3', 'Ve4']],
                                         data[['pe1', 'pe2', 'pe3', 'pe4']].rename({'pe1': 'p1', 'pe2': 'p2', 'pe3': 'p3', 'pe4': 'p4'}),
                                         )

    # Calculate Pi-D
    PiD = physics.PiD(data[['r1', 'r2', 'r3', 'r4']],
                      data[['Ve1', 'Ve2', 'Ve3', 'Ve4']],
                      data[['pe1', 'pe2', 'pe3', 'pe4']].rename({'pe1': 'p1',
                                                                 'pe2': 'p2',
                                                                 'pe3': 'p3',
                                                                 'pe4': 'p4'}),
                      data[['Pe1', 'Pe2', 'Pe3', 'Pe4']].rename({'Pe1': 'P1',
                                                                 'Pe2': 'P2',
                                                                 'Pe3': 'P3',
                                                                 'Pe4': 'P4'}))

    # Change in relative energy
    d_E_rel_dt = physics.relative_energy_d_dt('mms1', mode, 'des-dist', t0, t1)

    # Smooth the data
    PiD = tools.smooth(PiD, 0.5)
    ptheta = tools.smooth(ptheta, 0.5)
    d_E_rel_dt = tools.smooth(d_E_rel_dt, 0.5)

    # Plot the data
    fig, axes = plt.subplots(nrows=4, ncols=1, squeeze=False)
    plt.subplots_adjust(left=0.15, right=0.88, top=0.98, bottom=0.15)
    
    # Electron Frame Dissipation Measure
    ax = axes[0,0]
    De_moms['De1'].plot(ax=ax, color='Black', label='MMS1')
    De_moms['De2'].plot(ax=ax, color='Blue', label='MMS2')
    De_moms['De3'].plot(ax=ax, color='Green', label='MMS3')
    De_moms['De4'].plot(ax=ax, color='Red', label='MMS4')
    De_curl.plot(ax=ax, color='magenta', label='Curl')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('De\n[$nW/m^{3}$]')
    util.add_legend(ax, ax.get_lines(), corner='NE', outside=True)
    
    # p-theta
    ax = axes[1,0]
    ptheta.plot(ax=ax)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$p\\theta$\n[$nW/m^{3}$]')
    
    # Pi-D
    ax = axes[2,0]
    PiD.plot(ax=ax)
    ax.set_title('')
    ax.set_ylabel('$\Pi-D$\n[$nW/m^{3}$]')
    ax.set_xlabel('')
    ax.set_xticklabels([])

    # d/dt E_rel
    ax = axes[3,0]
    d_E_rel_dt.plot(ax=ax)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('$n dE_{rel}/dt$\n[$nW/m^{3}$]')
    util.format_axes(ax)

    plt.setp(axes, xlim=(t0, t1))

    plt.show()


def max_lut(sc, mode, optdesc, start_date, end_date):
    '''
    Plot a Maxwellian Look-Up Table (LUT). If a LUT file
    is not found, a LUT is created.

    Parameters
    ----------
    sc : str
        MMS spacecraft identifier ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Operating mode ('brst', 'srvy', 'fast')
    optdesc : str
        Filename optional descriptor ('dis-dist', 'des-dist')
    start_date, end_date : `datetime.datetime`
        Start and end of the time interval
    '''
    species = optdesc[1]
    
    # Create/Find the LUT
    lut_file = database.max_lut_load(sc, mode, optdesc, start_date, end_date)
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

    plt.show()


def max_lut_error(sc, mode, optdesc, start_date, end_date):
    '''
    Plot a Maxwellian Look-Up Table (LUT). If a LUT file
    is not found, a LUT is created.

    Parameters
    ----------
    sc : str
        MMS spacecraft identifier ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Operating mode ('brst', 'srvy', 'fast')
    optdesc : str
        Filename optional descriptor ('dis-dist', 'des-dist')
    start_date, end_date : `datetime.datetime`
        Start and end of the time interval
    '''

    # Get the LUT
    lut_file = database.max_lut_load(sc, mode, optdesc, start_date, end_date)
    lut = xr.load_dataset(lut_file)

    #
    #  Measured parameters
    #

    # Measured distrubtion function
    f = database.max_lut_precond_f(sc, mode, optdesc, start_date, end_date)
    
    # Moments and entropy parameters for the measured distribution
    n = fpi.density(f)
    V = fpi.velocity(f, N=n)
    T = fpi.temperature(f, N=n, V=V)
    P = fpi.pressure(f, N=n, T=T)
    t = ((T[:,0,0] + T[:,1,1] + T[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    p = ((P[:,0,0] + P[:,1,1] + P[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    s = fpi.entropy(f)
    sV = fpi.vspace_entropy(f, N=n, s=s)

    #
    #  Equivalent Maxwellian parameters
    #
    
    # Create equivalent Maxwellian distributions and calculate moments
    f_max = fpi.maxwellian_distribution(f, N=n, bulkv=V, T=t)
    
    s_max_moms = fpi.maxwellian_entropy(n, p)
    n_max = fpi.density(f_max)
    V_max = fpi.velocity(f_max, N=n_max)
    T_max = fpi.temperature(f_max, N=n_max, V=V_max)
    s_max = fpi.entropy(f_max)
    sV_max = fpi.vspace_entropy(f, N=n_max, s=s_max)
    t_max = ((T_max[:,0,0] + T_max[:,1,1] + T_max[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    sV_rel_max = physics.relative_entropy(f, f_max)

    #
    #  Optimized equivalent Maxwellian parameters
    #
    
    # Equivalent Maxwellian distribution function
    opt_lut = database.max_lut_optimize(lut, f, n, t, method='nt')

    sV_rel_opt = physics.relative_entropy(f, opt_lut['f_M'])

    #
    #  Plot
    #

    # Create the plot
    fix, axes = _max_lut_err(n, t, s, sV, s_max_moms,
                             n_max, t_max, s_max, sV_max, sV_rel_max,
                             opt_lut['n_M'], opt_lut['t_M'], opt_lut['s_M'],
                             opt_lut['sV_M'], sV_rel_opt)

    plt.show()


def _max_lut_err(n, t, s, sv, s_max_moms,
                 n_max, t_max, s_max, sv_max, sv_rel_max,
                 n_lut, t_lut, s_lut, sv_lut, sv_rel_lut):

    species = 'e'

    fig, axes = plt.subplots(nrows=5, ncols=1, squeeze=False, figsize=(6.5, 9))
    plt.subplots_adjust(top=0.95, right=0.95, left=0.25)

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
    l2 = sv_rel_lut.plot(ax=ax, label='$s_{V,'+species+',rel,lut}$')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('$s_{V,'+species+',rel}$\n[J/K/$m^{3}$]')
    # util.format_axes(ax)
    util.add_legend(ax, [l1[0], l2[0]], corner='SE', horizontal=True)

    fig.suptitle('Maxwellian Look-up Table')

    return fig, axes


def relative_entropy(sc, mode, optdesc, start_date, end_date):
    '''
    Plot a Maxwellian Look-Up Table (LUT). If a LUT file
    is not found, a LUT is created.

    Parameters
    ----------
    sc : str
        MMS spacecraft identifier ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Operating mode ('brst', 'srvy', 'fast')
    optdesc : str
        Filename optional descriptor ('dis-dist', 'des-dist')
    start_date, end_date : `datetime.datetime`
        Start and end of the time interval
    '''
    species = optdesc[1]

    # Get the LUT
    lut_file = database.max_lut_load(sc, mode, optdesc, start_date, end_date)
    lut = xr.load_dataset(lut_file)

    #
    #  Measured parameters
    #

    # Measured distrubtion function
    f = database.max_lut_precond_f(sc, mode, optdesc, start_date, end_date)
    
    # Moments and entropy parameters for the measured distribution
    n = fpi.density(f)
    V = fpi.velocity(f, N=n)
    T = fpi.temperature(f, N=n, V=V)
    t = ((T[:,0,0] + T[:,1,1] + T[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])

    #
    #  Optimized equivalent Maxwellian parameters
    #
    
    # Equivalent Maxwellian distribution function
    opt_lut = database.max_lut_optimize(lut, f, n, t, method='nt')

    #
    #  Relative velocity space entropy
    #

    sV_rel_opt = physics.relative_entropy(f, opt_lut['f_M'])

    # Sample interval
    delta_t = 1.0 / (float(np.diff(sV_rel_opt['time']).mean()) * 1e-9)

    # Gradiated of the relative entropy computed via centered difference
    d_sV_rel_dt = xr.DataArray(np.gradient(sV_rel_opt / opt_lut['n_M'], delta_t),
                               dims=('time',),
                               coords={'time': sV_rel_opt['time']})

    # Increment of the relative energy per particle
    d_E_rel_dt = 1e-6 * eV2K * opt_lut['t_M'] * d_sV_rel_dt # J/s = W

    #
    #  Plot
    #
    
    fig, axes = plt.subplots(nrows=3, ncols=1, squeeze=False, figsize=(6.5,5))
    plt.subplots_adjust(left=0.15, right=0.95, bottom=0.15, top=0.9)

    ax = axes[0,0]
    sV_rel_opt.plot(ax=ax)
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$s_{'+species+'V,rel}$\n[J/K/$m^{3}$]')

    ax = axes[1,0]
    d_sV_rel_dt.plot(ax=ax)
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$ds_{'+species+'V,rel}/dt$\n[J/K/$m^{3}$/s]')
    ax = axes[2,0]
    d_E_rel_dt.plot(ax=ax)
    ax.set_ylabel('$d\mathcal{E}_{'+species+'V,rel}/dt$\n[W]')

    plt.show()

    
