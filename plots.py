import numpy as np
import xarray as xr
from pathlib import Path
from matplotlib import pyplot as plt, dates as mdates, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pymms.data import fpi
from pymms import config
data_root = Path(config['data_root'])

import database, physics

# For now, take the plotting tools from pymms
#   - Tools are not in the `pip install pymms` library
#   - Change directories so they are discoverable
import os
os.chdir('/Users/argall/Documents/Python/pymms/examples/')
# os.chdir(r"D:\uni UNH\mms\pymms\examples\\")
# os.chdir('/Users/krmhanieh/Documents/GitHub/pymms/examples')
import util


data_path = Path(config['dropbox_root'])

def dissipation_measures(t0, t1):

    # Create a file name
    filename = '_'.join(('mms', 'hgio',
                         t0.strftime('%Y%m%d%_H%M%S'),
                         t1.strftime('%Y%m%d%_H%M%S')))
    file_path = (data_path / filename).with_suffix('.nc')
    if ~file_path.exists():
        file_path = database.load_data(t0, t1)

    # Load the data
    data = xr.load_dataset(file_path)
    
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

    # Plot the data
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
    
    # Electron Frame Dissipation Measure
    ax = axes[0,0]
    De_moms['De1'].plot(ax=ax, color='Black', label='MMS1')
    De_moms['De2'].plot(ax=ax, color='Blue', label='MMS1')
    De_moms['De3'].plot(ax=ax, color='Green', label='MMS1')
    De_moms['De4'].plot(ax=ax, color='Red', label='MMS1')
    De_curl.plot(ax=ax, color='magenta', label='Curl')
    ax.set_title('')
    ax.set_ylabel('De [$nW/m^{3}$]')
    ax.legend()

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
