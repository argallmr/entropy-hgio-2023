import numpy as np
import xarray as xr
from scipy import constants as c
from pathlib import Path
from matplotlib import pyplot as plt, dates as mdates, cm, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Create label locator
locator = mdates.AutoDateLocator()
formatter = mdates.ConciseDateFormatter(locator)

from pymms.data import fpi
from pymms import config
data_root = Path(config['data_root'])

eV2K = c.value('electron volt-kelvin relationship')

def format_axes(ax, xaxis='on', yaxis='on', time=True):
    '''
    Format the abcissa and ordinate axes

    Parameters
    ----------
    ax : `matplotlib.pyplot.Axes`
        Axes to be formatted
    time : bool
        If true, format the x-axis with dates
    xaxis, yaxis : str
        Indicate how the axes should be formatted. Options are:
        ('on', 'time', 'off'). If 'time', the ConciseDateFormatter is applied
        If 'off', the axis label and ticklabels are suppressed. If 'on', the
        default settings are used
    '''
    
    # All x-axes should be formatted with time
    if time:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
    else:
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    if xaxis == 'off':
        ax.set_xticklabels([])
        ax.set_xlabel('')
    
    if ax.get_yscale() == 'log':
        locmaj = ticker.LogLocator(base=10.0)
        ax.yaxis.set_major_locator(locmaj)
        
        locmin = ticker.LogLocator(base=10.0, subs=(0.3, 0.6, 0.9)) 
        ax.yaxis.set_minor_locator(locmin)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    else:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    
    if yaxis == 'off':
        ax.set_yticklabels([])
        ax.set_ylabel('')


def add_legend(ax, corner='NE', outside=False, horizontal=False,
               labelspacing=0.5):
    '''
    Add a legend to the axes. Legend elements will have the same color as the
    lines that they label.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes to which the legend is attached.
    lines : list of `matplotlib.lines.Line2D`
        The line elements that the legend format should match
    corner : str
        The bounding box of the legend will be tied to this corner:
        ('NE', 'NW', 'SE', 'SW')
    outside : bool
        The bounding box will extend outside the plot
    horizontal : bool
        The legend items will be placed in columns (side-by-side) instead of
        rows (stacked vertically)
    labelspacing : float
        Vertical space between legend labels, in font-size units
    '''
    
    lines = ax.get_lines()
    if horizontal:
        ncol = len(lines)
        columnspacing = 0.5
    else:
        ncol = 1
        columnspacing = 0.0
    
    if corner == 'NE':
        bbox_to_anchor = (1, 1)
        loc = 'upper left' if outside else 'upper right'
    elif corner == 'SE':
        bbox_to_anchor = (1, 0)
        loc = 'lower left' if outside else 'lower right'
    elif corner == 'NW':
        bbox_to_anchor = (0, 1)
        loc = 'upper right' if outside else 'upper left'
    elif corner == 'SW':
        bbox_to_anchor = (0, 0)
        loc = 'lower right' if outside else 'lower left'

    leg = ax.legend(bbox_to_anchor=bbox_to_anchor,
                    borderaxespad=0.0,
                    columnspacing=columnspacing,
                    handlelength=1,
                    handletextpad=0.25,
                    labelspacing=labelspacing,
                    loc=loc,
                    ncol=ncol)
    
    # Turn the frame off but still add it to the legend
    leg.get_frame().set_alpha(0)
    
    # Make the lines and labels have the same color
    #   - Labels that start with "_" are hidden from the legend
    #   - Empty labels are not added to the legend
    texts = leg.get_texts()
    idx = 0
    for line in lines:
        label = line.get_label()
        if (label != '') and (label[0] != '_'):
            texts[idx].set_color(line.get_color())
            idx += 1


def add_colorbar(ax, im):
    '''
    Add a colorbar to the axes.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axes to which the colorbar is attached.
    im : `matplotlib.axes.Axes.pcolorfast`
        The image that the colorbar will represent.
    '''
    cbaxes = inset_axes(ax,
                        width='2%', height='100%', loc=4,
                        bbox_to_anchor=(0, 0, 1.05, 1),
                        bbox_transform=ax.transAxes,
                        borderpad=0)
    cb = plt.colorbar(im, cax=cbaxes, orientation='vertical')
    cb.ax.minorticks_on()
    
    return cb


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


def dissipation_measures(t0, t1, mode='srvy', t_smooth=None):

    # Get the dataset
    fname = database.load_data(t0, t1, mode=mode)
    data = xr.load_dataset(fname)

    # Load the entropy data
    f_rel = database.filename('mms', mode, t0, t1, optdesc='des-dist-s')
    s_rel = xr.load_dataset(f_rel)
    
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

    # Smooth the entropy data
    if t_smooth is None:
        sV_rel = s_rel[['sV_rel1', 'sV_rel2', 'sV_rel3', 'sV_rel4']]
        n = s_rel[['sV_rel1', 'sV_rel2', 'sV_rel3', 'sV_rel4']]
        t = s_rel[['t1', 't2', 't3', 't4']]
        R = data[['r1', 'r2', 'r3', 'r4']]
    else:
        sV_rel = xr.Dataset()
        n = xr.Dataset()
        t = xr.Dataset()
        R = xr.Dataset()
        for name in ['sV_rel1', 'sV_rel2', 'sV_rel3', 'sV_rel4']:
            sV_rel[name] = tools.smooth(s_rel[name], t_smooth)
            # sV_rel[name] = tools.smooth(s_rel[name], 3,
            #                             weights=np.array([0.25, 0.5, 0.25]))
        for name in ['n1', 'n2', 'n3', 'n4']:
            n[name] = tools.smooth(s_rel[name], t_smooth)
            # n[name] = tools.smooth(s_rel[name], 3,
            #                        weights=np.array([0.25, 0.5, 0.25]))
        for name in ['t1', 't2', 't3', 't4']:
            t[name] = tools.smooth(s_rel[name], t_smooth)
            # t[name] = tools.smooth(s_rel[name], 3,
            #                             weights=np.array([0.25, 0.5, 0.25]))
        for name in ['r1', 'r2', 'r3', 'r4']:
            R[name] = tools.smooth(data[name], t_smooth)
        
        # Smooth pressure-strain interactions
        PiD = tools.smooth(PiD, t_smooth)
        ptheta = tools.smooth(ptheta, t_smooth)

    # Compute relative entropy
    pE_rel = physics.pE_rel_pt(R, sV_rel, n, t,
                               np.array([-232.0, 0, -59.0]))
    

    # Smooth the data
    # d_E_rel_dt = tools.smooth(d_E_rel_dt, 0.75)

    # Cut the data
    # s_rel_smooth = s_rel_smooth.sel(time=slice(s_rel_smooth['time'][3].data, None))
    pE_rel = pE_rel.sel(time=slice(pE_rel['time'][3].data, None))

    # Plot the data
    fig, axes = plt.subplots(nrows=4, ncols=1, squeeze=False, figsize=(7, 2))
    plt.subplots_adjust(left=0.18, right=0.88, top=0.98, bottom=0.15)
    
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
    add_legend(ax, ax.get_lines(), corner='NE', outside=True)
    
    # p-theta
    ax = axes[1,0]
    (-ptheta).plot(ax=ax, color='red')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$-p\\theta$\n[$nW/m^{3}$]')
    
    # Pi-D
    ax = axes[2,0]
    (-PiD).plot(ax=ax, color='blue')
    ax.set_title('')
    ax.set_ylabel('$-\Pi-D$\n[$nW/m^{3}$]')
    ax.set_xlabel('')
    ax.set_xticklabels([])

    # HORENET
    ax = axes[3,0]
    pE_rel['HORNET'].plot(ax=ax)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('$-nd\mathcal{E}_{'+'e'+'V,rel}/dt$\n[$nW/m^{3}$]')
    format_axes(ax)

    plt.setp(axes, xlim=(t0, t1))

    return fig, axes


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
    lut_file = database.load_max_lut(sc, mode, optdesc, start_date, end_date)
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
    format_axes(ax, time=False)

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

    return fig, axes


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
    lut_file = database.load_max_lut(sc, mode, optdesc, start_date, end_date)
    lut = xr.load_dataset(lut_file)

    #
    #  Measured parameters
    #

    # Measured distrubtion function
    f = database.max_lut_precond_f(sc, mode, optdesc, start_date, end_date)
    
    # Moments and entropy parameters for the measured distribution
    n = fpi.density(f)
    V = fpi.velocity(f, n=n)
    T = fpi.temperature(f, n=n, V=V)
    P = fpi.pressure(f, n=n, T=T)
    t = ((T[:,0,0] + T[:,1,1] + T[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    p = ((P[:,0,0] + P[:,1,1] + P[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])
    s = fpi.entropy(f)
    sV = fpi.vspace_entropy(f, n=n, s=s)

    #
    #  Equivalent Maxwellian parameters
    #
    
    # Create equivalent Maxwellian distributions and calculate moments
    f_max = fpi.maxwellian_distribution(f, n=n, bulkv=V, T=t)
    
    s_max_moms = fpi.maxwellian_entropy(n, p)
    n_max = fpi.density(f_max)
    V_max = fpi.velocity(f_max, n=n_max)
    T_max = fpi.temperature(f_max, n=n_max, V=V_max)
    s_max = fpi.entropy(f_max)
    sV_max = fpi.vspace_entropy(f, n=n_max, s=s_max)
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
    fig, axes = _max_lut_err(n, t, s, sV, s_max_moms,
                             n_max, t_max, s_max, sV_max, sV_rel_max,
                             opt_lut['n_M'], opt_lut['t_M'], opt_lut['s_M'],
                             opt_lut['sV_M'], sV_rel_opt)

    return fig, axes


def _max_lut_err(n, t, s, sv, s_max_moms,
                 n_max, t_max, s_max, sv_max, sv_rel_max,
                 n_lut, t_lut, s_lut, sv_lut, sv_rel_lut):

    species = 'e'

    fig, axes = plt.subplots(nrows=5, ncols=1, squeeze=False, figsize=(5.5, 7))
    plt.subplots_adjust(top=0.95, right=0.95, left=0.17)

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
    # format_axes(ax, xaxis='off')
    add_legend(ax, [l1[0], l2[0]], corner='SE', horizontal=True)

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
    # format_axes(ax, xaxis='off')
    add_legend(ax, [l1[0], l2[0]], corner='NE', horizontal=True)

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
    # format_axes(ax, xaxis='off')
    add_legend(ax, [l1[0], l2[0], l3[0]], corner='SE', horizontal=True)

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
    # format_axes(ax)
    add_legend(ax, [l1[0], l2[0]], corner='SE', horizontal=True)

    ax = axes[4,0]
    l1 = sv_rel_max.plot(ax=ax, label='$s_{V,'+species+',rel,Max}$')
    l2 = sv_rel_lut.plot(ax=ax, label='$s_{V,'+species+',rel,lut}$')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('$s_{V,'+species+',rel}$\n[J/K/$m^{3}$]')
    # format_axes(ax)
    add_legend(ax, [l1[0], l2[0]], corner='SE', horizontal=True)

    fig.suptitle('Maxwellian Look-up Table')

    return fig, axes


def max_lut_sample_error(sc, mode, optdesc, start_date, end_date, inset_loc=None):
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

    if inset_loc is None:
        inset_loc = 'NE'
    if inset_loc == 'NE':
        inset_loc = [0.5, 0.5, 0.47, 0.47] # [x0, y0, width, height]
    if inset_loc == 'SE':
        inset_loc = [0.0, 0.5, 0.47, 0.47]
    if inset_loc == 'NW':
        inset_loc = [0.0, 0.5, 0.47, 0.47]
    if inset_loc == 'SW':
        inset_loc = [0.0, 0.0, 0.47, 0.47]

    # Get the LUT
    lut_file = database.load_max_lut(sc, mode, optdesc, start_date, end_date)
    lut = xr.load_dataset(lut_file)

    #
    #  Measured parameters
    #

    # Read the data
    #   - This includes removal of the photo-electron model
    fpi_dist = fpi.load_dist(sc=sc, mode=mode, optdesc=optdesc,
                             start_date=start_date, end_date=end_date)

    kwargs = fpi.precond_params(sc, mode, 'l2', optdesc, start_date, end_date,
                                time=fpi_dist['time'])

    # f_pre = fpi.precondition(fpi_dist['dist'], **kwargs)

    t_sample = np.datetime64('2016-10-22T12:58:27.600819000')
    f = fpi_dist['dist'].sel(time=t_sample, method='nearest')
    f = fpi.Distribution_Function(f.data,
                                  f['phi'].data, f['theta'].data, f['energy'].data,
                                  mass=fpi.species_to_mass(species), time=f['time'].data,
                                  scpot=kwargs.pop('scpot').sel(time=t_sample, method='nearest').item(),
                                  **kwargs)

    # Moments and entropy parameters for the measured distribution
    n = f.density()
    V = f.velocity(N=n)
    t = f.scalar_temperature(N=n, V=V)

    # Equivalent Maxwellian
    f_M = f.maxwellian()
    n_M = f_M.density()
    V_M = f_M.velocity(N=n)
    t_M = f_M.scalar_temperature(N=n, V=V)

    #
    #  Equivalent Maxwellian parameters
    #
    
    # Create equivalent Maxwellian distributions and calculate moments
    # f_max = fpi.maxwellian_distribution(f, N=n, bulkv=V, T=t)

    #
    #  Optimized equivalent Maxwellian parameters
    #

    # Find the point that minimizes the sum squared error
    dims = lut['N'].shape
    imin = np.argmin(np.sqrt((lut['t'].data - t.item())**2
                              + (lut['N'].data - n.item())**2
                              ))
    irow = imin // dims[1]
    icol = imin % dims[1]

    n_lut = lut['N'][irow, icol]
    t_lut = lut['t'][irow, icol]

    # Find the point that minimizes the error in density
    imin = np.argmin(abs(lut['N'].data - n.item()))
    irow_n = imin // dims[1]
    icol_n = imin % dims[1]

    n_lut_n = lut['N'][irow_n, icol_n]
    t_lut_n = lut['t'][irow_n, icol_n]

    # Find the point that minimizes the error in temperature
    imin = np.argmin(abs(lut['t'].data - t.item()))
    irow_t = imin // dims[1]
    icol_t = imin % dims[1]

    n_lut_t = lut['N'][irow_t, icol_t]
    t_lut_t = lut['t'][irow_t, icol_t]

    # Nearest neighbor
    n_N = lut['N'].sel(N_data=n, t_data=t, method='nearest')
    t_N = lut['t'].sel(N_data=n, t_data=t, method='nearest')

    #
    # Error between LUT Maxwellians and measured distribution
    #
    dn_lut = 100 * (n - lut['N']) / n # %
    dt_lut = 100 * (t - lut['t']) / t # %

    #
    # Plots
    #

    # Columns are plotted as x, rows plotted as y, so take transpose
    N_lut = lut.N.transpose()
    T_lut = lut.t.transpose()
    dn_lut = dn_lut.transpose()
    dt_lut = dt_lut.transpose()

    # Plot the density and temperature from the lut
    # and the error for a single distribution function
    fig, axes = plt.subplots(nrows=2, ncols=2, squeeze=False)
    plt.subplots_adjust(wspace=0.8, hspace=0.4, left=0.09, right=0.87, top=0.95)

    #
    # 2D Density LUT
    #

    ax = axes[0,0]
    img = N_lut.plot(ax=ax, cmap=cm.get_cmap('rainbow', 15), add_colorbar=False,)
    ax.set_title('')
    ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
    ax.set_ylabel('$t_{'+species+'}$ (eV)')
    cb = add_colorbar(ax, img)
    cb.set_label('$n_{'+species+'}$ ($cm^{-3}$)')
    cb.ax.minorticks_on()
    
    # Plot the location of the measured density and temperature
    ax.plot(n, t, linestyle='None', marker='x', color='black')
    ax.plot(n_M, t_M, linestyle='None', marker=r'$\mathrm{M}$', color='black')
    ax.plot(n_lut, t_lut, linestyle='None', marker=r'$\mathrm{L}$', color='black')
    ax.plot(n_lut_n, t_lut_n, linestyle='None', marker=r'$\mathrm{n}$', color='black')
    ax.plot(n_lut_t, t_lut_t, linestyle='None', marker=r'$\mathrm{t}$', color='black')
    ax.plot(n_N, t_N, linestyle='None', marker=r'$\mathrm{N}$', color='black')

    # Inset axes to show 
    x1, x2, y1, y2 = 0.9*n, 1.1*n, 0.9*t, 1.1*t
    axins = ax.inset_axes(inset_loc,
                          xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    img = N_lut.plot(ax=axins, cmap=cm.get_cmap('rainbow', 15), add_colorbar=False, edgecolor='k')
    axins.set_xlabel('')
    axins.set_xlim(x1, x2)
    axins.set_ylabel('')
    axins.set_ylim(y1, y2)
    # axins.imshow(lut.N.data, extent=extent, cmap=cm.get_cmap('rainbow', 15), origin="lower")
    ax.indicate_inset_zoom(axins, edgecolor="black")

    # Plot the location of the measured density and temperature
    axins.plot(n, t, linestyle='None', marker='x', color='black')
    axins.plot(n_M, t_M, linestyle='None', marker=r'$\mathrm{M}$', color='black')
    axins.plot(n_lut, t_lut, linestyle='None', marker=r'$\mathrm{L}$', color='black')
    axins.plot(n_lut_n, t_lut_n, linestyle='None', marker=r'$\mathrm{n}$', color='black')
    axins.plot(n_lut_t, t_lut_t, linestyle='None', marker=r'$\mathrm{t}$', color='black')
    axins.plot(n_N, t_N, linestyle='None', marker='o', color='black')

    #
    # 2D Temperature LUT
    #
    ax = axes[0,1]
    img = T_lut.plot(ax=ax, cmap=cm.get_cmap('rainbow', 15), add_colorbar=False)
    ax.set_title('')
    ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
    ax.set_ylabel('$t_{'+species+'}$ (eV)')
    cb = add_colorbar(ax, img)
    cb.set_label('$t_{'+species+'}$ (eV)')
    cb.ax.minorticks_on()
    
    # Plot the location of the measured density and temperature
    ax.plot(n, t, linestyle='None', marker='x', color='black')
    ax.plot(n_M, t_M, linestyle='None', marker=r'$\mathrm{M}$', color='black')
    ax.plot(n_lut, t_lut, linestyle='None', marker=r'$\mathrm{L}$', color='black')
    ax.plot(n_lut_n, t_lut_n, linestyle='None', marker=r'$\mathrm{n}$', color='black')
    ax.plot(n_lut_t, t_lut_t, linestyle='None', marker=r'$\mathrm{t}$', color='black')
    ax.plot(n_N, t_N, linestyle='None', marker=r'$\mathrm{N}$', color='black')

    # Inset axes to show 
    x1, x2, y1, y2 = 0.9*n, 1.1*n, 0.9*t, 1.1*t
    axins = ax.inset_axes(inset_loc,
                          xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    img = T_lut.plot(ax=axins, cmap=cm.get_cmap('rainbow', 15), add_colorbar=False)
    axins.set_xlabel('')
    axins.set_xlim(x1, x2)
    axins.set_ylabel('')
    axins.set_ylim(y1, y2)
    # axins.imshow(lut.N.data, extent=extent, cmap=cm.get_cmap('rainbow', 15), origin="lower")
    ax.indicate_inset_zoom(axins, edgecolor="black")

    # Plot the location of the measured density and temperature
    axins.plot(n, t, linestyle='None', marker='x', color='black')
    axins.plot(n_M, t_M, linestyle='None', marker=r'$\mathrm{M}$', color='black')
    axins.plot(n_lut, t_lut, linestyle='None', marker=r'$\mathrm{L}$', color='black')
    axins.plot(n_lut_n, t_lut_n, linestyle='None', marker=r'$\mathrm{n}$', color='black')
    axins.plot(n_lut_t, t_lut_t, linestyle='None', marker=r'$\mathrm{t}$', color='black')
    axins.plot(n_N, t_N, linestyle='None', marker=r'$\mathrm{N}$', color='black')

    #
    # 2D Density LUT Error
    #

    ax = axes[1,0]
    img = dn_lut.plot(ax=ax, cmap=cm.get_cmap('rainbow', 15), add_colorbar=False)
    ax.set_title('')
    ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
    ax.set_ylabel('$t_{'+species+'}$ (eV)')
    cb = add_colorbar(ax, img)
    cb.set_label('$\Delta n_{'+species+'}$ (%)')
    cb.ax.minorticks_on()
    
    # Plot the location of the measured density and temperature
    ax.plot(n, t, linestyle='None', marker='x', color='black')
    ax.plot(n_M, t_M, linestyle='None', marker=r'$\mathrm{M}$', color='black')
    ax.plot(n_lut, t_lut, linestyle='None', marker=r'$\mathrm{L}$', color='black')
    ax.plot(n_lut_n, t_lut_n, linestyle='None', marker=r'$\mathrm{n}$', color='black')
    ax.plot(n_lut_t, t_lut_t, linestyle='None', marker=r'$\mathrm{t}$', color='black')
    ax.plot(n_N, t_N, linestyle='None', marker=r'$\mathrm{N}$', color='black')

    # Inset axes to show 
    x1, x2, y1, y2 = 0.9*n, 1.1*n, 0.9*t, 1.1*t
    axins = ax.inset_axes(inset_loc,
                          xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    img = dn_lut.plot(ax=axins, cmap=cm.get_cmap('rainbow', 15), add_colorbar=False)
    axins.set_xlabel('')
    axins.set_xlim(x1, x2)
    axins.set_ylabel('')
    axins.set_ylim(y1, y2)
    # axins.imshow(lut.N.data, extent=extent, cmap=cm.get_cmap('rainbow', 15), origin="lower")
    ax.indicate_inset_zoom(axins, edgecolor="black")

    # Plot the location of the measured density and temperature
    axins.plot(n, t, linestyle='None', marker='x', color='black')
    axins.plot(n_M, t_M, linestyle='None', marker=r'$\mathrm{M}$', color='black')
    axins.plot(n_lut, t_lut, linestyle='None', marker=r'$\mathrm{L}$', color='black')
    axins.plot(n_lut_n, t_lut_n, linestyle='None', marker=r'$\mathrm{n}$', color='black')
    axins.plot(n_lut_t, t_lut_t, linestyle='None', marker=r'$\mathrm{t}$', color='black')
    axins.plot(n_N, t_N, linestyle='None', marker=r'$\mathrm{N}$', color='black')

    #
    # 2D Temperature LUT Error
    #

    ax = axes[1,1]
    img = dt_lut.plot(ax=ax, cmap=cm.get_cmap('rainbow', 15), add_colorbar=False)
    ax.set_title('')
    ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
    ax.set_ylabel('$t_{'+species+'}$ (eV)')
    cb = add_colorbar(ax, img)
    cb.set_label('$\Delta t_{'+species+'}$ (%)')
    cb.ax.minorticks_on()
    
    # Plot the location of the measured density and temperature
    ax.plot(n, t, linestyle='None', marker='x', color='black')
    ax.plot(n_M, t_M, linestyle='None', marker=r'$\mathrm{M}$', color='black')
    ax.plot(n_lut, t_lut, linestyle='None', marker=r'$\mathrm{L}$', color='black')
    ax.plot(n_lut_n, t_lut_n, linestyle='None', marker=r'$\mathrm{n}$', color='black')
    ax.plot(n_lut_t, t_lut_t, linestyle='None', marker=r'$\mathrm{t}$', color='black')
    ax.plot(n_N, t_N, linestyle='None', marker=r'$\mathrm{N}$', color='black')

    # Inset axes to show 
    x1, x2, y1, y2 = 0.9*n, 1.1*n, 0.9*t, 1.1*t
    axins = ax.inset_axes(inset_loc,
                          xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    img = dt_lut.plot(ax=axins, cmap=cm.get_cmap('rainbow', 15), add_colorbar=False)
    axins.set_xlabel('')
    axins.set_xlim(x1, x2)
    axins.set_ylabel('')
    axins.set_ylim(y1, y2)
    # axins.imshow(lut.N.data, extent=extent, cmap=cm.get_cmap('rainbow', 15), origin="lower")
    ax.indicate_inset_zoom(axins, edgecolor="black")

    # Plot the location of the measured density and temperature
    axins.plot(n, t, linestyle='None', marker='x', color='black')
    axins.plot(n_M, t_M, linestyle='None', marker=r'$\mathrm{M}$', color='black')
    axins.plot(n_lut, t_lut, linestyle='None', marker=r'$\mathrm{L}$', color='black')
    axins.plot(n_lut_n, t_lut_n, linestyle='None', marker=r'$\mathrm{n}$', color='black')
    axins.plot(n_lut_t, t_lut_t, linestyle='None', marker=r'$\mathrm{t}$', color='black')
    axins.plot(n_N, t_N, linestyle='None', marker=r'$\mathrm{N}$', color='black')

    return fig, axes


def PiD(t0, t1, mode):

    # Get the dataset
    fname = database.load_data(t0, t1, mode=mode)
    data = xr.load_dataset(fname)

    #
    # p-theta Calculations
    #

    k = tools.recip_vec(data[['r1', 'r2', 'r3', 'r4']])

    # Theta: Divergence of electron velocity
    divU = tools.divergence(k, data[['Ve1', 'Ve2', 'Ve3', 'Ve4']])

    # Take the barycentric average of the scalar pressure
    p_bary = tools.barycentric_avg(data[['pe1', 'pe2', 'pe3', 'pe4']])

    # p-Theta: Pressure dilatation
    p_theta = p_bary * divU

    #
    # Pi-D Calculations
    #

    Pi = physics.traceless_pressure_tensor(data[['pe1', 'pe2', 'pe3', 'pe4']],
                                           data[['Pe1', 'Pe2', 'Pe3', 'Pe4']])

    D = physics.devoriak_pressure(data[['r1', 'r2', 'r3', 'r4']],
                                  data[['Ve1', 'Ve2', 'Ve3', 'Ve4']],
                                  data[['pe1', 'pe2', 'pe3', 'pe4']],
                                  data[['Pe1', 'Pe2', 'Pe3', 'Pe4']])
    
    # PiD
    PiD = Pi.dot(D, dims=('comp1', 'comp2'))

    #
    # Plot
    #

    fig, axes = plt.subplots(nrows=6, ncols=1, squeeze=False)

    # Div U
    ax = axes[0,0]
    divU.plot(ax=ax)
    ax.set_ylabel('$\theta = \nabla \cdot \mathbf{u}$\n[s^{-1}]')
    format_axes(ax, xaxis='off')

    # p
    ax = axes[1,0]
    p_bary.plot(ax=ax)
    ax.set_ylabel('$p$\n[nPa]')
    format_axes(ax, xaxis='off')

    # p-theta
    ax = axes[2,0]
    p_theta.plot(ax=ax)
    ax.set_ylabel('$p\theta$\n[nW/m^{3}]')
    format_axes(ax, xaxis='off')

    # Pi
    ax = axes[3,0]
    Pi.plot(ax=ax)
    ax.set_ylabel('$\Pi$\n[nPa]')
    format_axes(ax, xaxis='off')

    # D
    ax = axes[4,0]
    D.plot(ax=ax)
    ax.set_ylabel('$D$\n[s^{-1}]')
    format_axes(ax, xaxis='off')

    # Pi-D
    ax = axes[5,0]
    PiD.plot(ax=ax)
    ax.set_ylabel('$\Pi$-D\n[$nW/m_{3}$]')
    format_axes(ax)

    return fig, axes


def relative_entropy(mode, optdesc, start_date, end_date,
                     v_str=None, t_smooth=None):
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
    # Structure velocity
    if v_str is None:
        v_str = np.array([-232.0, 0, -59.0])

    species = optdesc[1]

    # Load standard data products
    fname = database.load_data(start_date, end_date, mode=mode)
    data = xr.load_dataset(fname)

    # Load the entropy data
    f_rel = database.filename('mms', mode, start_date, end_date, optdesc=optdesc+'-s')
    s_rel = xr.load_dataset(f_rel)

    # Smooth the entropy data
    if t_smooth is None:
        sV_rel = s_rel[['sV_rel1', 'sV_rel2', 'sV_rel3', 'sV_rel4']]
        n = s_rel[['n1', 'n2', 'n3', 'n4']]
        t = s_rel[['t1', 't2', 't3', 't4']]
        R = data[['r1', 'r2', 'r3', 'r4']]
    else:
        sV_rel = xr.Dataset()
        n = xr.Dataset()
        t = xr.Dataset()
        R = xr.Dataset()
        for name in ['sV_rel1', 'sV_rel2', 'sV_rel3', 'sV_rel4']:
            sV_rel[name] = tools.smooth(s_rel[name], t_smooth)
            # sV_rel[name] = tools.smooth(s_rel[name], 3,
            #                             weights=np.array([0.25, 0.5, 0.25]))
        for name in ['n1', 'n2', 'n3', 'n4']:
            n[name] = tools.smooth(s_rel[name], t_smooth)
            # n[name] = tools.smooth(s_rel[name], 3,
            #                        weights=np.array([0.25, 0.5, 0.25]))
        for name in ['t1', 't2', 't3', 't4']:
            t[name] = tools.smooth(s_rel[name], t_smooth)
            # t[name] = tools.smooth(s_rel[name], 3,
            #                             weights=np.array([0.25, 0.5, 0.25]))
        for name in ['r1', 'r2', 'r3', 'r4']:
            R[name] = tools.smooth(data[name], t_smooth)
            # t[name] = tools.smooth(s_rel[name], 3,
            #                             weights=np.array([0.25, 0.5, 0.25]))

    # Compute relative entropy
    pE_rel = physics.pE_rel_pt(R, sV_rel, n, t,
                               np.array(v_str))
    
    sV_rel = sV_rel.sel(time=slice(sV_rel['time'][3].data, None))
    pE_rel = pE_rel.sel(time=slice(pE_rel['time'][3].data, None))
    
    #
    #  Plot
    #
    
    fig, axes = plt.subplots(nrows=5, ncols=1, squeeze=False, figsize=(6.5,7))
    plt.subplots_adjust(left=0.15, right=0.88, bottom=0.1, top=0.95, hspace=0.2)

    ax = axes[0,0]
    sV_rel['sV_rel1'].plot(ax=ax, label='MMS1')
    sV_rel['sV_rel2'].plot(ax=ax, label='MMS2')
    sV_rel['sV_rel3'].plot(ax=ax, label='MMS3')
    sV_rel['sV_rel4'].plot(ax=ax, label='MMS4')
    ax.set_title('')
    ax.set_ylabel('$s_{'+species+'V,rel}$\n[J/K/$m^{3}$]')
    format_axes(ax, xaxis='off')

    ax = axes[1,0]
    pE_rel['sVr_n1'].plot(ax=ax, label='MMS1')
    pE_rel['sVr_n2'].plot(ax=ax, label='MMS2')
    pE_rel['sVr_n3'].plot(ax=ax, label='MMS3')
    pE_rel['sVr_n4'].plot(ax=ax, label='MMS4')
    pE_rel['sVr_n'].plot(ax=ax, label='Bary', color='magenta')
    ax.set_title('')
    ax.set_ylabel('$s_{'+species+'V,rel}/n$\n[J/K]')
    format_axes(ax, xaxis='off')
    add_legend(ax, outside=True)

    ax = axes[2,0]
    pE_rel['psV_rel1_pt'].plot(ax=ax, label='MMS1')
    pE_rel['psV_rel2_pt'].plot(ax=ax, label='MMS2')
    pE_rel['psV_rel3_pt'].plot(ax=ax, label='MMS3')
    pE_rel['psV_rel4_pt'].plot(ax=ax, label='MMS4')
    pE_rel['psV_rel_pt'].plot(ax=ax, label='Bary', color='magenta')
    ax.set_title('')
    ax.set_ylabel('$\partial (s_{'+species+'V,rel}/n)/\partial t$\n[J/K/s]')
    format_axes(ax, xaxis='off')

    ax = axes[3,0]
    pE_rel['dsV_rel_dr'].plot(ax=ax, label='$\mathbf{u} \cdot \\nabla$')
    ax.set_title('')
    ax.set_ylabel('$(\mathbf{u} \cdot \\nabla) (s_{'+species+'V,rel}/n)$\n[J/K/s]')
    format_axes(ax, xaxis='off')

    ax = axes[4,0]
    pE_rel['HORNET'].plot(ax=ax)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('$-nd\mathcal{E}_{'+species+'V,rel}/dt$\n[$nW/m^{3}$]')
    format_axes(ax)

    return fig, axes

    
