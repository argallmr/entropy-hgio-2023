#!pip install numpy==1.24.3
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt, dates as mdates
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from pymms.data import fgm, fpi

def gyrotropy(sc , mode ,species, t0,t1, ):
#    mass = me if species == 'e' else mp
    optdesc = 'd{0}s-moms'.format(species)
    
    des_data = fpi.load_moms(sc=sc, mode=mode, optdesc=optdesc, start_date=t0, end_date=t1)
    fgm_data = fgm.load_data(sc=sc, mode=mode, start_date=t0, end_date=t1)
    B = fgm_data['B_GSE'].interp_like(des_data, method='linear')

    # Unit vector
    b = B[:,0:3] / np.linalg.norm(B[:,0:3], axis=1)[:, np.newaxis]
    #I1
    I1 = des_data['prestensor'][:,0,0] + des_data['prestensor'][:,1,1] + des_data['prestensor'][:,2,2]
    I1 = I1.drop(['cart_index_dim1', 'cart_index_dim2'])
    #I2=PxxPyy+PxxPzz+PyyPzz−(PxyPyx+PxzPzx+PyzPzy)
    I2= (des_data['prestensor'][:,0,0] * des_data['prestensor'][:,1,1]
         + des_data['prestensor'][:,0,0] * des_data['prestensor'][:,2,2]
         + des_data['prestensor'][:,1,1] * des_data['prestensor'][:,2,2]
         - (des_data['prestensor'][:,0,1] * des_data['prestensor'][:,1,0]
        + des_data['prestensor'][:,0,2] * des_data['prestensor'][:,2,0]
        + des_data['prestensor'][:,1,2] * des_data['prestensor'][:,2,1]
            )
        )
    # P.b
    p_par = des_data['prestensor'].dot(b.rename({'b_index': 'cart_index_dim2'}), dims=('cart_index_dim2',))

    # b.(P.b)
    p_par = b.dot(p_par.rename({'cart_index_dim1': 'b_index'}), dims=('b_index',))

# p_par = (b[:,0]**2 * des_data['prestensor'][:,0,0]
#          + b[:,1]**2 * des_data['prestensor'][:,1,1]
#          + b[:,2]**2 * des_data['prestensor'][:,2,2]
#          + 2 * (b[:,0]*B[:,1]*des_data['prestensor'][:,0,1]
#                 + b[:,0]*B[:,2]*des_data['prestensor'][:,0,2]
#                 + b[:,1]*B[:,2]*des_data['prestensor'][:,1,2]
#                 )
#          )
    Q = 1 - ((4*I2) / ((I1-p_par) * (I1 + 3*p_par)))
    #This is not comlete.
    nrows = 7
    ncols = 1
    figsize = (7.0, 10.0)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                         figsize=figsize, squeeze=False)
    plt.subplots_adjust(right=0.85)


    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)

# Magnetic Field
    ax = axes[0,0]
    B[:,0].plot(ax=ax, label='Bx')
    B[:,1].plot(ax=ax, label='By')
    B[:,2].plot(ax=ax, label='Bz')
    ax.set_title('Agyrotropy')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('B [nT]')

#
# DES Energy Spectra
#

# Energy flux is a function of time and energy: F(t, E)
#   - pcolorfast requires time and energy to be nt x nE, where nt (nE) is the
#     number of time (energy) points
#   - Energy is already time dependent, but we have to repeat the time axis
#   - These define the bottom-left corner of the heatmap pixels
#   - Except the last point in the array is the top-right corner of the last pixel
    nt = des_data['omnispectr'].shape[0]
    nE = des_data['omnispectr'].shape[1]
    x0 = mdates.date2num(des_data['time'])[:, np.newaxis].repeat(nE, axis=1)
    x1 = des_data['energy']

# Plot the log of the flux
#   - Convert 0 values to 1 so that log(1) = 0 (should probably replace with NaN)
#   - The (x0, x1) coordinates should have one more point than the data because they are
#     the last two points define the bottom-left and upper-right points of the last pixel
#   - I should extend (x0, x1) by (dx0, dx1), but since dx1 (energy) is not constant, it
#     is easier to trip one point from F
#   - I should also move x0 (time) to the beginning of the sample interval
#     instead of the end
    y = np.where(des_data['omnispectr'] == 0, 1, des_data['omnispectr'])
    y = np.log(y[0:-1,0:-1])

# Create an image and add it to the axes
#ax = axes[1,0]
#im = ax.pcolorfast(x0, x1, y, cmap='nipy_spectral')
#ax.images.append(im)
#ax.xaxis.set_major_locator(locator)
#ax.xaxis.set_major_formatter(formatter)
#ax.set_xticklabels([])
#ax.set_xlabel('')
#ax.set_yscale('log')
#ax.set_ylabel('E e-\n(eV)')

    ax = axes[1, 0]

# Plot using pcolorfast directly
    im = ax.pcolorfast(x0, x1, y, cmap='nipy_spectral')

# Set other plot customizations
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_yscale('log')
    ax.set_ylabel('E e-\n(eV)')

# Show the colorbar (you can customize its position as needed)
#fig.colorbar(im, ax=ax)


# Create a colorbar axis just outside the right-edge of the plot
    cbaxes = inset_axes(ax,
                    width='2%', height='100%', loc=4,
                    bbox_to_anchor=(0, 0, 1.05, 1),
                    bbox_transform=ax.transAxes,
                    borderpad=0)
    cb = plt.colorbar(im, cax=cbaxes, orientation='vertical')
    cb.set_label('$log_{10}$DEF')

#
# DONE: DES Energy Spectra
#

# Trace
    ax = axes[2,0]
    I1.plot(ax=ax)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$I_{1}$ [nPa]')

# Principal Minors
    ax = axes[3,0]
    I2.plot(ax=ax)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$I_{2}$ [nPa]$^{2}$')

# Parallel Pressure
    ax = axes[4,0]
    p_par.plot(ax=ax)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('$P_{\parallel}$ [nPa]')

# Agyrotropy
    ax = axes[5,0]
    Q.plot(ax=ax)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_xticklabels([])
    ax.set_ylabel('Q')

# Agyrotropy - Sqrt
    ax = axes[6,0]
    np.sqrt(Q).plot(ax=ax)
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('$\sqrt{Q}$')

# Format the time axis nicer
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    for tick in ax.get_xticklabels():
        tick.set_rotation(45)

# Make all of the axes have the exact time range
    _ = plt.setp(axes, xlim=(t0, t1))
