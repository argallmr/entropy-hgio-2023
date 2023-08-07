#!/usr/bin/env python
# coding: utf-8

# # Agyrotropy
# The amount of agyrotropy in a distribution function is defined as (Swisdak 2016):
# $$
# \begin{equation}
#     Q = \frac{P_{12}^{2} + P_{13}^{2} + P_{23}^{2}} {P_{\bot}^{2} + 2 P_{\bot} P_{\parallel}}
# \end{equation}
# $$
# where $\mathbb{P}$ is the pressure tensor
# $$
# \begin{equation}
#     \mathbb{P} = \begin{pmatrix}
#                  P_{11} & P_{12} & P_{13} \\
#                  P_{21} & P_{22} & P_{23} \\
#                  P_{31} & P_{32} & P_{33}
#                  \end{pmatrix}
# \end{equation}
# $$
# 
# The parallel and perpendicular components of the pressure tensor ($P_{\parallel}$ and $P_{\bot}$) are defined in a coordinate system with one axis aligned with the magnetic field. To determine them, we make use of two of the three tensor invariants -- the trace and the sum of the principal minors (the third invariant is the determinant) -- and the parallel pressure. The trace is given by
# $$
# \begin{equation}
#     I_{1} = P_{xx} + P_{yy} + P_{zz}
# \end{equation}
# $$
# 
# the sum of the principal minors is
# $$
# \begin{equation}
#     I_{2} = P_{xx} P_{yy} + P_{xx} P_{zz} + P_{yy} P_{zz} - \left( P_{xy} P_{yx} + P_{xz} P_{zx} + P_{yz} P_{zy} \right)
# \end{equation}
# $$
# 
# and the parallel pressure is
# $$
# \begin{align}
#     P_{\parallel} &= \hat{\mathbf{b}} \cdot \mathbb{P} \cdot \hat{\mathbf{b}} \\
#     &= b_{x}^{2} P_{xx} + b_{y}^{2} P_{yy} + b_{z}^{2} P_{zz} + 2\left( b_{x} b_{y} P_{xy} + b_{x} b_{z} P_{xz} + b_{y} b_{z} P_{yz} \right).
# \end{align}
# $$
# 
# Using these, $Q$ becomes
# $$
# \begin{equation}
#     Q = 1 - \frac{4 I_{2}} {\left( I_{1} - P_{\parallel} \right) \left( I_{1} + 3 P_{\parallel} \right)}
# \end{equation}
# $$
# 
# Note that $Q$ is scales as the square of the pressure. It is more helpful if it scales linearly with pressure, so often $\sqrt{Q}$ is used instead.

# ## Reference
# Swisdak, M. (2016). Quantifying gyrotropy in magnetic reconnection. Geophysical Research Letters, 43(1), 43–49. https://doi.org/10.1002/2015GL066980

# ## Preamble

# In[1]:


# !pip install numpy==1.24.3
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt, dates as mdates
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from pymms.data import fgm, fpi


# ## Data
# Define the data interval

# In[ ]:


sc = 'mms1'
t0 = dt.datetime(2017, 7, 11, 22, 33, 30)
t1 = dt.datetime(2017, 7, 11, 22, 34, 30)


# Download the data

# In[ ]:


des_data = fpi.load_moms(sc=sc, mode='brst', optdesc='des-moms', start_date=t0, end_date=t1)
fgm_data = fgm.load_data(sc=sc, mode='brst', start_date=t0, end_date=t1)


# Interpolate FGM to DES

# In[ ]:


B = fgm_data['B_GSE'].interp_like(des_data, method='linear')

# Unit vector
b = B[:,0:3] / np.linalg.norm(B[:,0:3], axis=1)[:, np.newaxis]


# ## Derived Quantities

# ### Trace
# $$
# \begin{equation}
#     I_{1} = P_{xx} + P_{yy} + P_{zz}
# \end{equation}
# $$

# In[ ]:


#I1
I1 = des_data['prestensor'][:,0,0] + des_data['prestensor'][:,1,1] + des_data['prestensor'][:,2,2]
I1 = I1.drop(['cart_index_dim1', 'cart_index_dim2'])


# ### Sum of Principal Minors
# $$
# \begin{equation}
#     I_{2} = P_{xx} P_{yy} + P_{xx} P_{zz} + P_{yy} P_{zz} - \left( P_{xy} P_{yx} + P_{xz} P_{zx} + P_{yz} P_{zy} \right)
# \end{equation}
# $$

# In[ ]:


#I2=PxxPyy+PxxPzz+PyyPzz−(PxyPyx+PxzPzx+PyzPzy)
I2= (des_data['prestensor'][:,0,0] * des_data['prestensor'][:,1,1]
     + des_data['prestensor'][:,0,0] * des_data['prestensor'][:,2,2]
     + des_data['prestensor'][:,1,1] * des_data['prestensor'][:,2,2]
     - (des_data['prestensor'][:,0,1] * des_data['prestensor'][:,1,0]
        + des_data['prestensor'][:,0,2] * des_data['prestensor'][:,2,0]
        + des_data['prestensor'][:,1,2] * des_data['prestensor'][:,2,1]
        )
    )


# ### Parallel Pressure
# $$
# \begin{align}
#     P_{\parallel} &= \hat{\mathbf{b}} \cdot \mathbb{P} \cdot \hat{\mathbf{b}} \\
#     &= b_{x}^{2} P_{xx} + b_{y}^{2} P_{yy} + b_{z}^{2} P_{zz} + 2\left( b_{x} b_{y} P_{xy} + b_{x} b_{z} P_{xz} + b_{y} b_{z} P_{yz} \right).
# \end{align}
# $$

# In[ ]:


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


# ## Agyrotropy
# $$
# \begin{equation}
#     Q = 1 - \frac{4 I_{2}} {\left( I_{1} - P_{\parallel} \right) \left( I_{1} + 3 P_{\parallel} \right)}
# \end{equation}
# $$

# In[ ]:


Q = 1 - ((4*I2) / ((I1-p_par) * (I1 + 3*p_par)))


# ## Plot
# Plot $\sqrt{Q}$

# In[ ]:


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
ax = axes[1,0]
im = ax.pcolorfast(x0, x1, y, cmap='nipy_spectral')
ax.images.append(im)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(formatter)
ax.set_xticklabels([])
ax.set_xlabel('')
ax.set_yscale('log')
ax.set_ylabel('E e-\n(eV)')

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


# In[ ]:




