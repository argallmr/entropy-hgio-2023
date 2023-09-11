#Reciprocal Vectors

from pymms.data import fgm, edp, fpi, util
import datetime as dt
import xarray as xr
import numpy as np
from scipy import constants as c
from matplotlib import pyplot as plt

mu0 = c.physical_constants['vacuum mag. permeability']           # permeability of free space
epsilon0 = c.physical_constants['vacuum electric permittivity']  # permittivity of free space
q = c.physical_constants['atomic unit of charge']                # elementary charge

#data
#t0 = dt.datetime(2017, 7, 11, 22, 33, 30)
#t1 = dt.datetime(2017, 7, 11, 22, 34, 30)

mms1_r_data = util.load_data(sc='mms1', instr='mec', mode='srvy', level='l2', optdesc='epht89d', start_date=t0, end_date=t1)
mms2_r_data = util.load_data(sc='mms2', instr='mec', mode='srvy', level='l2', optdesc='epht89d', start_date=t0, end_date=t1)
mms3_r_data = util.load_data(sc='mms3', instr='mec', mode='srvy', level='l2', optdesc='epht89d', start_date=t0, end_date=t1)
mms4_r_data = util.load_data(sc='mms4', instr='mec', mode='srvy', level='l2', optdesc='epht89d', start_date=t0, end_date=t1)



#FGM is sampled at 4 S/s in survey mode and 128 S/s in burst mode. Select burst mode for its higher time resolution.

mms1_b_data = fgm.load_data(sc='mms1', mode='brst', start_date=t0, end_date=t1)
mms2_b_data = fgm.load_data(sc='mms2', mode='brst', start_date=t0, end_date=t1)
mms3_b_data = fgm.load_data(sc='mms3', mode='brst', start_date=t0, end_date=t1)
mms4_b_data = fgm.load_data(sc='mms4', mode='brst', start_date=t0, end_date=t1)

#Download EDP data to calculate charge density by applying the reciprocal vectors:


#EDP data rate is 32 S/s in survey mode and 4098 S/s in burst mode. Since survey mode sampling rate is closer to the FGM burst mode sample rate, we will load EDP survey data.

mms1_e_data = edp.load_data(sc='mms1', mode='srvy', start_date=t0, end_date=t1)
mms2_e_data = edp.load_data(sc='mms2', mode='srvy', start_date=t0, end_date=t1)
mms3_e_data = edp.load_data(sc='mms3', mode='srvy', start_date=t0, end_date=t1)
mms4_e_data = edp.load_data(sc='mms4', mode='srvy', start_date=t0, end_date=t1)



mms1_des_data = fpi.load_moms(sc='mms1', mode='brst', optdesc='des-moms', start_date=t0, end_date=t1)
mms2_des_data = fpi.load_moms(sc='mms2', mode='brst', optdesc='des-moms', start_date=t0, end_date=t1)
mms3_des_data = fpi.load_moms(sc='mms3', mode='brst', optdesc='des-moms', start_date=t0, end_date=t1)
mms4_des_data = fpi.load_moms(sc='mms4', mode='brst', optdesc='des-moms', start_date=t0, end_date=t1)


#Reciprocal Vectorsd
def recip_vec(R):
    
    r = ['r1', 'r2', 'r3', 'r4']
    i = 0 # alpha
    j = 1 # beta
    k = 2 # gamma
    m = 3 # lambda
    recvec = xr.Dataset()
    
    for i in range(len(R)):
        # r_ij = position vector pointing from S/C i to S/C j
        r_ji = R[r[i]] - R[r[j]]
        r_jk = R[r[k]] - R[r[j]]
        r_jm = R[r[m]] - R[r[j]]
        
        # The reciprocal vector for vertex m of the tetrahedron points normal to the area
        # of the face of the tetrahedron opposite to vertex m. The normal vector to this
        # surface can be found by taking the cross product between two vectors that lie in
        # the plane of the surface.
        area = np.stack([(r_jk[:,1]*r_jm[:,2] - r_jk[:,2]*r_jm[:,1]),
                         (r_jk[:,2]*r_jm[:,0] - r_jk[:,0]*r_jm[:,2]),
                         (r_jk[:,0]*r_jm[:,1] - r_jk[:,1]*r_jm[:,0])],
                        axis=1)
        
        # Calculate the volume of the tetrahedron
        volume = r_ji[:,0]*area[:,0] + r_ji[:,1]*area[:,1] + r_ji[:,2]*area[:,2]
        
        # The reciprical vector is the area's normal vector normalized to the tetrahedron
        # volume
        k_i = xr.DataArray(np.stack([area[:,0] / volume,
                                     area[:,1] / volume,
                                     area[:,2] / volume], axis=1),
                           dims=('time', 'component'),
                           coords={'time': R['time'],
                                   'component': R['component']}
                           )
        
        recvec['k{0:1d}'.format(i+1)] = k_i
        
        # k[0,0,i] = [ area[0,:] / volume,
        #              area[1,:] / volume,
        #              area[2,:] / volume ]
        
        # These should exist but my version of XArray does not have them
        # area = xr.cross(r_jk, r_jm, dim='component')
        # volume = xr.dot(r_ji, area, dims='component')
        # recvec = area / volume
        
        # increase the indices cyclically
        j = (j + 1) % 4
        k = (k + 1) % 4
        m = (m + 1) % 4
    
    return recvec
#curl
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


def divergence(K, V, dim='component'):
    div = 0
    
    for k_name, v_name in zip(K, V):
        k = K[k_name]
        v = V[v_name]
        #div += k.dot(v, dim=dim) #gave me the error: TypeError: 
        #DataArray.dot() got an unexpected keyword argument 'dim'
        div += (k * v).sum(dim=dim)
    return div



def gradient(K, G):
    '''
    Note that XArray will automatically broadcasts arrays based on their dimension names.
    Therefore, when taking the spatial gradient (Nx3) of a scalar (N), the scalar will be broadcast
    to Nx3 and multiplied. When taking the spatial gradient (Nx3) of a vector (Nx3), the second dimension
    must have different names so that XArray broadcasts them to (Nx3x3) and calculates the outer product.

    See:
    https://stackoverflow.com/questions/51144786/computing-the-outer-product-of-two-variables-in-xarray-dataset
    '''
    grad = 0
    
    for k_name, g_name in zip(K, G):
        k = K[k_name]
        g = G[g_name]
        grad += k*g
    
    return grad



## Application

### Interpolate
#Make sure all of the data has the same time stamps. Make MMS1 the reference spacecraft. Use the EDP time stamps. 
#FGM is twice as fast (linearly interpolate to downsample) and FPI is nearly the same (nearest neighbor interpolate).
#Because the spacecraft position changes slowly, it is safe to upsample (linearly interpolate) the MEC data to the EDP time cadence.

e_mms1 = mms1_e_data['E_GSE']
e_mms2 = mms2_e_data['E_GSE'].interp_like(e_mms1, method='nearest')
e_mms3 = mms3_e_data['E_GSE'].interp_like(e_mms1, method='nearest')
e_mms4 = mms4_e_data['E_GSE'].interp_like(e_mms1, method='nearest')

b_mms1 = mms1_b_data['B_GSE'].interp_like(e_mms1, method='linear')
b_mms2 = mms2_b_data['B_GSE'].interp_like(b_mms1, method='linear')
b_mms3 = mms3_b_data['B_GSE'].interp_like(b_mms1, method='linear')
b_mms4 = mms4_b_data['B_GSE'].interp_like(b_mms1, method='linear')

# Interpolate the Dataframe because we will need bulk velocity and pressure
des_mms1 = mms1_des_data.interp_like(e_mms1, method='nearest')
des_mms2 = mms2_des_data.interp_like(e_mms1, method='nearest')
des_mms3 = mms3_des_data.interp_like(e_mms1, method='nearest')
des_mms4 = mms4_des_data.interp_like(e_mms1, method='nearest')

r_mms1 = mms1_r_data['mms1_mec_r_gse'].rename({'Epoch': 'time'}).interp_like(e_mms1, method='nearest')
r_mms2 = mms2_r_data['mms2_mec_r_gse'].rename({'Epoch': 'time'}).interp_like(e_mms1, method='nearest')
r_mms3 = mms3_r_data['mms3_mec_r_gse'].rename({'Epoch': 'time'}).interp_like(e_mms1, method='nearest')
r_mms4 = mms4_r_data['mms4_mec_r_gse'].rename({'Epoch': 'time'}).interp_like(e_mms1, method='nearest')




#dataset
# Put the position vectors into their own Dataset
R = xr.Dataset({'r1': r_mms1.rename({'mms1_mec_r_gse_label': 'component'}).assign_coords({'component': ['x', 'y', 'z']}),
                'r2': r_mms2.rename({'mms2_mec_r_gse_label': 'component'}).assign_coords({'component': ['x', 'y', 'z']}),
                'r3': r_mms3.rename({'mms3_mec_r_gse_label': 'component'}).assign_coords({'component': ['x', 'y', 'z']}),
                'r4': r_mms4.rename({'mms4_mec_r_gse_label': 'component'}).assign_coords({'component': ['x', 'y', 'z']})})

B = xr.Dataset({'B1': b_mms1,
                'B2': b_mms2,
                'B3': b_mms3,
                'B4': b_mms4})

E = xr.Dataset({'E1': e_mms1.rename({'mms1_edp_label1_fast_l2': 'component'}).assign_coords({'component': ['x', 'y', 'z']}),
                'E2': e_mms2.rename({'mms2_edp_label1_fast_l2': 'component'}).assign_coords({'component': ['x', 'y', 'z']}),
                'E3': e_mms3.rename({'mms3_edp_label1_fast_l2': 'component'}).assign_coords({'component': ['x', 'y', 'z']}),
                'E4': e_mms4.rename({'mms4_edp_label1_fast_l2': 'component'}).assign_coords({'component': ['x', 'y', 'z']})})

U = xr.Dataset({'U1': des_mms1['velocity'].rename({'velocity_index': 'component'}).assign_coords({'component': ['x', 'y', 'z']}),
                'U2': des_mms2['velocity'].rename({'velocity_index': 'component'}).assign_coords({'component': ['x', 'y', 'z']}),
                'U3': des_mms3['velocity'].rename({'velocity_index': 'component'}).assign_coords({'component': ['x', 'y', 'z']}),
                'U4': des_mms4['velocity'].rename({'velocity_index': 'component'}).assign_coords({'component': ['x', 'y', 'z']})
               })

p = xr.Dataset({'p1': des_mms1['p'],
                'p2': des_mms2['p'],
                'p3': des_mms3['p'],
                'p4': des_mms4['p']
               })

P = xr.Dataset({'P1': des_mms1['prestensor'].rename({'cart_index_dim1': 'comp_1', 'cart_index_dim2': 'comp_2'}),
                'P2': des_mms2['prestensor'].rename({'cart_index_dim1': 'comp_1', 'cart_index_dim2': 'comp_2'}),
                'P3': des_mms3['prestensor'].rename({'cart_index_dim1': 'comp_1', 'cart_index_dim2': 'comp_2'}),
                'P4': des_mms4['prestensor'].rename({'cart_index_dim1': 'comp_1', 'cart_index_dim2': 'comp_2'}),
               })

#Current Density

# Create the reciprocal vectors
k = recip_vec(R)

# Calculate the curl
curlB = curl(k, B)

# Current density
J = 1e6 * mu0[0] * curlB


#Charge Density
# Calculate the divergence
divE = divergence(k, E)

# Charge density
n_rho = 1e-12 * epsilon0[0] * divE / q[0]

# Theta: Divergence of electron velocity
divU = divergence(k, U)

# Take the barycentric average of the scalar pressure
p_bary = (p['p1'] + p['p2'] + p['p3'] + p['p4']) / 4

# p-Theta: Pressure dilatation
p_theta = p_bary * divU

# Gradient of the bulk velocity
gradU = gradient(R.rename(component='comp1'), U.rename(component='comp2'))

# Theta - divergence of the bulk velocity
divU = divergence(R, U)

# Barycentric average of the pressure tensor and scalar pressure
P_bary = (P['P1'] + P['P2'] + P['P3'] + P['P4']) / 4
p_bary = (p['p1'] + p['p2'] + p['p3'] + p['p4']) / 4

# Pi - Traceless pressure tensor
Pi = P_bary - p_bary * xr.DataArray(np.broadcast_to(np.identity(3), (len(p_bary), 3, 3)),
                                    dims=('time', 'comp1', 'comp2'))

# D - the devoriak pressure
D = (0.5 * (gradU + gradU.transpose('time', 'comp2', 'comp1').rename(comp2='comp1', comp1='comp2'))
     - 1/3 * divU * xr.DataArray(np.broadcast_to(np.identity(3), (len(p_bary), 3, 3)),
                                 dims=('time', 'comp1', 'comp2'))
     )

# PiD
PiD = Pi * D


#Plot
#Compare the plots of current and charge density to the figure in the second column and second row
#Matthew Argall, Jason Shuster, Ivan Dors, et al. How neutral is quasi-neutral: Charge Density in the Reconnection Diffusion Region Observed by MMS. Authorea. December 16, 2019. DOI: 10.1002/essoar.10501410.1

fig, axes = plt.subplots(nrows=4, ncols=1, squeeze=False)
​
# Current Density
ax = axes[0,0]
J.loc[:,'x'].plot(ax=ax, label='x')
J.loc[:,'y'].plot(ax=ax, label='y')
J.loc[:,'z'].plot(ax=ax, label='z')
ax.set_title('Application of Reciprocal Vectors')
ax.set_xlabel('')
ax.set_xticklabels([''])
ax.set_ylabel('J [$\\mu A/m^{2}$]')
ax.legend()
​
# Charge Density
ax = axes[1,0]
n_rho.plot(ax=ax)
ax.set_xlabel('')
ax.set_xticklabels([''])
ax.set_ylabel('$n_{\\rho}$ [$cm^{-3}$]')
​
# Pressure Dilatation
ax = axes[2,0]
p_theta.plot(ax=ax)
ax.set_title('')
ax.set_xlabel('Time (UTC)')
ax.set_ylabel('$p \\theta$ [nW/$m^{3}$]')
​
# Collisionless Viscosity
ax = axes[3,0]
PiD.plot(ax=ax)
ax.set_title('')
ax.set_xlabel('Time (UTC)')
_ = ax.set_ylabel('$p \\theta$ [nW/$m^{3}$]')
