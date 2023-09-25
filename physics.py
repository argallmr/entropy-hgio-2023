import numpy as np
import xarray as xr
from scipy import constants as c
from tools import recip_vec, curl, divergence, gradient, barycentric_avg

mu0 = c.physical_constants['vacuum mag. permeability']           # permeability of free space
epsilon0 = c.physical_constants['vacuum electric permittivity']  # permittivity of free space
q = c.physical_constants['atomic unit of charge']                # elementary charge


def convection_efield(v, B):
    '''
    Calculate the convection electric field Ec = v x B

    Parameters
    ----------
    v : array-like (N, 3)
        Bulk velocity in units of [km/s]
    B : array-like (N, 3)
        Vector magnetic field in units of [nT]
    
    Returns
    -------
    Ec : array-like (N, 3)
        Convective electric field in units of [mV/m]
    '''
    # Should be able to use xr.cross() but I must not have the right version
    Ec = 1e-3 * xr.DataArray(np.stack([(v[:,1]*B[:,2] - v[:,2]*B[:,1]),
                                       (v[:,2]*B[:,0] - v[:,0]*B[:,2]),
                                       (v[:,0]*B[:,1] - v[:,1]*B[:,0])], axis=1),
                            dims=('time', 'component'),
                            coords={'time': v['time'],
                                    'component': ['x', 'y', 'z']}
                            )

    return Ec


def current_density_curl(R, B):
    '''
    Calculate the current density via Ampere's Law: the curl of the magnetic field.

    Parameters
    ----------
    R : `xarray.Dataset` of (N, 3) arrays
        Spacecraft position vectors in units of [km]
    B : array-like (N, 3)
        Vector magnetic field in units of [nT]
    
    Returns
    -------
    J : array-like (N, 3)
        Current density in units of [µA/m^2]
    '''

    # Create the reciprocal vectors
    k = recip_vec(R)

    # Calculate the curl
    curlB = curl(k, B)

    # Current density
    J = 1e6 * mu0[0] * curlB

    return J


def current_density_moms(n, vi, ve):
    '''
    Calculate the current density from plasma moments.

    Parameters
    ----------
    n : array-like (N, 3)
        Number density in units of [km]
    vi : array-like (N, 3)
        Ion bulk velocity in units of [km/s]
    ve : array-like (N, 3)
        Electron bulk velocity in units of [km/s]
    
    Returns
    -------
    J : array-like (N, 3)
        Current density in units of [µA/m^2]
    '''
    # Current density
    J = 1e15 * q[0] * n * (vi - ve)

    return J


def charge_density(R, E):
    '''
    Calculate the density of free charges.

    Parameters
    ----------
    R : `xarray.Dataset` of (N, 3) arrays
        Spacecraft position vectors in units of [km]
    E : `xarray.Dataset` of (N, 3) arrays
        Electric field measured by each spacecraft in units of [mV/m]
    
    Returns
    -------
    n_rho : array-like (N, 3)
        Density of free charges in units of [cm^{-3}]
    '''

    # Create the reciprocal vectors
    k = recip_vec(R)

    # Calculate the divergence
    divE = divergence(k, E)

    # Charge density
    n_rho = 1e-12 * epsilon0[0] * divE / q[0]

    return n_rho


def De_moms(E, B, n, Vi, Ve):
    '''
    Calculate the electron frame dissipation measure using data from a single
    spacecraft. This uses the plasma moments to calculate the current density.

    Parameters
    ----------
    E : array-like (N, 3)
        Electric field measured in units of [mV/m]
    B : array-like (N, 3)
        Vector magnetic field in units of [nT]
    Ve : array-like (N, 3)
        Electron bulk velocity in units of [km/s]
    n : array-like (N, 3)
        Number density in units of [km]
    Vi : array-like (N, 3)
        Ion bulk velocity in units of [km/s]
    R : array-like (N, 3)
        Spacecraft position vectors in units of [km]
    
    Returns
    -------
    De : array-like (N,)
        Electron frame dissipation measure in units of [nW/m^{3}]
    '''
    
    # Electric field in the electron rest frame
    Ec = convection_efield(Ve, B)
    E_prime = E + Ec

    # Current density
    J = current_density_moms(n, Vi, Ve)
    
    # Electron frame dissipation measure
    De = J.dot(E_prime, dims='component')

    return De


def De_curl(E, B, Ve, R):
    '''
    Calculate the electron frame dissipation measure using data from a single
    spacecraft. This uses the Ampere's Law to calculate the current density.

    Parameters
    ----------
    E : `xarray.Dataset` of (N, 3) arrays
        Electric field measured in units of [mV/m] from each spacecraft
    B : `xarray.Dataset` of (N, 3) arrays
        Vector magnetic field in units of [nT] from each spacecraft
    Ve : `xarray.Dataset` of (N, 3) arrays
        Bulk velocity in units of [km/s] from each spacecraft
    R : `xarray.Dataset` of (N, 3) arrays
        Position vector in usnits of [km] from each spacecraft
    
    Returns
    -------
    De : array-like (N,)
        Electron frame dissipation measure in units of [nW/m^{3}]
    '''
    
    # Convective electric field
    Ec = xr.Dataset()
    for idx, (vname, Bname) in enumerate(zip(Ve, B)):
        Ec['Ec{0}'.format(idx)] = convection_efield(Ve[vname], B[Bname])
    
    # Electric field in the electron rest frame
    E_prime = barycentric_avg(E) + barycentric_avg(Ec)

    # Current density
    J = current_density_curl(R, B)
    
    # Electron frame dissipation measure
    De = J.dot(E_prime, dims='component')

    return De


def pressure_dilatation(R, U, p):
    '''
    Calculate the density of free charges.

    Parameters
    ----------
    R : `xarray.Dataset` of (N, 3) arrays
        Spacecraft position vectors in units of [km]
    U : `xarray.Dataset` of (N, 3) arrays
        Bulk velocity measured by each spacecraft in units of [km/s]
    p : `xarray.Dataset` of (N, 3) arrays
        Scalar pressure measured by each spacecraft in units of [nPa]
    
    Returns
    -------
    p_theta : array-like (N, 3)
        Pressure dilatation in units of [nW/m^{3}]
    '''

    # Create the reciprocal vectors
    k = recip_vec(R)

    # Theta: Divergence of electron velocity
    divU = divergence(k, U)

    # Take the barycentric average of the scalar pressure
    p_bary = (p['p1'] + p['p2'] + p['p3'] + p['p4']) / 4

    # p-Theta: Pressure dilatation
    p_theta = p_bary * divU

    return p_theta


def traceless_pressure_tensor(p, P):
    '''
    Calculate the barycentric average of the traceless pressure tensor.

    Parameters
    ----------
    p : `xarray.Dataset` of (N, 3) arrays
        Scalar pressure measured by each spacecraft in units of [nPa]
    P : `xarray.Dataset` of (N, 3) arrays
        Pressure tensor measured by each spacecraft in units of [nPa]
    
    Returns
    -------
    p_theta : array-like (N, 3)
        Pressure dilatation in units of [nW/m^{3}]
    '''

    # Barycentric average of the pressure tensor and scalar pressure
    P_bary = (P['P1'] + P['P2'] + P['P3'] + P['P4']) / 4
    p_bary = (p['p1'] + p['p2'] + p['p3'] + p['p4']) / 4

    # Pi - Traceless pressure tensor
    Pi = P_bary - p_bary * xr.DataArray(np.broadcast_to(np.identity(3), (len(p_bary), 3, 3)),
                                        dims=('time', 'comp1', 'comp2'))

    return Pi


def devoriak_pressure(R, U, p, P):
    '''
    Calculate the devoriak pressure.

    Parameters
    ----------
    R : `xarray.Dataset` of (N, 3) arrays
        Spacecraft position vectors in units of [km]
    U : `xarray.Dataset` of (N, 3) arrays
        Bulk velocity measured by each spacecraft in units of [km/s]
    p : `xarray.Dataset` of (N, 3) arrays
        Scalar pressure measured by each spacecraft in units of [nPa]
    P : `xarray.Dataset` of (N, 3) arrays
        Pressure tensor measured by each spacecraft in units of [nPa]
    
    Returns
    -------
    D : array-like
        Devoriak pressure in units of [nW/m^{3}]
    '''

    # Gradient of the bulk velocity
    gradU = gradient(R.rename(component='comp1'), U.rename(component='comp2'))

    # Theta - divergence of the bulk velocity
    divU = divergence(R, U)

    # Barycentric scalar pressure
    p_bary = (p['p1'] + p['p2'] + p['p3'] + p['p4']) / 4

    # D - the devoriak pressure
    D = (0.5 * (gradU + gradU.transpose('time', 'comp2', 'comp1').rename(comp2='comp1', comp1='comp2'))
        - 1/3 * divU * xr.DataArray(np.broadcast_to(np.identity(3), (len(p_bary), 3, 3)),
                                    dims=('time', 'comp1', 'comp2'))
        )

    return D


def PiD(R, U, p, P):
    '''
    Calculate Pi-D.

    Parameters
    ----------
    R : `xarray.Dataset` of (N, 3) arrays
        Spacecraft position vectors in units of [km]
    U : `xarray.Dataset` of (N, 3) arrays
        Bulk velocity measured by each spacecraft in units of [km/s]
    p : `xarray.Dataset` of length-N arrays
        Scalar pressure measured by each spacecraft in units of [nPa]
    P : `xarray.Dataset` of (N, 3) arrays
        Pressure tensor measured by each spacecraft in units of [nPa]
    
    Returns
    -------
    PiD : array-like
        Devoriak pressure in units of [nW/m^{3}]
    '''

    Pi = traceless_pressure_tensor(p, P)

    D = devoriak_pressure(R, U, p, P)
    
    # PiD
    PiD = Pi.dot(D, dims=('comp1', 'comp2'))

    return PiD


def agyrotropy(P, b):
    '''
    Calculate agyrotropy.

    Parameters
    ----------
    b : array-like (N,)
        Normalized vector magnetic field
    P : array-like (N,3,3)
        Pressure tensor measured by each spacecraft in units of [nPa]
    
    Returns
    -------
    Q : array-like (N,)
        Agyrotropy measure of the pressure tensor
    '''

    # I1
    I1 = P[:,0,0] + P[:,1,1] + P[:,2,2]
    I1 = I1.drop(['cart_index_dim1', 'cart_index_dim2'])


    # I2 = PxxPyy + PxxPzz + PyyPzz − (PxyPyx + PxzPzx + PyzPzy)
    I2= (P[:,0,0] * P[:,1,1]
        + P[:,0,0] * P[:,2,2]
        + P[:,1,1] * P[:,2,2]
        - (P[:,0,1] * P[:,1,0]
            + P[:,0,2] * P[:,2,0]
            + P[:,1,2] * P[:,2,1]
            )
        )

    # P.b
    p_par = P.dot(b.rename({'b_index': 'cart_index_dim2'}), dims=('cart_index_dim2',))

    # b.(P.b)
    p_par = b.dot(p_par.rename({'cart_index_dim1': 'b_index'}), dims=('b_index',))

    # p_par = (b[:,0]**2 * P[:,0,0]
    #          + b[:,1]**2 * P[:,1,1]
    #          + b[:,2]**2 * P[:,2,2]
    #          + 2 * (b[:,0]*B[:,1]*P[:,0,1]
    #                 + b[:,0]*B[:,2]*P[:,0,2]
    #                 + b[:,1]*B[:,2]*P[:,1,2]
    #                 )
    #          )

    ## Agyrotropy
    Q = 1 - ((4*I2) / ((I1-p_par) * (I1 + 3*p_par)))

    return Q