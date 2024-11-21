import numpy as np
import datetime as dt
import xarray as xr
from pymms.data import fpi
from scipy import constants
from tqdm import tqdm

import matplotlib as mpl
from matplotlib import pyplot as plt, dates as mdates, cm, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import plots

eV2K = constants.value('electron volt-kelvin relationship')
K2eV = constants.value('kelvin-electron volt relationship')
eV2J = constants.value('electron volt-joule relationship')
J2eV = constants.value('joule-electron volt relationship')
e = constants.value('elementary charge')
kB   = constants.Boltzmann

class Lookup_Table():

    def __init__(self, deltan_n=0.005, deltat_t=0.005,
                 deltas_s=0.005, deltasV_sV=0.005, species='e'):
        '''
        Create a look-up table of maxwellian distributions

        Parameters
        ----------
        f : `pymms.data.fpi.Distribution_Function`
            Find the equivalent Maxwellian distriboution
        n : float
            Density of `f`. If not provided, it will be calculated.
        t : float
            Scalar temperature of `f`. If not provided, it will be calculated.
        species : str
            Species of the distribution function: ('e', 'i')
        '''
        
        self.deltan_n = deltan_n
        self.deltat_t = deltat_t
        self.deltas_s = deltas_s
        self.deltasV_sV = deltasV_sV
        self.species = species
        self.mass = self.species_to_mass(species)
    
    def density(self):
        
        # Integrate the azimuthal angle
        n = self.lut.integrate('phi')

        # Integrate over polar angle
        n = (np.sin(n['theta']) * n).integrate('theta')

        # Integrate over energy
        #   - U ranges from [0, inf] and np.inf/np.inf = nan
        #   - Set the last element of the energy dimension of y to 0
        with np.errstate(invalid='ignore', divide='ignore'):
            y = np.sqrt(self.lut['U']) / (1-self.lut['U'])**(5/2)    
        # the DataArray version of where, other is unimplemented
        y = np.where(np.isfinite(y), y, 0)

        coeff = 1e6 * np.sqrt(2) * (eV2J * self.lut.attrs['Energy_e0'] / self.mass)**(3/2)
        n = coeff * (y * n).integrate('U')

        return n.data

    def velocity(self, n=None):

        if n is None:
            n = self.density()
        
        # Integrate over azimuth angle
        vx = (np.cos(self.lut['phi']) * self.lut).integrate('phi')
        vy = (np.sin(self.lut['phi']) * self.lut).integrate('phi')
        vz = self.lut.integrate('phi')

        # Integrate over polar angle
        vx = (np.sin(self.lut['theta'])**2 * vx).integrate('theta')
        vy = (np.sin(self.lut['theta'])**2 * vy).integrate('theta')
        vz = (np.cos(self.lut['theta']) * np.sin(self.lut['theta']) * vz).integrate('theta')

        # Integrate over energy
        #   - U ranges from [0, 1] and 0/0 = nan
        #   - Set the last element of the energy dimension of y to 0
        with np.errstate(divide='ignore', invalid='ignore'):
            y = self.lut['U'] / (1 - self.lut['U'])**3
        y = np.where(np.isfinite(y), y, 0)
    
        coeff = -1e3 * 2 * (eV2J * self.lut.attrs['Energy_e0'] / self.mass)**2 / n
        vx = coeff * (y * vx).integrate('U')
        vy = coeff * (y * vy).integrate('U')
        vz = coeff * (y * vz).integrate('U')

        return np.stack([vx, vy, vz], axis=2)

    def temperature(self, n=None, V=None):
        
        if n is None:
            n = self.density()
        if V is None:
            V = self.velocity()
        
        # Integrate over azimuth angle
        Txx = (np.cos(self.lut['phi'])**2 * self.lut).integrate('phi')
        Tyy = (np.sin(self.lut['phi'])**2 * self.lut).integrate('phi')
        Tzz = self.lut.integrate('phi')
        Txy = (np.cos(self.lut['phi']) * np.sin(self.lut['phi']) * self.lut).integrate('phi')
        Txz = (np.cos(self.lut['phi']) * self.lut).integrate('phi')
        Tyz = (np.sin(self.lut['phi']) * self.lut).integrate('phi')

        # Integreate over polar angle
        #   - trapz returns a ndarray so use np.trapz() instead of DataArray.inegrate()
        #   - Dimensions should now be: before: [time, theta, energy], after: [time, energy]
        Txx = (np.sin(self.lut['theta'])**3 * Txx).integrate('theta')
        Tyy = (np.sin(self.lut['theta'])**3 * Tyy).integrate('theta')
        Tzz = (np.cos(self.lut['theta'])**2 * np.sin(self.lut['theta']) * Tzz).integrate('theta')
        Txy = (np.sin(self.lut['theta'])**3 * Txy).integrate('theta')
        Txz = (np.cos(self.lut['theta']) * np.sin(self.lut['theta'])**2 * Txz).integrate('theta')
        Tyz = (np.cos(self.lut['theta']) * np.sin(self.lut['theta'])**2 * Tyz).integrate('theta')

        # Create a temperature tensor
        T = np.stack([np.stack([Txx, Txy, Txz], axis=3),
                      np.stack([Txy, Tyy, Tyz], axis=3),
                      np.stack([Txz, Tyz, Tzz], axis=3)
                      ], axis=4
                     )
    
        # Create a velocity tensor
        Vij = np.stack((np.stack([V[:,:,0]*V[:,:,0], V[:,:,0]*V[:,:,1], V[:,:,0]*V[:,:,2]], axis=2),
                        np.stack([V[:,:,1]*V[:,:,0], V[:,:,1]*V[:,:,1], V[:,:,1]*V[:,:,2]], axis=2),
                        np.stack([V[:,:,2]*V[:,:,0], V[:,:,2]*V[:,:,1], V[:,:,2]*V[:,:,2]], axis=2)),
                       axis=3)
        
        # Integrate over energy
        with np.errstate(divide='ignore', invalid='ignore'):
            y = self.lut['U']**(3/2) / (1 - self.lut['U'])**(7/2)
        y = np.where(np.isfinite(y), y, 0)

        coeff = 1e6 * (2/self.mass)**(3/2) / (n * kB / K2eV) * (self.lut.attrs['Energy_e0']*eV2J)**(5/2)
        T = (coeff[..., np.newaxis, np.newaxis]
             * np.trapz(y[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                        * T, self.lut['U'], axis=2)
             - (1e6 * self.mass / kB * K2eV * Vij)
             )

        return T

    def scalar_temperature(self, n=None, V=None, T=None):
        
        if T is None:
            T = self.temperature(n=n, V=V)
        return (T[:,:,0,0] + T[:,:,1,1] + T[:,:,2,2]) / 3.0
    
    def pressure(self, n=None, T=None, **kwargs):

        if n is None:
            n = self.density()
        if T is None:
            T = self.temperature(n=n, **kwargs)

        P = 1e15 * n[...,np.newaxis,np.newaxis] * kB * eV2K * T
    
        return P

    def scalar_pressure(self, **kwargs):

        P = self.pressure(**kwargs)
        return (P[:,:,0,0] + P[:,:,1,1] + P[:,:,2,2]) / 3.0


    def maxwellian_entropy(self, n=None, p=None, **kwargs):
        if n is None:
            n = self.density()
        if p is None:
            p = self.scalar_pressure(n=n, **kwargs)

        sM = (-kB * 1e6 * n
              * (np.log((1e19 * self.mass * n**(5.0/3.0)
                        / 2 / np.pi / p)**(3/2)
                       )
                 - 3/2
                 )
              )
        
        return sM
    
    def entropy(self):
        
        # Integrate over phi and theta
        #   - Measurement bins with zero counts result in a
        #     phase space density of 0
        #   - Photo-electron correction can result in negative
        #     phase space density.
        #   - Log of value <= 0 is nan. Avoid by replacing
        #     with 1 so that log(1) = 0
        S = 1e12 * self.lut
        S = S.where(S > 0, 1)
        S = (S * np.log(S)).integrate('phi')
        S = (np.sin(S['theta']) * S).integrate('theta')
    
        # Integrate over Energy
        with np.errstate(invalid='ignore', divide='ignore'):
            y = np.sqrt(S['U']) / (1 - S['U'])**(5/2)
        y = np.where(np.isfinite(y), y, 0)
        
        E0 = self.lut.attrs['Energy_e0'] * eV2J
        coeff = -kB * np.sqrt(2) * (E0 / self.mass)**(3/2)
        S = coeff * (y * S).integrate('U')
    
        return S
    
    def vspace_entropy(self, n=None, s=None):

        if n is None:
            n = self.density()
        if s is None:
            s = self.entropy()
        
        # Assume that the azimuth and polar angle bins are equal size
        dtheta = self.lut['theta'].diff(dim='theta').mean()
        dphi = self.lut['phi'].diff(dim='phi').mean()
        
        # Calculate the factors that associated with the normalized
        # volume element
        #   - U ranges from [0, inf] and np.inf/np.inf = nan
        #   - Set the last element of y along U manually to 0
        #   - log(0) = -inf; Zeros come from theta and y. Reset to zero
        #   - Photo-electron correction can result in negative phase space
        #     density. log(-1) = nan
        E0 = self.lut.attrs['Energy_e0']*eV2J
        coeff = np.sqrt(2) * (E0/self.mass)**(3/2) # m^3/s^3
        with np.errstate(invalid='ignore', divide='ignore'):
            y = np.sqrt(self.lut['U']) / (1 - self.lut['U'])**(5/2)
            lnydy = (np.log(y
                     * np.sin(self.lut['theta'])
                     * dtheta * dphi))
        y = y.where(np.isfinite(y), 0)
        lnydy = lnydy.where(np.isfinite(lnydy), 0)
    
        # Terms in that make up the velocity space entropy density
        sv1 = s # J/K/m^3 ln(s^3/m^6) -- Already multiplied by -kB
        sv2 = kB * (1e6*n) * np.log(1e6*n/coeff) # 1/m^3 * ln(1/m^3)
        
        sv3 = (y * lnydy * np.sin(self.lut['theta']) * self.lut).integrate('phi')
        sv3 = sv3.integrate('theta')
        sv3 = -kB * 1e12 * coeff * sv3.integrate('U') # 1/m^3
    
        sv4 = (y * np.sin(self.lut['theta']) * self.lut).integrate('phi')
        sv4 = sv4.integrate('theta')
        sv4 = -kB * 1e12 * coeff * sv4.integrate('U')
    
        # Velocity space entropy density
        sv = sv1 + sv2 + sv3 + sv4 # J/K/m^3
    
        return sv
    
    @staticmethod
    def deltaE(energy):
        '''
        Compute the size of each energy bin
        
        dE/E = const -> d(lnE) = const -> d[log(E) / log(exp)]

        So, dE = E * dlogE / log(exp)

        Return
        ------
        dE : `numpy.ndarray`
            Size of each energy bin
        '''
        
        dlogE = np.log10(energy[1]) - np.log10(energy[0])
        dE = energy * dlogE / np.log10(np.exp(1))

        return dE

    # Maybe static method?
    def equivalent_maxwellian(self, f, n=None, V=np.zeros((3,)), t=None):
        
        if n is None:
            n = f.density()
        if t is None:
            t = f.scalar_temperature(n=n, V=V)

        f_M = f.maxwellian(n=n, V=V, t=t)

        return f_M

    def load(filename):
        return xr.load_dataset(filename)
    
    def precondition(self):
        if self.is_preconditioned():
            return
        
        f = self.f.copy()
        phi = self.phi.copy()
        theta = self.theta.copy()
        energy = self.energy.copy()
        
        # Make the distribution periodic in phi
        if self.wrap_phi:
            phi = np.deg2rad(np.append(phi, phi[0] + 360))
            f = np.append(f, f[np.newaxis, 0, :, :], axis=0)
        
        # Add endpoints at 0 and 180 degrees (sin(0,180) = 0)
        if self.theta_extrapolation:
            theta = np.deg2rad(np.append(np.append(0, theta), 180))
            f = np.append(np.zeros((f.shape[0], 1, f.shape[2])), f, axis=1)
            f = np.append(f, np.zeros((f.shape[0], 1, f.shape[2])), axis=1)
        
        # Spacecraft potential correction
        if self.scpot is not None:
            sign = -1
            energy = energy + (sign * J2eV * e * self.scpot)
            
            mask = energy >= 0
            energy = energy[mask]
            f = f[:, :, mask]
        
        # Lower integration limit
        if self.E_low is not None:
            mask = energy >= self.E_low
            energy = energy[mask]
            f = f[:, :, mask]
        
        # Upper integration limit
        if self.E_high is not None:
            mask = energy <= self.E_high
            energy = energy[mask]
            f = f[:, :, mask]
        
        # Normalize energy
        U = energy / (energy + self.E0)
        
        # Low energy extrapolation
        if self.low_energy_extrapolation:
            energy = np.append(0, energy)
            U = np.append(0, U)
            f = np.append(np.zeros((*f.shape[0:2], 1)), f, axis=2)
        
        # High energy extrapolation
        if self.high_energy_extrapolation:
            energy = np.append(energy, np.inf)
            U = np.append(U, 1)
            f = np.append(f, np.zeros((*f.shape[0:2], 1)), axis=2)
        
        # Preconditioned parameters
        self._phi = phi
        self._theta = theta
        self._energy = energy
        self._U = U
        self._f = f
        self._is_preconditioned = True

    def fill_grid(self, f, **kwargs):

        # Fill the grid with Maxwellian distributions
        self.lut = self.maxwellian(f=f)

        # Calculate the density and temperature of the grid
        self.n_lut = self.density()
        self.t_lut = self.scalar_temperature(n=self.n_lut)
        self.s_M_lut = self.maxwellian_entropy(n=self.n_lut)
        self.sV_lut = self.vspace_entropy(n=self.n_lut)


    def set_grid(self, n, t, s_M, sV):
        '''
        Create the density-temperature grid.

        Parameters
        ----------
        n : float
            Density around which to create the grid
        t : float
            Scalar temperature around which to create the grid
        '''

        # This does not have to be much bigger than deltan_n or deltat_t
        n_range = np.array([0.9, 1.1]) * n
        t_range = np.array([0.9, 1.1]) * t
        s_M_range = np.array([0.9, 1.1]) * s_M
        sV_range = np.array([0.9, 1.1]) * sV

        n_data, t_data, s_M_data, sV_data = self.grid_coords(n_range, t_range,
                                                             s_M_range, sV_range)
        n_data, t_data = np.meshgrid(n_data, t_data, indexing='ij')
        self.n_data = n_data
        self.t_data = t_data
        self.s_M_data = s_M_data
        self.sV_data = sV_data

    def grid_coords(self, n_range, t_range, s_M_range, sV_range):
        
        # Determine the number of cells
        N = self.grid_resolution(n_range, self.deltan_n)
        M = self.grid_resolution(t_range, self.deltat_t)
        L = self.grid_resolution(s_M_range, self.deltas_s)
        P = self.grid_resolution(sV_range, self.deltasV_sV)

        # Create the grid
        n = np.logspace(np.log10(n_range[0]), np.log10(n_range[1]), N)
        t = np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), M)
        s_M = np.logspace(np.log10(s_M_range[0]), np.log10(s_M_range[1]), L)
        sV = np.logspace(np.log10(sV_range[0]), np.log10(sV_range[1]), P)

        # Set the grid
        return n, t, s_M, sV
    
    @staticmethod
    def grid_resolution(lim, err):
        '''
        Calculate the number of logarithmically-spaced points between two limits,
        given than the relative spacing between points is constant.

        Parameters
        ----------
        lim : (2,), float
            Minimum and maximum of the data range
        err : float
            Relative spacing between points (∆x/x)
        
        Returns
        -------
        N : int
            Number of points that span data range with constant `err`
        '''
        N = np.ceil((np.log10(lim[1]) - np.log10(lim[0]))
                    / np.log10(err + 1)
                    )
        return int(N)

    def grid_err(lim, N):
        '''
        Calculate the number of logarithmically-spaced points between two limits,
        given than the relative spacing between points is constant.

        Parameters
        ----------
        lim : (2,), float
            Minimum and maximum of the data range
        err : float
            Relative spacing between points (∆x/x)
        
        Returns
        -------
        N : int
            Number of points that span data range with constant `err`
        '''
        delta = 10**((np.log10(lim[1]) - np.log10(lim[0])) / N) - 1
        return delta
    
    def maxwellian(self, f=None,
                   phi=None, theta=None, energy=None,
                   phi_range=(0, 360), theta_range=(0, 180), energy_range=(10, 30000),
                   nphi=32, ntheta=16, nenergy=32):
        """
        Given a measured velocity distribution function, create a Maxwellian
        distribution function with the same density, bulk velociy, and
        temperature.
        
        Parameters
        ----------
        dist : `xarray.DataSet`
            A time series of 3D velocity distribution functions
        N : `xarray.DataArray`
            Number density computed from `dist`.
        bulkv : `xarray.DataArray`
            Bulk velocity computed from `dist`.
        T : `xarray.DataArray`
            Scalar temperature computed from `dist`
        
        Returns
        -------
        f_max : `xarray.DataSet`
            Maxwellian distribution function.
        """
        
        #
        # Establish the velocity-space grid in energy coordinates
        #

        if f is not None:
            phi = f._phi
            theta = f._theta
            energy = f._energy
        
        if phi is None:
            dphi = (phi_range[1] - phi_range[0]) / nphi
            phi = np.arange(phi_range[0], phi_range[1], dphi) + dphi/2

        if theta is None:
            dtheta = (theta_range[1] - theta_range[0]) / ntheta
            theta = np.arange(theta_range[0], theta_range[1], dtheta)

        if energy is None:
            energy = np.logspace(energy_range[0], energy_range[1], nenergy, endpoint=False)
            denergy = self.deltaE(energy)
        
        # Velocity does not factor into the entropy of a Maxwellian distribution
        V = np.zeros((3,))

        # Calculate the velocity of each energy bin
        #   - Assume non-relativistic: E = 1/2 m v^2
        #   - If the distribution has been preconditioned, energy will have
        #     endpoints 0 and inf. The inf will trigger a NaN below
        v_mag = np.sqrt(2.0 * eV2J / self.mass * energy)  # m/s
        
        # Expand into a grid
        phi, theta, v_mag = np.meshgrid(phi, theta, v_mag, indexing='ij')

        #
        # Convert spherical energy coordinates to cartesian velocity coordinates
        #
        
        # Comput the components of the look directions of each energy bin
        #   - Negate so that the directions are incident into the detector
        #   - Ignore multiplies by inf
        with np.errstate(invalid='ignore'):
            vxsqr = (-v_mag * np.sin(theta) * np.cos(phi) - (1e3*V[0]))**2
            vysqr = (-v_mag * np.sin(theta) * np.sin(phi) - (1e3*V[1]))**2
            vzsqr = (-v_mag * np.cos(theta) - (1e3*V[2]))**2

        #
        # Expand the LUT grid and Maxwellian targets so they can be broadcast
        # together
        #

        # Velocity targets need 2 new dimensions for the LUT coordinates
        vxsqr = vxsqr[np.newaxis, np.newaxis, ...]
        vysqr = vysqr[np.newaxis, np.newaxis, ...]
        vzsqr = vzsqr[np.newaxis, np.newaxis, ...]

        # LUT coordinates need 3 new dimensions for the velocity targets
        n_data = self.n_data[..., np.newaxis, np.newaxis, np.newaxis]
        t_data = self.t_data[..., np.newaxis, np.newaxis, np.newaxis]
        
        #
        # Calculate the Maxwellian distribution
        #

        f_M = (1e-6 * n_data
               * (self.mass / (2 * np.pi * kB * eV2K * t_data))**(3.0/2.0)
               * np.exp(-self.mass * (vxsqr + vysqr + vzsqr)
                       / (2.0 * kB * eV2K * t_data))
               )
    
        # If there is high energy extrapolation, the last velocity bin will be
        # infinity, making the Maxwellian distribution inf or nan (inf*0=nan).
        # Make the distribution zero at v=inf.
        f_M = np.where(np.isfinite(f_M), f_M, 0)

        f_M = xr.DataArray(f_M,
                           name='max_lut',
                           dims=('n_data', 't_data', 'phi', 'theta', 'energy'),
                           coords={'n_data': self.n_data[:,0],
                                   't_data': self.t_data[0,:],
                                   'phi': phi[:,0,0],
                                   'theta': theta[0,:,0],
                                   'energy': energy,
                                   'U': (('energy',), f._U)}
                           )

        f_M.attrs['Energy_e0'] = f.E0
        f_M.attrs['Lower_energy_integration_limit'] = f.E_low
        f_M.attrs['Upper_energy_integration_limit'] = f.E_high
        f_M.attrs['scpot'] = f.scpot
        f_M.attrs['mass'] = f.mass
        f_M.attrs['time'] = f.time
        f_M.attrs['wrap_phi'] = f.wrap_phi
        f_M.attrs['theta_extrapolation'] = f.theta_extrapolation
        f_M.attrs['high_energy_extrapolation'] = f.high_energy_extrapolation
        f_M.attrs['low_energy_extrapolation'] = f.low_energy_extrapolation

        return f_M

    def apply(self, f, n=None, t=None, s_M=None, sV=None, method='nt'):
        '''
        Create a look-up table of Maxwellian distributions based on density and
        temperature.
        
        Parameters
        ----------
        f : `pymms.data.fpi.Distribution_Function`
            A velocity distribution function from which to take the
            azimuthal and polar look direction, and the energy target
            coordinates
        n : float
            Density of `f`. If `None`, it is calculated.
        t : float
            Scalar temperature of `f`. If `None`, it is calculated.
        
        Returns
        -------
        lookup_table : `xarray.DataArray`
            A Maxwellian distribution at each value of N and T. Returned only if
            *fname* is not specified.
        '''
        if n is None:
            n = f.density()
        if t is None:
            t = f.scalar_temperature(n=n)
        if s_M is None:
            s_M = f.maxwellian_entropy(n=n)
        if sV is None:
            sV = f.vspace_entropy(n=n)
        
        # This will be needed if entropy is another search parameter
        # if s_M is None:
        #     s_M = f.maxwellian_entropy(n=n)
        
        # Create the LUT grid
        self.set_grid(n, t, s_M, sV)

        # Fill the grid with Maxwellian distributions
        self.fill_grid(f)

        # Dimensions of the look-up table
        dims = self.n_lut.shape

        # Minimize density error
        if method == 'n':
            imin = np.argmin(np.abs(self.n_lut - n))
            irow = imin // dims[1]
            icol = imin % dims[1]
            n_M = self.n_lut[irow, icol]
            t_M = self.t_lut[irow, icol]
            s_M = self.s_M_lut[irow, icol]
            sV_M = self.sV_lut[irow, icol]
            f_M = self.lut[irow, icol, ...]

        # Minimize temperature error
        elif method == 't':
            imin = np.argmin(np.abs(self.t_lut - t))
            irow = imin // dims[1]
            icol = imin % dims[1]
            n_M = self.n_lut[irow, icol]
            t_M = self.t_lut[irow, icol]
            s_M = self.s_M_lut[irow, icol]
            sV_M = self.sV_lut[irow, icol]
            f_M = self.lut[irow, icol, ...]

        # Minimize error in both density and temperature
        elif method == 'nt':
            imin = np.argmin(np.sqrt((self.t_lut - t)**2
                                     + (self.n_lut.data - n)**2
                                     ))
            irow = imin // dims[1]
            icol = imin % dims[1]
            n_M = self.n_lut[irow, icol]
            t_M = self.t_lut[irow, icol]
            s_M = self.s_M_lut[irow, icol]
            sV_M = self.sV_lut[irow, icol]
            f_M = self.lut[irow, icol, ...]
        
        # Interpolate
        elif method == 'interp':
            lut_interp = self.lut.interp({'n_data': n, 't_data': t}, method='linear')
            n_M = lut_interp['n']
            t_M = lut_interp['t']
            f_M = lut_interp['f']
        
        n_err = np.abs(n - n_M) / n
        t_err = np.abs(t - t_M) / t
        if (n_err > self.deltan_n):
            raise ValueError('Lookup table density error greater than allowed: '
                             '{0:0.6f} > {1:0.6f}'.format(n_err, self.deltan_n))
        elif (t_err > self.deltat_t):
            raise ValueError('Lookup table density error greater than allowed: '
                             '{0:0.6f} > {1:0.6f}'.format(t_err, self.deltat_t))

        f_M.attrs['species'] = self.species
        return f_M, n_M, t_M, s_M, sV_M

    @staticmethod
    def species_to_mass(species):
        '''
        Return the mass (kg) of the given particle species.
        
        Parameters
        ----------
        species : str
            Particle species: 'i' or 'e'
        
        Returns
        ----------
        mass : float
            Mass of the given particle species
        '''
        if species == 'i':
            mass = constants.m_p
        elif species == 'e':
            mass = constants.m_e
        else:
            raise ValueError(('Unknown species {}. Select "i" or "e".'
                            .format(species))
                            )
        
        return mass
    
    def _plot(self, ax, style, xx, x_M, x_lut, yy, y_M, y_lut, zz=None):

        fig = ax.get_figure()

        if style == 'ntn':
            x = self.n_data
            y = self.t_data
            z = self.n_lut
        elif style == 'ntt':
            x = self.n_data
            y = self.t_data
            z = self.t_lut
        elif style == 'nss':
            x = self.n_data
            y = self.s_M_data[np.newaxis,:].repeat(x.shape[0], axis=0)
            z = self.s_M_lut
        elif style == 'sts':
            y = self.t_data
            x = self.s_M_data[:,np.newaxis].repeat(y.shape[1], axis=1)
            z = self.s_M_lut
        elif style == 'nsVsV':
            x = self.n_data
            y = self.sV_data[np.newaxis,:].repeat(x.shape[0], axis=0)
            z = self.sV_lut
        elif style == 'sVtsV':
            y = self.t_data
            x = self.sV_data[:,np.newaxis].repeat(y.shape[1], axis=1)
            z = self.sV_lut
        elif style == 'Mbar':
            x = self.n_data
            y = self.t_data
            z = (self.sV_lut - zz) / self.sV_lut
        else:
            raise ValueError('style {0} not recognized. Must be (ntn, ntt, nss, tss).')
        
        img = ax.pcolormesh(x, y, z)
        ax.set_title('')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.minorticks_on()

        # Create a colorbar that is aware of the image's new position
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="5%", pad=0.1)
        fig.add_axes(cax)
        cb = fig.colorbar(img, cax=cax, orientation="vertical")
        cb.ax.minorticks_on()
    
        # Plot the location of the measured density and temperature
        ax.plot(xx, yy, linestyle='None', marker='x', color='black')
        ax.plot(x_M, y_M, linestyle='None', marker=r'$\mathrm{E}$', color='black')
        ax.plot(x_lut, y_lut, linestyle='None', marker=r'$\mathrm{L}$', color='black')

        '''
        # Inset axes to show 
        x1, x2, y1, y2 = 0.95*n, 1.05*n, 0.95*t, 1.05*t
        axins = ax.inset_axes(inset_loc,
                              xlim=(x1, x2), ylim=(y1, y2),
                              xticklabels=[], yticklabels=[])
        img = axins.pcolormesh(self.n_data, self.t_data, self.n_lut)
        # img = N_lut.plot(ax=axins, cmap=cm.get_cmap('rainbow', 15), add_colorbar=False, edgecolor='k')
        axins.set_xlabel('')
        axins.set_xlim(x1, x2)
        axins.set_ylabel('')
        axins.set_ylim(y1, y2)
        # axins.imshow(lut.N.data, extent=extent, cmap=cm.get_cmap('rainbow', 15), origin="lower")
        ax.indicate_inset_zoom(axins, edgecolor="black")
    
        # Plot the location of the measured density and temperature
        axins.plot(n, t, linestyle='None', marker='x', color='black')
        axins.plot(n_M, t_M, linestyle='None', marker=r'$\mathrm{M}$', color='black')
        axins.plot(n_EM, t_EM, linestyle='None', marker=r'$\mathrm{E}$', color='black')
        '''

    def plot(self, n, t, s_M, sV,
             n_EM, t_EM, s_EM, sV_EM,
             n_M, t_M, s_lut_M, sV_M, inset_loc='NE'):
        species = 'e'
        
        # Find the error in n and T between the Maxwellian and Measured
        # distribution
        dn_lut = (n - self.n_lut) / n * 100.0
        dt_lut = (t - self.t_lut) / t * 100.0

        if inset_loc == 'NE':
            inset_loc = [0.5, 0.5, 0.47, 0.47] # [x0, y0, width, height]
        elif inset_loc == 'SE':
            inset_loc = [0.0, 0.5, 0.47, 0.47]
        elif inset_loc == 'NW':
            inset_loc = [0.0, 0.5, 0.47, 0.47]
        elif inset_loc == 'SW':
            inset_loc = [0.0, 0.0, 0.47, 0.47]
        
        fig, axes = plt.subplots(nrows=3, ncols=3, squeeze=False, figsize=(11, 7))
        plt.subplots_adjust(left=0.12, right=0.9, bottom=0.1, top=0.95, wspace=1.0, hspace=0.4)

        #
        # 2D Density LUT: t vs. n
        #

        ax = axes[0,0]
        self._plot(ax, 'ntn', n, n_EM, n_M, t, t_EM, t_M)
        ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
        ax.set_ylabel('$t_{'+species+'}$ (eV)')
        ax.collections[0].colorbar.ax.set_ylabel('$n_{'+species+'}$ [$cm^{-3}$]')
        ax.tick_params(axis="x", rotation=45)

        #
        # 2D Temperature LUT: t vs. n
        #

        ax = axes[0,1]
        self._plot(ax, 'ntt', n, n_EM, n_M, t, t_EM, t_M)
        ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
        ax.set_ylabel('$t_{'+species+'}$ (eV)')
        ax.collections[0].colorbar.ax.set_ylabel('$t_{'+species+'}$ [eV]')
        ax.tick_params(axis="x", rotation=45)

        #
        # 2D Mbar LUT: t vs. n
        #

        ax = axes[0,2]
        self._plot(ax, 'Mbar', n, n_EM, n_M, t, t_EM, t_M, zz=sV)
        ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
        ax.set_ylabel('$t_{'+species+'}$ (eV)')
        ax.collections[0].colorbar.ax.set_ylabel('$\overline{M}_{'+species+'}$')
        ax.tick_params(axis="x", rotation=45)

        #
        # 2D Entropy LUT: s vs. n
        #

        ax = axes[1,0]
        self._plot(ax, 'nss', n, n_EM, n_M, s_M, s_EM, s_lut_M)
        ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
        ax.set_ylabel('$s_{M,'+species+'}$ ($J/K/m^{3}$)')
        ax.collections[0].colorbar.ax.set_ylabel('$s_{M,'+species+'}$ [$J/K/m^{3}$]')
        ax.tick_params(axis="x", rotation=45)

        #
        # 2D Entropy LUT: t vs. s
        #

        ax = axes[1,1]
        self._plot(ax, 'sts', s_M, s_EM, s_lut_M, t, t_EM, t_M)
        ax.set_ylabel('$t_{'+species+'}$ (eV)')
        ax.set_xlabel('$s_{M,'+species+'}$ ($J/K/m^{3}$)')
        ax.collections[0].colorbar.ax.set_ylabel('$s_{M,'+species+'}$ [$J/K/m^{3}$]')
        ax.tick_params(axis="x", rotation=45)

        #
        # 2D V-Space Entropy LUT: sV vs. n
        #

        ax = axes[2,0]
        self._plot(ax, 'nsVsV', n, n_EM, n_M, sV, sV_EM, sV_M)
        ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
        ax.set_ylabel('$s_{V,'+species+'}$ ($J/K/m^{3}$)')
        ax.collections[0].colorbar.ax.set_ylabel('$s_{V,'+species+'}$ [$J/K/m^{3}$]')
        ax.tick_params(axis="x", rotation=45)

        #
        # 2D V-Space Entropy LUT: t vs. sV
        #

        ax = axes[2,1]
        self._plot(ax, 'sVtsV', sV, sV_EM, sV_M, t, t_EM, t_M)
        ax.set_ylabel('$t_{'+species+'}$ (eV)')
        ax.set_xlabel('$s_{V,'+species+'}$ ($J/K/m^{3}$)')
        ax.collections[0].colorbar.ax.set_ylabel('$s_{V,'+species+'}$ [$J/K/m^{3}$]')
        ax.tick_params(axis="x", rotation=45)

        return fig, axes


def plot_max_lut(n, t, # s, sv, s_max_moms,
                 n_max, t_max, # s_max, sv_max, sv_rel_max,
                 n_lut, t_lut): #, s_lut, sv_lut, sv_rel_lut):

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
    # util.format_axes(ax, xaxis='off')
    plots.add_legend(ax, [l1[0], l2[0]], corner='SE', horizontal=True)

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
    plots.add_legend(ax, [l1[0], l2[0]], corner='NE', horizontal=True)

    '''
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
    '''

    fig.suptitle('Maxwellian Look-up Table')

    return fig, axes


def main_ts():
    
    # Time at which to select the distribution
    ti = np.datetime64('2017-07-11T22:34:02')

    # Define some input parameters
    sc = 'mms4'
    mode = 'brst'
    optdesc = 'des-dist'
    t0 = dt.datetime(2017, 7, 11, 22, 34, 0)
    t1 = dt.datetime(2017, 7, 11, 22, 34, 5)

    # Load and precondition the distribution functions
    des_dist = fpi.load_dist(sc=sc, mode=mode, optdesc=optdesc,
                             start_date=t0, end_date=t1)
    kwargs = fpi.precond_params(sc=sc, mode=mode, level='l2', optdesc=optdesc,
                                start_date=t0, end_date=t1, time=des_dist['time'])
    des_pre = fpi.precondition(des_dist['dist'], **kwargs)

    scpot = kwargs.pop('scpot')

    # Create the equivalent Maxwellian
    f_M = fpi.maxwellian_distribution(des_pre)
    n_M = fpi.density(f_M)
    t_M = fpi.scalar_temperature(f_M)
    
    # Now find the optimized equivalent Maxwellian distributions
    f_lut = list()
    n_lut = list()
    t_lut = list()

    # Loop over each of the distribution functions
    for idx in tqdm(range(len(des_pre))): # f_fpi, Vsc in tqdm(zip(des_pre, scpot), total=len(scpot)):
        f_fpi = des_pre[idx,...]
        Vsc = scpot[idx]
        _fM = f_M[idx,...]
        _nM = n_M[idx]
        _tM = t_M[idx]

        # Create a distribution function object from the measured distribution
        #   - Provide the preconditioning keywords so that the original and
        #     preconditioned data are present
        fi = fpi.Distribution_Function.from_fpi(f_fpi, scpot=Vsc, **kwargs)
        fi.precondition()
        ni = fi.density()
        ti = fi.scalar_temperature(n=ni)

        # Create a table of Maxwellian distributions around the measured distribution
        species = optdesc[1]
        deltan_n = 0.005
        deltat_t = 0.005

        # Check if they are below the grid resolution
        n_err = np.abs(ni - _nM) / ni   # is this smaller than ∆n/n?
        t_err = np.abs(ti - _tM) / ti   # is this smaller than ∆t/t?
        if (n_err <= deltan_n) and (t_err <= deltat_t):
            print('Equivalent Maxwellian is good enough. Do not apply LUT.')
            f_lut.append(f_M)
            n_lut.append(n_M)
            t_lut.append(t_M)
            
        else:
            lut = Lookup_Table(deltan_n=deltan_n, deltat_t=deltat_t, species=species)

            # Apply the look-up table to the distribution
            #   - Find the Maxwellian distribution within the look-up table that has
            #     a density and temperature most similar to the measured distribution
            _f, _n, _t = lut.apply(fi, n=ni, t=ti) # Does not exist yet
            f_lut.append(_f)
            n_lut.append(_n)
            t_lut.append(_t)
        
        if idx == 25:
            return

    # Plot the results
    fig, axes = lut.plot_max_lut(ni, ti, n_M, t_M, n_lut, t_lut)
    plt.show()


def main(): #sc, mode, optdesc, t0, t1, ti):
    
    # Time at which to select the distribution

    # Define some input parameters
    sc = 'mms4'
    mode = 'brst'
    optdesc = 'des-dist'
    # t0 = dt.datetime(2017, 7, 11, 22, 34, 0)
    # t1 = dt.datetime(2017, 7, 11, 22, 34, 5)
    # tj = np.datetime64('2017-07-11T22:34:02')

    t0 = dt.datetime(2016, 10, 22, 12, 59, 4)
    t1 = dt.datetime(2016, 10, 22, 12, 59, 17)
    tj = np.datetime64('2016-10-22T12:59:08')

    # Load and precondition the distribution functions
    des_dist = fpi.load_dist(sc=sc, mode=mode, optdesc=optdesc,
                             start_date=t0, end_date=t1)
    kwargs = fpi.precond_params(sc=sc, mode=mode, level='l2', optdesc=optdesc,
                                start_date=t0, end_date=t1, time=des_dist['time'])
    f = fpi.precondition(des_dist['dist'], **kwargs)

    # Pick a specific time to create a look-up table
    #   - Select the spacecraft potential and distribution function at that time
    scpot = kwargs.pop('scpot')
    Vsci = scpot.sel(time=tj, method='nearest').data
    f_fpi = des_dist['dist'].sel(time=tj, method='nearest')

    # Create a distribution function object from the measured distribution
    #   - Provide the preconditioning keywords so that the original and
    #     preconditioned data are present
    fi = fpi.Distribution_Function.from_fpi(f_fpi, scpot=Vsci, **kwargs)
    fi.precondition()
    ni = fi.density()
    ti = fi.scalar_temperature(n=ni)
    sVi = fi.vspace_entropy(n=ni)
    si_M = fi.maxwellian_entropy(n=ni)

    # Create a table of Maxwellian distributions around the measured distribution
    species = optdesc[1]
    deltan_n = 0.005
    deltat_t = 0.005
    deltas_s = 0.005
    lut = Lookup_Table(deltan_n=deltan_n, deltat_t=deltat_t, deltas_s=deltas_s, species=species)

    # Create a maxwellian distribution with the same density and temperature
    # as the measured distribution
    f_M = lut.equivalent_maxwellian(fi)
    n_M = f_M.density()
    t_M = f_M.scalar_temperature()
    sV_M = f_M.vspace_entropy(n=n_M)
    s_M = f_M.maxwellian_entropy(n=n_M)

    # Check if they are below the grid resolution
    n_err = np.abs(ni - n_M) / ni   # is this smaller than ∆n/n?
    t_err = np.abs(ti - t_M) / ti   # is this smaller than ∆t/t?
    if (n_err <= deltan_n) and (t_err <= deltat_t):
        print('Equivalent Maxwellian is good enough. Do not apply LUT.')
    else:
        print('n_err = {0:0.4f}\n'
              't_err = {1:0.4f}'
              .format(n_err, t_err))

    # Apply the look-up table to the distribution
    #   - Find the Maxwellian distribution within the look-up table that has
    #     a density and temperature most similar to the measured distribution
    f_M_lut, n_lut, t_lut, s_M_lut, sV_lut = lut.apply(fi, n=ni, t=ti, s_M=si_M, sV=sVi)
    f_M_lut = fpi.Distribution_Function.from_fpi(f_M_lut)

    # Non-Maxwellanity
    Mbar_M = (sV_M - sVi) / sV_M
    Mbar_lut = (sV_lut - sVi) / sV_lut

    # Plot the results
    # Measured = X
    # Equivalent Maxwellian = EM = E
    # LUT Maxwellian = M = 
    fig, axes = lut.plot(ni, ti, si_M, sVi,
                         n_M, t_M, s_M, sV_M,
                         n_lut, t_lut, s_M_lut, sV_lut)

    # Plot reduced distribution functions
    fi_E = fi.reduce('E')
    f_M_E = f_M.reduce('E')
    f_M_lut_E = f_M_lut.reduce('E')
    f_fpi_E = fpi.Distribution_Function.from_fpi(f_fpi).reduce('E')
    f_fpi_pre_E = fpi.Distribution_Function.from_fpi(f.sel(time=tj, method='nearest')).reduce('E')

    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.12)

    ax = axes[0,0]
    f_fpi_E.plot(ax=ax, marker='s', linestyle='None', label='$f_{e,\mathrm{fpi}}$')
    f_fpi_pre_E.plot(ax=ax, marker='p', label='$f_{e,\mathrm{fpi},\mathrm{pre}}$')
    f_M_E.plot(ax=ax, marker='^', label='$f_{e,\mathrm{M}}$')
    f_M_lut_E.plot(ax=ax, marker='*', label='$f_{e,\mathrm{M},\mathrm{LUT}}$')
    fi_E.plot(ax=ax, marker='o', label='$f_{e,\mathrm{pre}}$')
    ax.axvline(Vsci * e * J2eV, color='black')
    ax.set_xlabel('E (eV)')
    ax.set_xscale('log')
    ax.set_ylabel('$f_{e}$\n[$s^{3}/cm^{6}$]')
    ax.set_yscale('log')
    ax.set_ylim(0.5*f_fpi_E[f_fpi_E > 0].min(), 5e1*f_fpi_E.max())
    plots.add_legend(ax, outside=True)
    plt.show(block=False)

    print('                   Measured   Maxwellian         LUT')
    print('Density:            {0:7.4f}      {1:7.4f}     {2:7.4f}'.format(ni, n_M, n_lut))
    print('Temperature:         {0:>7.4f}       {1:>7.4f}     {2:>7.4f}'.format(ti, t_M, t_lut))
    print('Maxwellian Entropy:  {0:2.4e}       {1:2.4e}     {2:2.4e}'.format(si_M, s_M, s_M_lut))
    print('V-Space Entropy:     {0:2.4e}       {1:2.4e}     {2:2.4e}'.format(sVi, sV_M, sV_lut))
    print('Non-Maxwellianity:                  {0:2.4f}     {1:2.4f}'.format(Mbar_M, Mbar_lut))


if __name__ == '__main__':
    main()
