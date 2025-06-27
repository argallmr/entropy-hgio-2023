import numpy as np
import datetime as dt
import xarray as xr
from pathlib import Path
from pymms import config
from pymms.data import fpi
from scipy import constants
from tqdm import tqdm

import matplotlib as mpl
from matplotlib import pyplot as plt, dates as mdates, cm, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import plots

dropbox_root = Path(config['dropbox_root'])
eV2K = constants.value('electron volt-kelvin relationship')
K2eV = constants.value('kelvin-electron volt relationship')
eV2J = constants.value('electron volt-joule relationship')
J2eV = constants.value('joule-electron volt relationship')
e = constants.value('elementary charge')
kB = constants.Boltzmann
m_p = constants.m_p
m_e = constants.m_e

class Lookup_Table():

    def __init__(self, deltan_n=0.005, deltat_t=0.005, species='e'):
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
        S = coeff * (y * S).integrate('U') # J/K/m^3
    
        return S
    
    def _vspace_entropy(self, n=None, s=None):

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
        sv4 = -kB * 1e12 * coeff * self._trapz(sv4.data, sv4['U'].data)

        # Velocity space entropy density
        sv = sv1 + sv2 + sv3 + sv4 # J/K/m^3
    
        return sv
    
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
        dOmega = np.sin(self.lut['theta']) * dtheta * dphi
        with np.errstate(invalid='ignore', divide='ignore'):
            y = np.sqrt(self.lut['U']) / (1 - self.lut['U'])**(5/2)
            lny = np.log(y)
            lndOmega = np.log(dOmega)
            lnydOmega = np.log(y * dOmega)
        y = y.where(np.isfinite(y), 0)
        lny = lny.where(np.isfinite(lny), 0)
        lndOmega = lndOmega.where(np.isfinite(lndOmega), 0)
        lnydOmega = lnydOmega.where(np.isfinite(lnydOmega), 0)
    
        # Terms in that make up the velocity space entropy density
        sv1 = s # J/K/m^3 ln(s^3/m^6) -- Already multiplied by -kB
        sv2 = kB * (1e6*n) * np.log(1e6*n/coeff) # 1/m^3 * ln(1/m^3)
        
        sv3 = (y * lndOmega * np.sin(self.lut['theta']) * self.lut).integrate('phi')
        sv3 = sv3.integrate('theta')
        sv3 = -kB * 1e12 * coeff * sv3.integrate('U') # 1/m^3
    
        sv4 = (np.sin(self.lut['theta']) * self.lut).integrate('phi')
        sv4 = sv4.integrate('theta')
        sv4 = (-kB * 1e12 * coeff
               * ((y * lny * sv4).integrate('U')
                  + self._trapz((y * sv4).data, sv4['U'].data)
                  )
               )
        
        # Velocity space entropy density
        sv = sv1 + sv2 + sv3 + sv4 # J/K/m^3
    
        return sv
    
    @staticmethod
    def _trapz(f, x):
        dx = x[1:] - x[0:-1]
        with np.errstate(divide='ignore', invalid='ignore'):
            F = 0.5  * (f[1:,:,:] + f[0:-1,:,:]) * (dx * np.log(dx))[:,np.newaxis, np.newaxis]
        F = np.where(np.isfinite(F), F, 0)
        
        return np.sum(F, axis=0)
    
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


    def set_grid(self, n, t):
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

        n_data, t_data = self.grid_coords(n_range, t_range)
        n_data, t_data = np.meshgrid(n_data, t_data, indexing='ij')
        self.n_data = n_data
        self.t_data = t_data

    def grid_coords(self, n_range, t_range):
        
        # Determine the number of cells
        N = self.grid_resolution(n_range, self.deltan_n)
        M = self.grid_resolution(t_range, self.deltat_t)

        # Create the grid
        n = np.logspace(np.log10(n_range[0]), np.log10(n_range[1]), N)
        t = np.logspace(np.log10(t_range[0]), np.log10(t_range[1]), M)

        # Set the grid
        return n, t
    
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
        v_mag = v_mag[np.newaxis, np.newaxis, ...]

        # LUT coordinates need 3 new dimensions for the velocity targets
        n_data = self.n_data[..., np.newaxis, np.newaxis, np.newaxis]
        t_data = self.t_data[..., np.newaxis, np.newaxis, np.newaxis]
        
        #
        # Calculate the Maxwellian distribution
        #

        '''
        f_M = (1e-6 * n_data
               * (self.mass / (2 * np.pi * kB * eV2K * t_data))**(3.0/2.0)
               * np.exp(-self.mass * (vxsqr + vysqr + vzsqr)
                       / (2.0 * kB * eV2K * t_data))
               )
        '''
        

        with np.errstate(invalid='ignore'):
            vt = np.sqrt(2 * eV2J / self.mass * t_data) # thermal velocity
            coeff = 1e-6 * n_data / (np.sqrt(np.pi) * vt)**3 #s^3/cm^6
            f_M = coeff * np.exp(-(vxsqr + vysqr + vzsqr) / vt**2) # s^3/cm^6

        # If there is high energy extrapolation, the last velocity bin will be
        # infinity, making the Maxwellian distribution inf or nan (inf*0=nan).
        # Make the distribution zero at v=inf.
        f_M = np.where(np.isfinite(f_M), f_M, 0)

        # Make sure the distribution goes to zero at E=0
        f_M[..., f._energy == 0] = 0

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
        
        # This will be needed if entropy is another search parameter
        # if s_M is None:
        #     s_M = f.maxwellian_entropy(n=n)
        
        # Create the LUT grid
        self.set_grid(n, t)

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
            f_M = self.lut[irow, icol, ...]
        
        # Interpolate
        elif method == 'interp':
            lut_interp = self.lut.interp({'n_data': n, 't_data': t}, method='linear')
            n_M = lut_interp['n']
            t_M = lut_interp['t']
            f_M = lut_interp['f']
        
        n_err = np.abs(n - n_M) / n
        t_err = np.abs(t - t_M) / t

        '''
        if (n_err > self.deltan_n):
            raise ValueError('Lookup table density error greater than allowed: '
                             '{0:0.6f} > {1:0.6f}'.format(n_err, self.deltan_n))
        elif (t_err > self.deltat_t):
            raise ValueError('Lookup table density error greater than allowed: '
                             '{0:0.6f} > {1:0.6f}'.format(t_err, self.deltat_t))
        '''
        
        f_M.attrs['species'] = self.species
        return f_M, n_M, t_M

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
    
    def _plot(self, ax, data, x, x_M, x_lut, y, y_M, y_lut, xx=None, yy=None):

        if xx is None:
            xx = self.n_data
        if yy is None:
            yy = self.t_data

        fig = ax.get_figure()

        img = ax.pcolormesh(xx, yy, data)
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
        ax.plot(x, y, linestyle='None', marker='x', color='black')
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

    def plot(self, n, t, s, sV, s_M,
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
        
        # Derived products

        s_lut = self.entropy()
        s_M_lut = self.maxwellian_entropy(n=self.n_lut)
        sV_lut = self.vspace_entropy(n=self.n_lut, s=s_lut)
        Mbar = (sV_lut - sV) / sV_lut
        MbarKP = (s_lut - s) / (3/2 * kB * 1e6 * self.n_lut)
        
        fig, axes = plt.subplots(nrows=3, ncols=3, squeeze=False, figsize=(11, 7))
        plt.subplots_adjust(left=0.12, right=0.9, bottom=0.1, top=0.95, wspace=1.0, hspace=0.4)

        #
        # 2D Density LUT: t vs. n
        #

        ax = axes[0,0]
        self._plot(ax, self.n_lut, n, n_EM, n_M, t, t_EM, t_M)
        ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
        ax.set_ylabel('$t_{'+species+'}$ (eV)')
        ax.collections[0].colorbar.ax.set_ylabel('$n_{'+species+'}$ [$cm^{-3}$]')
        ax.tick_params(axis="x", rotation=45)

        #
        # 2D Temperature LUT: t vs. n
        #

        ax = axes[0,1]
        self._plot(ax, self.t_lut, n, n_EM, n_M, t, t_EM, t_M)
        ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
        ax.set_ylabel('$t_{'+species+'}$ (eV)')
        ax.collections[0].colorbar.ax.set_ylabel('$t_{'+species+'}$ [eV]')
        ax.tick_params(axis="x", rotation=45)

        #
        # 2D Mbar LUT: t vs. n
        #

        ax = axes[0,2]
        self._plot(ax, Mbar, n, n_EM, n_M, t, t_EM, t_M)
        ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
        ax.set_ylabel('$t_{'+species+'}$ (eV)')
        ax.tick_params(axis="x", rotation=45)
        im = ax.collections[0]
        vlim = np.abs(np.array(im.get_clim())).max()
        im.set_clim(-vlim, vlim)
        im.set_cmap('seismic')
        cb = im.colorbar
        cb.ax.set_ylabel('$\overline{M}_{'+species+'}$')

        #
        # 2D Entropy LUT: s vs. n
        #

        ax = axes[1,0]
        # self._plot(ax, s_lut, n, n_EM, n_M, t, t_EM, t_M)
        self._plot(ax, s_lut, n, n_EM, n_M, s_M, s_EM, s_lut_M,
                   xx=self.n_data, yy=s_M_lut)
        ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
        ax.set_ylabel('$s_{'+species+'}$ ($J/K/m^{3}$)')
        ax.collections[0].colorbar.ax.set_ylabel('$s_{M,'+species+'}$ [$J/K/m^{3}$]')
        ax.tick_params(axis="x", rotation=45)

        #
        # 2D Entropy LUT: s_M vs. t
        #

        ax = axes[1,1]
        self._plot(ax, s_M_lut, t, t_EM, t_M, s_M, s_EM, s_lut_M,
                   xx=self.t_data, yy=s_M_lut)
        ax.set_xlabel('$t_{'+species+'}$ (eV)')
        ax.set_ylabel('$s_{M,'+species+'}$ ($J/K/m^{3}$)')
        ax.collections[0].colorbar.ax.set_ylabel('$s_{M,'+species+'}$ [$J/K/m^{3}$]')
        ax.tick_params(axis="x", rotation=45)

        #
        # 2D M-bar-KP LUT: t vs. n
        #

        ax = axes[1,2]
        self._plot(ax, MbarKP, n, n_EM, n_M, t, t_EM, t_M)
        ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
        ax.set_ylabel('$t_{'+species+'}$ (eV)')
        ax.tick_params(axis="x", rotation=45)
        im = ax.collections[0]
        vlim = np.abs(np.array(im.get_clim())).max()
        im.set_clim(-vlim, vlim)
        im.set_cmap('seismic')
        cb = im.colorbar
        cb.ax.set_ylabel('$\overline{M}_{\mathrm{KP},'+species+'}$')

        #
        # 2D V-Space Entropy LUT: sV vs. n
        #

        ax = axes[2,0]
        self._plot(ax, sV_lut, n, n_EM, n_M, sV, sV_EM, sV_M,
                   xx=self.n_lut, yy=sV_lut)
        ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
        ax.set_ylabel('$s_{V,'+species+'}$ ($J/K/m^{3}$)')
        ax.collections[0].colorbar.ax.set_ylabel('$s_{V,'+species+'}$ [$J/K/m^{3}$]')
        ax.tick_params(axis="x", rotation=45)

        #
        # 2D V-Space Entropy LUT: sV vs. t
        #

        ax = axes[2,1]
        self._plot(ax, sV_lut, t, t_EM, t_M, sV, sV_EM, sV_M,
                   xx=self.n_data, yy=sV_lut)
        ax.set_xlabel('$t_{'+species+'}$ (eV)')
        ax.set_ylabel('$s_{V,'+species+'}$ ($J/K/m^{3}$)')
        ax.collections[0].colorbar.ax.set_ylabel('$s_{V,'+species+'}$ [$J/K/m^{3}$]')
        ax.tick_params(axis="x", rotation=45)

        return fig, axes


def M_bar(sV_M, sV):
    return (sV_M - sV) / sV_M


def M_bar_KP(s_M, s, n):
    return (s_M - s) / (3/2 * kB * 1e6 * n)


def thermal_velocity(T, m):
    v_th = 1e-3 * np.sqrt(2 * kB * T * eV2K / m) # km/s
    return v_th


def plot_max_lut(data):

    species = 'e'

    fig, axes = plt.subplots(nrows=5, ncols=1, squeeze=False, figsize=(5.5, 7))
    plt.subplots_adjust(top=0.95, right=0.95, left=0.17)

    # Error in the adjusted look-up table
    dn_max = (data['n'] - data['n_M']) / data['n'] * 100.0
    dn_lut = (data['n'] - data['n_lut']) / data['n'] * 100.0

    ax = axes[0,0]
    l1 = dn_max.plot(ax=ax, label='$\Delta n_{'+species+',Max}/n_{'+species+',Max}$')
    l2 = dn_lut.plot(ax=ax, label='$\Delta n_{'+species+',lut}/n_{'+species+',lut}$')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('$\Delta n_{'+species+'}/n_{'+species+'}$ (%)')
    plots.format_axes(ax, xaxis='off')
    plots.add_legend(ax, corner='SE', horizontal=True)

    # Deviation in temperature
    dt_max = (data['t'] - data['t_M']) / data['t'] * 100.0
    dt_lut = (data['t'] - data['t_lut']) / data['t'] * 100.0

    ax = axes[1,0]
    l1 = dt_max.plot(ax=ax, label='$\Delta T_{'+species+',Max}/T_{'+species+',Max}$')
    l2 = dt_lut.plot(ax=ax, label='$\Delta T_{'+species+',lut}/T_{'+species+',lut}$')
    ax.set_title('')
    ax.set_xticklabels([])
    ax.set_xlabel('')
    ax.set_ylabel('$\Delta T_{'+species+'}/T_{'+species+'}$ (%)')
    # ax.set_ylim(-1,2.5)
    plots.format_axes(ax, xaxis='off')
    plots.add_legend(ax, corner='NE', horizontal=True)

    '''
    # Deviation in entropy
    ds_moms = (data['s_M_moms'] - data['s']) / data['s'] * 100.0
    ds_max = (data['s_M'] - data['s']) / data['s'] * 100.0
    ds_lut = (data['s_lut'] - data['s']) / data['s'] * 100.0

    ax = axes[2,0]
    l1 = ds_max.plot(ax=ax, label='$\Delta s_{'+species+',Max}/s_{'+species+',Max}$')
    l2 = ds_lut.plot(ax=ax, label='$\Delta s_{'+species+',lut}/s_{'+species+',lut}$')
    l3 = ds_moms.plot(ax=ax, label='$\Delta s_{'+species+',moms}/s_{'+species+',moms}$')
    ax.set_title('')
    ax.set_ylabel('$\Delta s_{'+species+'}/s_{'+species+'}$ (%)')
    # ax.set_ylim(-9,2.5)
    plots.format_axes(ax, xaxis='off')
    plots.add_legend(ax, corner='SE', horizontal=True)

    # Deviation in velocity-space entropy
    dsv_max = (data['sv_M'] - data['sv']) / data['sv'] * 100.0
    dsv_lut = (data['sv_lut'] - data['sv']) / data['sv'] * 100.0

    ax = axes[3,0]
    l1 = dsv_max.plot(ax=ax, label='$\Delta s_{V,'+species+',Max}/s_{V,'+species+',Max}$')
    l2 = dsv_lut.plot(ax=ax, label='$\Delta s_{V,'+species+',lut}/s_{V,'+species+',lut}$')
    ax.set_title('')
    ax.set_ylabel('$\Delta s_{V,'+species+'}/s_{V,'+species+'}$ (%)')
    plots.format_axes(ax, xaxis='off')
    plots.add_legend(ax, corner='SE', horizontal=True)
    '''
    
    #
    # M-Bar
    #
    ax = axes[2,0]
    data['Mbar_M'].plot(ax=ax, label='$\overline{M}_{'+species+',Max}$')
    data['Mbar_lut'].plot(ax=ax, label='$\overline{M}_{'+species+',lut}$')
    ax.set_title('')
    ax.set_ylabel('$\overline{M}_{'+species+'}$')
    plots.format_axes(ax=ax, xaxis='off')
    plots.add_legend(ax=ax, corner='SE', horizontal=True)

    #
    # M-Bar-KP
    #
    ax = axes[3,0]
    data['MbarKP_M'].plot(ax=ax, label='$\overline{M}_{KP,'+species+',Max}$')
    data['MbarKP_lut'].plot(ax=ax, label='$\overline{M}_{KP,'+species+',lut}$')
    data['MbarKP_moms'].plot(ax=ax, label='$\overline{M}_{KP,'+species+',moms}$')
    # data['MbarKP_lut_sv'].plot(ax=ax, label='$\overline{M}_{KP,sV,'+species+',lut}$')
    ax.set_title('')
    ax.set_ylabel('$\overline{M}_{KP,'+species+'}$')
    plots.format_axes(ax, xaxis='off')
    plots.add_legend(ax, corner='SE', horizontal=True)

    #
    # Thermal Velocity
    #
    ax = axes[4,0]
    data['v_th'].plot(ax=ax, label='$v_{th,'+species+'}$')
    data['v_th_M'].plot(ax=ax, label='$v_{th,'+species+',Max}$')
    data['v_th_lut'].plot(ax=ax, label='$v_{th,'+species+',lut}$')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('$v_{th}'+species+'}$\n[km/s]')
    plots.format_axes(ax)
    plots.add_legend(ax, corner='SE', horizontal=True)


    '''
    ax = axes[4,0]
    l1 = sv_rel_max.plot(ax=ax, label='$s_{V,'+species+',rel,Max}$')
    l2 = sv_rel_lut.plot(ax=ax, label='$s_{V,'+species+',rel,lut}$')
    ax.set_title('')
    ax.set_xlabel('')
    ax.set_ylabel('$s_{V,'+species+',rel}$\n[J/K/$m^{3}$]')
    plots.format_axes(ax)
    plots.add_legend(ax, corner='SE', horizontal=True)
    '''

    fig.suptitle('Maxwellian Look-up Table')

    return fig, axes


def main_ts(sc, mode, optdesc, t0, t1):


    #
    # Check if we already ran this interval
    #

    # Create a file name to output the data
    t0_str = t0.strftime('%Y%m%d_%H%M%S')
    if t0.date == t1.date:
        t1_str = t1.strftime('%H%M%S')
    else:
        t1_str = t1.strftime('%Y%m%d_%H%M%S')
    fname = (dropbox_root / '_'.join((sc, optdesc[0:3], mode, 's', t0_str, t1_str))
             ).with_suffix('.nc')
    
    if fname.exists():
        lut_data = xr.load_dataset(fname)
        fig, axes = plot_max_lut(lut_data)
        plt.show()
        return fname, fig, axes
    
    #
    # Get the FPI data
    #

    # Load and precondition the distribution functions
    des_dist = fpi.load_dist(sc=sc, mode=mode, optdesc=optdesc,
                             start_date=t0, end_date=t1)
    kwargs = fpi.precond_params(sc=sc, mode=mode, level='l2', optdesc=optdesc,
                                start_date=t0, end_date=t1, time=des_dist['time'])
    des_pre = fpi.precondition(des_dist['dist'], **kwargs)

    scpot = kwargs.pop('scpot')

    #
    # Determine the Equivalent Maxwellian and its Moments
    #

    # Create the equivalent Maxwellian
    f_M = fpi.maxwellian_distribution(des_pre)
    n_M = fpi.density(f_M)
    t_M = fpi.scalar_temperature(f_M)
    s_M = fpi.entropy(f_M)
    sv_M = fpi.vspace_entropy(f_M, n=n_M, s=s_M)
    sv_rel_M = fpi.relative_entropy(des_pre, f_M)

    #
    # Make lists to gether output data
    #
    
    # Now find the optimized equivalent Maxwellian distributions
    f_lut = list()
    n_lut = list()
    t_lut = list()
    s_lut = list()
    sv_lut = list()
    sv_rel_lut = list()

    n = list()
    t = list()
    s = list()
    sv = list()
    s_M_moms = list()

    #
    # Loop over each distribution, calculate its moments, and
    # apply the look-up table
    #

    # Loop over each of the distribution functions
    for idx in tqdm(range(len(des_pre))): # f_fpi, Vsc in tqdm(zip(des_pre, scpot), total=len(scpot)):
        f_fpi = des_pre[idx,...]
        Vsc = scpot[idx]
        # _fM = f_M[idx,...]
        # _nM = n_M[idx]
        # _tM = t_M[idx]

        # Create a distribution function object from the measured distribution
        #   - Provide the preconditioning keywords so that the original and
        #     preconditioned data are present
        fi = fpi.Distribution_Function.from_fpi(f_fpi, scpot=Vsc, **kwargs)
        fi.precondition()
        ni = fi.density()
        ti = fi.scalar_temperature(n=ni)
        si = fi.entropy()
        svi = fi.vspace_entropy(n=ni, s=si)
        s_max_moms = fi.maxwellian_entropy(n=ni)

        n.append(ni)
        t.append(ti)
        s.append(si)
        sv.append(svi)
        s_M_moms.append(s_max_moms)

        # Create a table of Maxwellian distributions around the measured distribution
        species = optdesc[1]
        deltan_n = 0.005
        deltat_t = 0.005

        # Check if they are below the grid resolution
        n_err = np.abs(ni - n_M[idx]) / ni   # is this smaller than ∆n/n?
        t_err = np.abs(ti - t_M[idx]) / ti   # is this smaller than ∆t/t?
        if (n_err <= deltan_n) and (t_err <= deltat_t):
            print('Equivalent Maxwellian is good enough. Do not apply LUT.')
            _f = fpi.Distribution_Function.from_fpi(f_M[idx,...])
            f_lut.append(_f)
            n_lut.append(n_M[idx].data)
            t_lut.append(t_M[idx].data)
            
        else:
            lut = Lookup_Table(deltan_n=deltan_n, deltat_t=deltat_t, species=species)

            # Apply the look-up table to the distribution
            #   - Find the Maxwellian distribution within the look-up table that has
            #     a density and temperature most similar to the measured distribution
            _f, _n, _t = lut.apply(fi, n=ni, t=ti) # Does not exist yet
            _f = fpi.Distribution_Function.from_fpi(_f, time=des_pre['time'][idx].data)
            f_lut.append(_f)
            n_lut.append(_n)
            t_lut.append(_t)
        
        # Calculate LUT entropy parameters
        _s = _f.entropy()
        s_lut.append(_s)
        sv_lut.append(_f.vspace_entropy(n=_n, s=_s))
        sv_rel_lut.append(fi.relative_entropy(_f))

    #
    # Put everything into a dataset
    #

    # Store the data
    lut_data = xr.Dataset({'n': (('time',), n),
                           't': (('time',), t),
                           's': (('time',), s),
                           'sv': (('time',), sv),
                           's_M_moms': (('time',), s_M_moms),
                           'n_M': n_M,
                           't_M': t_M,
                           's_M': s_M,
                           'sv_M': sv_M,
                           'sv_rel_M': sv_rel_M,
                           'n_lut': (('time',), n_lut),
                           't_lut': (('time',), t_lut),
                           's_lut': (('time',), s_lut),
                           'sv_lut': (('time',), sv_lut),
                           'sv_rel_lut': (('time',), sv_rel_lut)},
                           coords={'time': des_dist['time']})

    # Thermal velocity
    mass = m_e if species == 'e' else m_p
    lut_data['v_th'] = thermal_velocity(lut_data['t'], mass)
    lut_data['v_th_M'] = thermal_velocity(lut_data['t_M'], mass)
    lut_data['v_th_lut'] = thermal_velocity(lut_data['t_lut'], mass)
    
    # Non-Maxwellianity
    lut_data['Mbar_M'] = M_bar(lut_data['sv_M'], lut_data['sv'])
    lut_data['Mbar_lut'] = M_bar(lut_data['sv_lut'], lut_data['sv'])
    lut_data['MbarKP_moms'] = M_bar_KP(lut_data['s_M_moms'], lut_data['s'], lut_data['n'])
    lut_data['MbarKP_M'] = M_bar_KP(lut_data['s_M'], lut_data['s'], lut_data['n'])
    lut_data['MbarKP_lut'] = M_bar_KP(lut_data['s_lut'], lut_data['s'], lut_data['n'])
    lut_data['MbarKP_lut_sv'] = M_bar_KP(lut_data['s_lut'], lut_data['sv'], lut_data['n'])

    # Output data to file
    lut_data.to_netcdf(fname)

    # Plot the results
    fig, axes = plot_max_lut(lut_data)
    
    return fname, fig, axes


def main(sc, mode, optdesc, t0, t1, tj):

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
    si = fi.entropy()
    ti = fi.scalar_temperature(n=ni)
    sVi = fi.vspace_entropy(n=ni, s=si)
    si_M = fi.maxwellian_entropy(n=ni)

    # Create a table of Maxwellian distributions around the measured distribution
    species = optdesc[1]
    deltan_n = 0.005
    deltat_t = 0.005
    deltas_s = 0.005
    lut = Lookup_Table(deltan_n=deltan_n, deltat_t=deltat_t, species=species)

    # Create a maxwellian distribution with the same density and temperature
    # as the measured distribution
    f_M = lut.equivalent_maxwellian(fi)
    n_M = f_M.density()
    t_M = f_M.scalar_temperature()
    sV_M = f_M.vspace_entropy(n=n_M)
    s_M = f_M.maxwellian_entropy(n=n_M)
    sV_rel_M = fi.relative_entropy(f_M)

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
    f_M_lut, n_lut, t_lut = lut.apply(fi, n=ni, t=ti)
    f_M_lut = fpi.Distribution_Function.from_fpi(f_M_lut)
    s_M_lut = f_M_lut.maxwellian_entropy(n=n_lut)
    sV_lut = f_M_lut.vspace_entropy(n=n_lut)
    sV_rel_lut = fi.relative_entropy(f_M_lut)

    # Non-Maxwellanity
    Mbar_M = (sV_M - sVi) / sV_M
    Mbar_lut = (sV_lut - sVi) / sV_lut

    # Plot the results
    # Measured = X
    # Equivalent Maxwellian = EM = E
    # LUT Maxwellian = M = 
    fig, axes = lut.plot(ni, ti, si, sVi, si_M,
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
    ax.set_title(f_fpi['time'].data)
    ax.set_xlabel('E (eV)')
    ax.set_xscale('log')
    ax.set_ylabel('$f_{e}$\n[$s^{3}/cm^{6}$]')
    ax.set_yscale('log')
    ax.set_ylim(0.02*f_fpi_E[f_fpi_E > 0].min(), 5e1*f_fpi_E.max())
    plots.add_legend(ax, outside=True)

    print('                   Measured   Maxwellian         LUT')
    print('Density:            {0:7.4f}      {1:7.4f}     {2:7.4f}'.format(ni, n_M, n_lut))
    print('Temperature:         {0:>7.4f}       {1:>7.4f}     {2:>7.4f}'.format(ti, t_M, t_lut))
    print('Maxwellian Entropy:  {0:2.4e}       {1:2.4e}     {2:2.4e}'.format(si_M, s_M, s_M_lut))
    print('V-Space Entropy:     {0:2.4e}       {1:2.4e}     {2:2.4e}'.format(sVi, sV_M, sV_lut))
    print('Relative Entropy:                   {0:2.4e}     {1:2.4e}'.format(sV_rel_M, sV_rel_lut))
    print('Non-Maxwellianity:                  {0:2.4f}     {1:2.4f}'.format(Mbar_M, Mbar_lut))

    return fig, axes


if __name__ == '__main__':
    import argparse
    import datetime as dt
    from os import path
    
    parser = argparse.ArgumentParser(
        description='Plot an overview of MMS data.'
        )
    
    parser.add_argument('sc', 
                        type=str,
                        help='Spacecraft Identifier')
    
    parser.add_argument('mode', 
                        type=str,
                        help='Data rate mode')
    
    parser.add_argument('start_date', 
                        type=str,
                        help='Start date of the data interval: '
                             '"YYYY-MM-DDTHH:MM:SS""'
                        )
    
    parser.add_argument('end_date', 
                        type=str,
                        help='End date of the data interval: '
                             '"YYYY-MM-DDTHH:MM:SS""'
                        )
    
    parser.add_argument('-t', '--time', 
                        type=str,
                        help='A single look-up table time: '
                             '"YYYY-MM-DDTHH:MM:SS""'
                        )
                        
    parser.add_argument('-o', '--optdesc',
                        type=str,
                        default='des-dist',
                        help='Optional descriptor specifying species')
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-d', '--dir',
                       type=str,
                       help='Path to output destination',
                       )
                        
    group.add_argument('-f', '--filename',
                       type=str,
                       help='Output file name',
                       )
                        
    parser.add_argument('-n', '--no-show',
                        help='Do not show the plot.',
                        action='store_true')

    args = parser.parse_args()
    t0 = dt.datetime.strptime(args.start_date, '%Y-%m-%dT%H:%M:%S')
    t1 = dt.datetime.strptime(args.end_date, '%Y-%m-%dT%H:%M:%S')
    tj = args.time
    if tj is not None:
        try:
            tj = dt.datetime.strptime(tj, '%Y-%m-%dT%H:%M:%S')
        except ValueError:
            tj = dt.datetime.strptime(tj, '%Y-%m-%dT%H:%M:%S.%f')
    
    # Create the look-up table
    if tj is None:
        lut_file, fig, axes = main_ts(args.sc, args.mode, args.optdesc, t0, t1)
        print('LUT file saved to: {0}'.format(lut_file))
    else:
        fig, axes = main(args.sc, args.mode, args.optdesc, t0, t1, tj)
    
    # Save to plot to directory
    if args.dir is not None:
        if tj is not None:
            fname = '_'.join((args.sc, args.optdesc[0:3], args.mode, 'l2', 'lut',
                              tj.strftime('%Y%m%d'), tj.strftime('%H%M%S')))
        elif t0.date() == t1.date():
            fname = '_'.join((args.sc, args.optdesc[0:3], args.mode, 'l2', 'lut',
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%H%M%S')))
        else:
            fname = '_'.join((args.sc, args.optdesc[0:3], args.mode, 'l2', 'lut',
                              t0.strftime('%Y%m%d'), t0.strftime('%H%M%S'),
                              t1.strftime('%Y%m%d'), t1.strftime('%H%M%S')))
        plt.savefig(path.join(args.dir, fname + '.png'))
    
    # Save to file
    if args.filename is not None:
        plt.savefig(args.filename)
    
    # Show on screen
    if not args.no_show:
        plt.show()