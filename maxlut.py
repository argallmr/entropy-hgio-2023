import numpy as np
import xarray as xr
from pymms.data import fpi
from scipy import constants

import matplotlib as mpl
from matplotlib import pyplot as plt, dates as mdates, cm, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    
eV2K = constants.value('electron volt-kelvin relationship')
eV2J = constants.eV
kB   = constants.k
K2eV = constants.value('kelvin-electron volt relationship')

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

        coeff = 1e6 * np.sqrt(2) * (eV2J * self.lut.attrs['E0'] / self.mass)**(3/2)
        n = coeff * (y * n).integrate('U')

        return n

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
    
        coeff = -1e3 * 2 * (eV2J * self.lut.attrs['E0'] / self.mass)**2 / n
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

        coeff = 1e6 * (2/self.mass)**(3/2) / (n * kB / K2eV) * (self.lut.attrs['E0']*eV2J)**(5/2)
        T = (coeff.data[..., np.newaxis, np.newaxis]
             * np.trapz(y[np.newaxis, np.newaxis, :, np.newaxis, np.newaxis]
                        * T, self.lut['U'], axis=2)
             - (1e6 * self.mass / kB * K2eV * Vij)
             )

        return T

    def scalar_temperature(self, n=None, V=None, T=None):
        
        if T is None:
            T = self.temperature(n=n, V=V)
        return (T[:,:,0,0] + T[:,:,1,1] + T[:,:,2,2]) / 3.0

    
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
        print('Look-up Table will be NxM = {0}x{1}'.format(N, M))

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

        f_M.attrs['E0'] = f.E0
        f_M.attrs['E_low'] = f.E_low
        f_M.attrs['E_high'] = f.E_high
        f_M.attrs['scpot'] = f.scpot
        f_M.attrs['mass'] = f.mass
        f_M.attrs['time'] = f.time
        f_M.attrs['wrap_phi'] = f.wrap_phi
        f_M.attrs['theta_extrapolation'] = f.theta_extrapolation
        f_M.attrs['high_energy_extrapolation'] = f.high_energy_extrapolation
        f_M.attrs['low_energy_extrapolation'] = f.low_energy_extrapolation

        return f_M

    def apply(self, f, n=None, t=None, method='nt'):
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
            f_M = self.lut[irow, icol, ...]

        # Minimize temperature error
        elif method == 't':
            imin = np.argmin(np.abs(self.t_lut - t))
            irow = imin // dims[1]
            icol = imin % dims[1]
            n_M = self.n_lut[irow, icol]
            t_M = self.t_lut[irow, icol]
            f_M = self.lut[irow, icol, ...]

        # Minimize error in both density and temperature
        elif method == 'nt':
            imin = np.argmin(np.sqrt((self.t_lut - t)**2
                                     + (self.n_lut.data - n)**2
                                     ))
            irow = imin // dims[1]
            icol = imin % dims[1]
            n_M = self.n_lut[irow, icol].data
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
        if (n_err > self.deltan_n):
            raise ValueError('Lookup table density error greater than allowed: '
                             '{0:0.6f} > {1:0.6f}'.format(n_err, self.deltan_n))
        elif (t_err > self.deltat_t):
            raise ValueError('Lookup table density error greater than allowed: '
                             '{0:0.6f} > {1:0.6f}'.format(t_err, self.deltat_t))
        else:
            print('n_err = {0:0.4f}\n'
                  't_err = {1:0.4f}'
                  .format(n_err, t_err))

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

    def plot(self, n, t, n_EM, t_EM, n_M, t_M, inset_loc='NE'):
        species = optdesc[1]
        
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
        
        fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8.5, 4))
        plt.subplots_adjust(wspace=0.9)

        #
        # 2D Density LUT
        #

        ax = axes[0,0]
        img = ax.pcolormesh(self.n_data, self.t_data, self.n_lut)
        ax.set_title('')
        ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
        ax.set_xscale('log')
        ax.set_ylabel('$t_{'+species+'}$ (eV)')
        ax.set_yscale('log')
        ax.minorticks_on()

        # Create a colorbar that is aware of the image's new position
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="5%", pad=0.1)
        fig.add_axes(cax)
        cb = fig.colorbar(img, cax=cax, orientation="vertical")
        cb.set_label('$n_{'+species+'} [cm^{-3}]$')
        cb.ax.minorticks_on()
    
        # Plot the location of the measured density and temperature
        ax.plot(n, t, linestyle='None', marker='x', color='black')
        ax.plot(n_M, t_M, linestyle='None', marker=r'$\mathrm{M}$', color='black')
        ax.plot(n_EM, t_EM, linestyle='None', marker=r'$\mathrm{E}$', color='black')

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

        #
        # 2D Temperature LUT
        #
        ax = axes[0,1]
        img = ax.pcolormesh(self.n_data, self.t_data, self.t_lut)
        ax.set_title('')
        ax.set_xlabel('$n_{'+species+'}$ ($cm^{-3}$)')
        ax.set_xscale('log')
        ax.set_ylabel('$t_{'+species+'}$ (eV)')
        ax.set_yscale('log')

        # Create a colorbar that is aware of the image's new position
        divider = make_axes_locatable(ax)
        cax = divider.new_horizontal(size="5%", pad=0.1)
        fig.add_axes(cax)
        cb = fig.colorbar(img, cax=cax, orientation="vertical")
        cb.set_label('$t_{'+species+'} [cm^{-3}]$')
        cb.ax.minorticks_on()
    
        # Plot the location of the measured density and temperature
        ax.plot(n, t, linestyle='None', marker='x', color='black')
        ax.plot(n_M, t_M, linestyle='None', marker=r'$\mathrm{M}$', color='black')
        ax.plot(n_EM, t_EM, linestyle='None', marker=r'$\mathrm{E}$', color='black')
        
        return fig, axes


if __name__ == '__main__':
    from pymms.data import fpi
    import datetime as dt

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
    f = fpi.precondition(des_dist['dist'], **kwargs)

    ti = np.datetime64('2017-07-11T22:34:02')
    scpot = kwargs.pop('scpot')
    Vsci = scpot.sel(time=ti, method='nearest').data
    f_fpi = des_dist['dist'].sel(time=ti, method='nearest')
    fi = fpi.Distribution_Function.from_fpi(f_fpi, scpot=Vsci, **kwargs)
    fi.precondition()
    ni = fi.density()
    ti = fi.scalar_temperature(n=ni)

    # Create a Maxwellian distribution
    species = optdesc[1]
    deltan_n = 0.005
    deltat_t = 0.005
    lut = Lookup_Table(deltan_n=deltan_n, deltat_t=deltat_t, species=species)

    f_M = lut.equivalent_maxwellian(fi)
    n_M = f_M.density()
    t_M = f_M.scalar_temperature()

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
    f_M, n_lut, t_lut = lut.apply(fi, n=ni, t=ti) # Does not exist yet

    # Plot the results
    fig, axes = lut.plot(ni, ti, n_M, t_M, n_lut, t_lut)
    plt.show()