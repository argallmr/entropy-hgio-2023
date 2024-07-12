import numpy as np
import xarray as xr
from pymms.data import fpi
from scipy import constants
    
eV2K = constants.value('electron volt-kelvin relationship')
eV2J = constants.eV
kB   = constants.k

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
    def equivalent_maxwellian(self, f, n=None, t=None):
        
        if n is None:
            n = f.density()
        if t is None:
            t = f.scalar_temperature(n=n)

        f_M = f.maxwellian(n, t,
                           phi=f.phi, theta=f.theta, energy=f.energy)

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
        self.lut = self.maxwellian(phi=f.phi, theta=f.theta, energy=f.energy)

        # Precondition the distribution functions
        self.precondition(**kwargs)


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
        n_range = [0.9, 1.1] * n
        t_range = [0.9, 1.1] * t

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
    
    def maxwellian(self,
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
        
        if phi is None:
            dphi = (phi_range[1] - phi_range[0]) / nphi
            phi = np.arange(phi_range[0], phi_range[1], dphi) + dphi/2

        if theta is None:
            dtheta = (theta_range[1] - theta_range[0]) / ntheta
            theta = np.arange(theta_range[0], theta_range[1], dtheta)

        if energy is None:
            energy = np.logspace(energy_range[0], energy_range[1], nenergy, endpoint=False)
            denergy = self.deltaE(energy)
        
        if V is None:
            V = np.zeros((3,))

        # Calculate the velocity of each energy bin
        #   - Assume non-relativistic: E = 1/2 m v^2
        v_mag = np.sqrt(2.0 * eV2J / self.mass * energy)  # m/s
        
        # Expand into a grid
        phi, theta, v_mag = np.meshgrid(phi, theta, v_mag, indexing='ij')

        #
        # Convert spherical energy coordinates to cartesian velocity coordinates
        #
        
        # Comput the components of the look directions of each energy bin
        #   - Negate so that the directions are incident into the detector
        vxsqr = (-v_mag * np.sin(theta) * np.cos(phi) - (1e3*V[0]))**2
        vysqr = (-v_mag * np.sin(theta) * np.sin(phi) - (1e3*V[1]))**2
        vzsqr = (-v_mag * np.cos(theta) - (1e3*V[2]))**2
        
        #
        # Expand the LUT grid and Maxwellian targets so they can be broadcast
        # together
        #

        # Velocity targets need 2 new dimensions for the LUT coordinates
        vxsqr = vxsqr[np.newaxis, np.newaxis, ...]
        vxsqr = vxsqr[np.newaxis, np.newaxis, ...]
        vxsqr = vxsqr[np.newaxis, np.newaxis, ...]

        # LUT coordinates need 3 new dimensions for the velocity targets
        n_data = self.n_data[..., np.newaxis, np.newaxis, np.newaxis]
        t_data = self.t_data[..., np.newaxis, np.newaxis, np.newaxis]
        
        #
        # Calculate the Maxwellian distribution
        #

        f_M = (1e-6 * n_data
               * (self.mass / (2 * np.pi * kB * eV2K * t))**(3.0/2.0)
               * np.exp(-self.mass * (vxsqr + vysqr + vzsqr)
                       / (2.0 * kB * eV2K * t_data))
               )

        f_M = xr.DataArray(f_M,
                           name='max_lut',
                           dims=('n_data', 't_data', 'phi', 'theta', 'energy'),
                           coords={'n_data': self.n_data[:,0],
                                   't_data': self.t_data[0,:],
                                   'phi': phi,
                                   'theta': theta,
                                   'energy': energy}
                           )

        return f_M

    def apply(self, f, n=None, t=None, fname=None):
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
        self.fill_grid()

        dens, temp = grid(n_range, t_range, deltan_n=deltan_n, deltat_t=deltat_t)
        vel = xr.DataArray(np.zeros((1,3)),
                        dims=['time', 'velocity_index'],
                        coords={'velocity_index': ['Vx', 'Vy', 'Vz']})
        
        # lookup_table = xr.zeros_like(dist.squeeze()).expand_dims({'N': N, 'T': T})
        lookup_table = np.zeros((N, M, *np.squeeze(dist).shape))
        n_lookup = np.zeros((N, M))
        v_lookup = np.zeros((N, M, 3))
        t_lookup = np.zeros((N, M))
        # s_lookup = np.zeros(dims)
        # sv_lookup = np.zeros(dims)
        for jn, n in enumerate(dens):
            for it, t in enumerate(temp):
                f_M = maxwellian_distribution(dist, N=n, bulkv=vel, T=t)
                n_M = density(f_M) # f_M.densit()
                V_M = velocity(f_M, n=n_M)
                t_M = scalar_temperature(f_M, n=n_M, V=V_M)
                # s = entropy(f_max)
                # sv = vspace_entropy(f_max, N=n, s=s)
                
                lookup_table[jn, it, ...] = f_M.squeeze()
                n_lookup[jn, it] = n_M
                v_lookup[jn, it, :] = V_M
                t_lookup[jn, it] = t_M
                # s_lookup[idens, itemp] = s
                # sv_lookup[idens, itemp] = sv
        
        
        # Maxwellian density, velocity, and temperature are functions of input data
        dens = xr.DataArray(dens, dims=('n_data',), attrs={'err': deltan_n})
        temp = xr.DataArray(temp, dims=('t_data',), attrs={'err': deltat_t})
        n = xr.DataArray(n_lookup,
                        dims = ('n_data', 't_data'),
                        coords = {'n_data': dens,
                                't_data': temp})
        V = xr.DataArray(v_lookup,
                        dims = ('n_data', 't_data', 'v_index'),
                        coords = {'n_data': dens,
                                't_data': temp,
                                'v_index': ['Vx', 'Vy', 'Vz']})
        t = xr.DataArray(t_lookup,
                        dims = ('n_data', 't_data'),
                        coords = {'n_data': dens,
                                't_data': temp})
        
        # delete duplicate data
        del n_lookup, v_lookup, t_lookup
        
        
        # The look-up table is a function of Maxwellian density, velocity, and
        # temperature. This provides a mapping from measured data to discretized
        # Maxwellian values
        f = xr.DataArray(lookup_table,
                        dims = ('n_data', 't_data', 'phi_index', 'theta', 'energy_index'),
                        coords = {'n': n,
                                't': t,
                                'phi': dist['phi'].squeeze(),
                                'theta': dist['theta'],
                                'energy': dist['energy'].squeeze(),
                                'U': dist['U'].squeeze(),
                                'n_data': dens,
                                't_data': temp})
        '''
        s = xr.DataArray(s_lookup,
                        dims = ('N_data', 't_data'),
                        coords = {'N': n,
                                'T': t,
                                'N_data': N,
                                't_data': T})
        
        sv = xr.DataArray(sv_lookup,
                        dims = ('N_data', 't_data'),
                        coords = {'N': n,
                                    'T': t,
                                    'N_data': N,
                                    't_data': T})
        '''
        
        # Delete duplicate data
        del lookup_table #, s_lookup, sv_lookup
        
        # Put everything into a dataset
        ds = (xr.Dataset({'n': n, 'V': V, 't': t, 'f': f})
            .reset_coords(names=['n', 't'])
            )
        #                 's': s, 'sv': sv})
        
        if fname is None:
            return ds
        else:
            ds.to_netcdf(path=fname)
            return fname
    
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

    # Pick a time to create a Maxwellian distribution
    ti = np.datetime64('2017-07-11T22:34:02')
    fi = fpi.Distribution_Function.from_fpi(f.sel(time=ti, method='nearest'))
    ni = fi.density()
    ti = fi.scalar_temperature()

    import pdb
    pdb.set_trace()

    # Create a Maxwellian distribution
    species = optdesc[1]
    lut = Lookup_Table(species)
    f_M = lut.equivalent_maxwellian(fi)
    n_M = f_M.density()
    t_M = f_M.scalar_temperature()

    # Check if they are below the grid resolution
    n_err = (ni - n_M) / ni   # is this smaller than ∆n/n?
    t_err = (ti - t_M) / ti   # is this smaller than ∆t/t?

    # Create the look-up table
    lut.set_grid(ni, ti) # already in lut.apply, but good to test separately
    
    lut.fill_grid(fi) # already in lut.apply, but good to test separately
    
    lut.apply() # Does not exist yet
