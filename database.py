import datetime as dt
import xarray as xr
import numpy as np
from scipy import constants as c
from pathlib import Path

from pymms.data import util, fgm, edp, fpi, edi
from pymms import config

import tools, physics

eV2K = c.value('electron volt-kelvin relationship')

# Put data created for this project in ~/data/2023-hgio/
output_dir = (Path(config['data_root']) / '../2023-hgio/').resolve()

def filename(sc, mode, t0, t1, optdesc=None):
    '''
    Create a filename for database files.

    Parameters
    ----------
    sc : str
        Spacecraft identifier: (mms, mms1, mms2, mms3, mms4)
    mode : str
        Data rate mode (srvy, brst)
    t0, t1 : `datetime.datetime`
        Start and end of the time of the data interval
    optdesc : str
        Optional filename descriptor
    
    Returns
    -------
    file_path : path-like
        Full file path.
    '''
    valid_sc = ('mms', 'mms1', 'mms2', 'mms3', 'mms4')
    if sc not in valid_sc:
        raise ValueError('sc {0} not in {1}'.format(sc, valid_sc))
    
    # Create a file name
    if optdesc is not None:
        stem = '_'.join((sc, 'hgio', mode, optdesc))
    else:
        stem = '_'.join((sc, 'hgio', mode))
    
    # Time part
    if t0.date() == t1.date():
        fname = '_'.join((stem,
                          t0.strftime('%Y%m%d'),
                          t0.strftime('%H%M%S'),
                          t1.strftime('%H%M%S')))
    else:
        fname = '_'.join((stem,
                          t0.strftime('%Y%m%d'),
                          t0.strftime('%H%M%S'),
                          t1.strftime('%Y%m%d'),
                          t1.strftime('%H%M%S')))
    
    # Put everything together
    file_path = (output_dir / fname).with_suffix('.nc')
    
    return file_path


def load_data(t0, t1, mode='brst', overwrite=False, dt_out=np.timedelta64(30, 'ms')):
    '''
    Load MMS data for a given time interval. Data loaded are the
      * MEC - spacecraft position
      * FGM - magnetic field in GSE
      * EDP - electric field in GSE
      * DIS - ion velocity, scalar pressure, pressure tensor
      * DES - electron velocity, scalar pressure, pressure tensor
    
    Data coordinates and time cadence are standardized.

    Parameters
    ----------
    t0, t1 : `datetime.datetime`
        Start and end of the time interval to be loaded
    mode : str
        Data rate mode (srvy | brst)
    dt_out, `numpy.timedelta64`
        Sample interval to which all data is resampled. Default is the 30ms
        sample interval of DES
    
    Output
    ------
    fname : `pathlib.Path`
        Path to netCDF file containing the data.
    '''

    # Create a file name
    fname = filename('mms', mode, t0, t1)

    # If it already exists, return it
    if fname.exists() and (not overwrite):
        return fname

    # Get the data from each spacecraft
    des_data = get_data('des', mode, t0, t1, dt_out)
    dis_data = get_data('dis', mode, t0, t1, dt_out)
    edi_data = get_data('edi', mode, t0, t1, dt_out)
    edp_data = get_data('edp', mode, t0, t1, dt_out)
    fgm_data = get_data('fgm', mode, t0, t1, dt_out)
    mec_data = get_data('mec', mode, t0, t1, dt_out)

    # Combine into a single dataset
    data = xr.merge([des_data, dis_data, edi_data,
                     edp_data, fgm_data, mec_data])

    # Save to data file
    if 'units' in data['time'].attrs:
        data['time'].attrs['UNITS'] = data['time'].attrs['units']
        del data['time'].attrs['units']

    data.to_netcdf(fname)

    return fname


def get_data(instr, mode, t0, t1, dt_out=np.timedelta64(30, 'ms')):
    '''
    Get data from a single instrument. Data coordinates and time cadence
    are standardized.

    Parameters
    ----------
    instr : str
        Name of the instrument for which to load data
        ('fgm', 'edp', 'des', 'dis', 'mec')
    mode : str
        Data rate mode (srvy | brst)
    t0, t1 : `datetime.datetime`
        Start and end of the time interval to be loaded
    dt_out, `numpy.timedelta64`
        Sample interval to which all data is resampled. Default is the 30ms
        sample interval of DES
    
    Output
    ------
    data : `xarray.Dataset`
        The data
    '''

    extrapolate = False
    method = 'nearest'

    # Pick the proper data-loading function
    if instr == 'mec':
        func = get_mec_data
        method = 'linear' # Upsample 30s to 30ms
        extrapolate = True
    elif instr == 'fgm':
        func = get_fgm_data
        method = 'linear'
    elif instr == 'edp':
        func = get_edp_data
        method = 'nearest' # Upsample 31.25ms to 30ms
    elif instr == 'des':
        func = get_des_data
        method = 'interp_gaps' # Resample 30ms to 30ms; Nearest w/gap detection
        extrapolate = True
    elif instr == 'dis':
        func = get_dis_data
        method = 'interp_gaps' # Resample 150ms to 30ms; Nearest w/gap detection
        extrapolate = True
    elif instr == 'edi':
        func = get_edi_data
        method = 'None'
        extrapolate = True
    else:
        raise ValueError('Instrument {0} not recognized'.format(instr))
    
    # Download the data from each spacecraft
    spacecraft = ['mms1', 'mms2', 'mms3', 'mms4']

    # A place to store the data from each spacecraft
    ds = xr.Dataset()

    # Get the data from each spacecraft
    for sc in spacecraft:
        # Load the data
        try:
            data = func(sc, mode, t0, t1)
        except Exception as E:
            print('Error processing {0} {1} {2} during interval {3} - {4}'
                  .format(sc, instr, mode, t0, t1))
            print('\t{0}'.format(type(E)))
            print('\t{0}'.format(E))
            continue

        # Resample the data to a common time stamp
        data = tools.resample(data, t0, t1, dt_out,
                              extrapolate=extrapolate, method=method)

        # Store the data in the database
        try:
            ds[data.name] = data
        except AttributeError:
            ds = xr.merge([ds, data])
    
    return ds


def get_mec_data(sc, mode, t0, t1):
    '''
    Get MEC data for a given spacecraft and time interval.

    Parameters
    ----------
    sc : str
        The spacecraft for which data is to be loaded
        ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Data rate mode (srvy | brst)
    t0, t1 : `datetime.datetime`
        Start and end of the time interval to be loaded
    
    Output
    ------
    r_data : `xarray.Dataset`
        The spacecraft position in GSE coordinates
    '''
    if mode != 'srvy':
        mode = 'srvy'

    # Make sure we get at least two samples to enable linear interpolation.
    #   - Sample rate is once per 30 sec. M
    if (t1 - t0) < dt.timedelta(seconds=30):
        t0 -= dt.timedelta(seconds=30)
        t1 += dt.timedelta(seconds=30)

    # MEC
    #   - Sampled once per 30 sec
    r_data = util.load_data(sc=sc, instr='mec', mode='srvy', level='l2', optdesc='epht89d', start_date=t0, end_date=t1)

    # Select the position in GSE coordinates
    #   - Standardize time and componets
    #   - dt_plus is required by tools.resample to determine sample rate
    #   - Use unique names for variables from each spacecraft
    r_data = (r_data[[sc+'_mec_r_gse', sc+'_mec_mlat', sc+'_mec_mlt', sc+'_mec_l_dipole']]
              .rename({'Epoch': 'time',
                       sc+'_mec_r_gse_label': 'component',
                       sc+'_mec_r_gse': 'r' + sc[-1],
                       sc+'_mec_mlat': 'mlat' + sc[-1],
                       sc+'_mec_mlt': 'mlt' + sc[-1],
                       sc+'_mec_l_dipole': 'l' + sc[-1],})
              .assign_coords({'component': ['x', 'y', 'z'],
                              'dt_plus': np.timedelta64(int(30e9), 'ns')})
              )

    return r_data

def get_fgm_data(sc, mode, t0, t1):
    '''
    Get FGM data for a given spacecraft and time interval.

    Parameters
    ----------
    sc : str
        The spacecraft for which data is to be loaded
        ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Data rate mode (srvy | brst)
    t0, t1 : `datetime.datetime`
        Start and end of the time interval to be loaded
    
    Output
    ------
    b_data : `xarray.Dataset`
        The vector magnetic field in GSE coordinates
    '''

    # FGM
    #   - sampled at 4 S/s in survey mode and 128 S/s in burst mode.
    #   - Select burst mode for its higher time resolution.
    b_data = fgm.load_data(sc=sc, mode=mode, start_date=t0, end_date=t1)

    # Select the magnetic field in GSE coordinates
    #   - Remove the total magnetic field
    #   - NOTE: in Survey mode there can be a mix of sample rates (fast/slow)
    b_data = (b_data['B_GSE'][:,0:3]
              .rename({'b_index': 'component'})
              .assign_coords({'dt_plus': (b_data['time_delta'].mean() * 1e9).astype('timedelta64[ns]')})
              )

    # Move the time stamp to the beginning of the sample interval
    b_data['time'] = b_data['time'] - b_data['dt_plus']
    b_data['dt_plus'] = 2*b_data['dt_plus']

    # Name with the spacecraft number to make it unique
    b_data.name = 'B' + sc[-1]

    return b_data

def get_edi_data(sc, mode, t0, t1):
    '''
    Load EDI data. Time tags of EDI data are moved to the beginning of the
    accumulation interval to facilitate binning of other data products.
    Parameters
    ----------
    sc : str
        Spacecraft identifier: {'mms1', 'mms2', 'mms3', 'mms4'}
    mode : str
        Data rate mode: {'srvy', 'slow', 'fast', 'brst'}
    level : str
        Data level: {'l1a', 'l2'}
    t0, t1: `datetime.datetime`
        Start and end of the data interval
    binned : bool
        Bin/average data into 5-minute intervals
    Returns
    -------
    edi_data : `xarray.Dataset`
        EDI electric field data
    '''

    level = 'l2'
    tm_vname = '_'.join((sc, 'edi', 't', 'delta', 'minus', mode, level))
    tp_vname = '_'.join((sc, 'edi', 't', 'delta', 'plus', mode, level))

    # Get EDI data
    # edi_data = edi.load_data(sc, mode, level, optdesc='efield', start_date=ti, end_date=te)
    edi_data = edi.load_efield(sc=sc, mode=mode, level=level, start_date=t0, end_date=t1)

    # Timestamps begin on 0's and 5's and span 5 seconds. The timestamp is at
    # the weighted mean of all beam hits. To get the beginning of the timestamp,
    # subtract the time's DELTA_MINUS. But this is inaccurate by a few nanoseconds
    # so we have to round to the nearest second.
    edi_time = edi_data['Epoch'] - edi_data[tm_vname].astype('timedelta64[ns]')
    edi_time = [(t - tdelta)
                if tdelta.astype(int) < 5e8
                else (t + np.timedelta64(1, 's') - tdelta)
                for t, tdelta in zip(edi_time.data, edi_time.data - edi_time.data.astype('datetime64[s]'))
                ]

    # Replace the old time data with the corrected time data
    edi_data['Epoch'] = edi_time
    edi_data = edi_data.assign_coords({'dt_plus': np.timedelta64(int((edi_data[tm_vname]
                                                                      + edi_data[tp_vname]
                                                                      ).mean()), 'ns')}
                                      )

    # Rename Epoch to time for consistency across all data files
    edi_data = (edi_data[['E_GSE', 'V_GSE', 'dt_plus']]
                .drop('V_index')
                .rename({'Epoch': 'time',
                         'E_index': 'component',
                         'V_index': 'component',
                         'E_GSE': 'E'+sc[-1]+'_GSE',
                         'V_GSE': 'V'+sc[-1]+'_GSE'})
                .assign_coords({'component': ['x', 'y', 'z']})
                )

    return edi_data


def get_edp_data(sc, mode, t0, t1):
    '''
    Get EDP data for a given spacecraft and time interval.

    Parameters
    ----------
    sc : str
        The spacecraft for which data is to be loaded
        ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Data rate mode (srvy | brst)
    t0, t1 : `datetime.datetime`
        Start and end of the time interval to be loaded
    
    Output
    ------
    e_data : `xarray.Dataset`
        The vector electric field in GSE coordinates
    '''

    # The data will be interpolated to the DES time scale
    #   - Only use survey data
    if mode != 'srvy':
        mode = 'srvy'

    # EDP
    #   - data rate is 32 S/s in survey mode and 4098 S/s in burst mode.
    #   - Since survey mode sampling rate is closer to the FGM burst mode sample rate, we will load EDP survey data.
    e_data = edp.load_data(sc=sc, mode=mode, start_date=t0, end_date=t1)

    # Select the electric field in GSE coordinates
    #   - Standardize components
    e_data = (e_data['E_GSE']
              .rename({sc+'_edp_label1_fast_l2': 'component'})
              .assign_coords({'dt_plus': np.timedelta64((e_data[sc+'_edp_deltap_fast_l2'].item()), 'ns'),
                              'component': ['x', 'y', 'z']})
              )

    # Put the time stamp at the beginning of the sample interval
    e_data['time'] = e_data['time'] - e_data['dt_plus']
    e_data['dt_plus'] = 2*e_data['dt_plus']

    # Name with the spacecraft number to make it unique
    e_data.name = 'E' + sc[-1]
    
    return e_data


def get_des_data(sc, mode, t0, t1):
    '''
    Get DES data for a given spacecraft and time interval.

    Parameters
    ----------
    sc : str
        The spacecraft for which data is to be loaded
        ('mms1', 'mms2', 'mms3', 'mms4')
    mode : str
        Data rate mode (srvy | brst)
    t0, t1 : `datetime.datetime`
        Start and end of the time interval to be loaded
    
    Output
    ------
    des_data : `xarray.Dataset`
        The velocity, scalar pressure, and pressure tensor
    '''

    # DES
    des_data = fpi.load_moms(sc=sc, mode=mode, optdesc='des-moms',
                             start_date=t0, end_date=t1, center_times=False)

    # Select the velocity
    v_data = (des_data['velocity']
              .rename({'velocity_index': 'component'})
              .assign_coords({'component': ['x', 'y', 'z']})
              )
    
    # Select the scalar pressure
    p_data =  des_data['p']

    # Select the pressure tensor
    #   - Stanardize the components
    P_data = (des_data['prestensor']
              .drop(['cart_index_dim1', 'cart_index_dim2'])
              .rename({'cart_index_dim1': 'comp1',
                       'cart_index_dim2': 'comp2'})
              .assign_coords({'comp1': ['x', 'y', 'z'],
                              'comp2': ['x', 'y', 'z']})
              )
    
    # Combine each into a single dataset
    return (xr.Dataset({'ne'+sc[-1]: des_data['density'],
                        'Ve'+sc[-1]: v_data,
                        'pe'+sc[-1]: p_data,
                        'Pe'+sc[-1]: P_data})
            .assign_coords({'dt_plus': np.timedelta64(int(1e9*des_data['Epoch_plus_var']), 'ns')})
            )


def get_dis_data(sc, mode, t0, t1):
    '''
    Get DIS data for a given spacecraft and time interval.

    Parameters
    ----------
    sc : str
        The spacecraft for which data is to be loaded
        ('mms1', 'mms2', 'mms3', 'mms4')
    t0, t1 : `datetime.datetime`
        Start and end of the time interval to be loaded
    
    Output
    ------
    dis_data : `xarray.Dataset`
        The velocity, scalar pressure, and pressure tensor
    '''

    # DIS
    dis_data = fpi.load_moms(sc=sc, mode=mode, optdesc='dis-moms',
                             start_date=t0, end_date=t1, center_times=False)

    # Select the velocity
    v_data = (dis_data['velocity']
              .rename({'velocity_index': 'component'})
              .assign_coords({'component': ['x', 'y', 'z']})
              )
    
    # Select the scalar pressure
    p_data =  dis_data['p']

    # Select the pressure tensor
    #   - Stanardize the components
    P_data = (dis_data['prestensor']
              .drop(['cart_index_dim1', 'cart_index_dim2'])
              .rename({'cart_index_dim1': 'comp1',
                       'cart_index_dim2': 'comp2'})
              .assign_coords({'comp1': ['x', 'y', 'z'],
                              'comp2': ['x', 'y', 'z']})
              )
    
    # Combine each into a single dataset
    return (xr.Dataset({'ni'+sc[-1]: dis_data['density'],
                        'Vi'+sc[-1]: v_data,
                        'pi'+sc[-1]: p_data,
                        'Pi'+sc[-1]: P_data})
            .assign_coords({'dt_plus': np.timedelta64(int(1e9*dis_data['Epoch_plus_var']), 'ns')})
            )


def load_entropy(mode, start_date, end_date,
                 optdesc='des-dist'):
    '''
    Calculate the change in relative energy per particle

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

    # Load the dataset (but do not download/wrangle the data)
    fname = filename('mms', mode, start_date, end_date)
    if not fname.exists():
        raise ValueError('Run load_data first. File does not exist:\n{0}'
                         .format(fname))
    data = xr.load_dataset(fname)

    # Entropy parameters for each spacecraft
    s_rel = xr.Dataset()
    for sc in ['mms1', 'mms2', 'mms3', 'mms4']:

        # Get the LUT
        lut_file = max_lut_filename(sc, mode, start_date, end_date, optdesc=optdesc)
        if not fname.exists():
            raise ValueError('Run load_max_lut first. File does not exist:\n{0}'
                             .format(lut_file))
        lut = xr.load_dataset(lut_file)

        #
        #  Measured parameters
        #

        # Measured distrubtion function
        #   - Energy affected in time-dependent manner by S/C pot correction
        f = max_lut_precond_f(sc, mode, optdesc, start_date, end_date)

        # Make sure they all have the same timestamps
        #   - Infinity is replaced by NaN. Put infinity back
        #   - NetCDF files do not like None values for attrs, so replace
        f = f.interp_like(data['time'], kwargs={'fill_value': 'extrapolate'})
        f['energy'][:,-1] = np.infty
        f.attrs['Upper_energy_integration_limit'] = np.infty
        
        # Moments and entropy parameters for the measured distribution
        n = fpi.density(f)
        V = fpi.velocity(f, N=n)
        T = fpi.temperature(f, N=n, V=V)
        t = ((T[:,0,0] + T[:,1,1] + T[:,2,2]) / 3.0).drop(['t_index_dim1', 't_index_dim2'])

        #
        #  Optimized equivalent Maxwellian parameters
        #
        
        # Equivalent Maxwellian distribution function
        opt_lut = max_lut_optimize(lut, f, n, t, method='nt')

        #
        #  Relative velocity space entropy
        #

        sV_rel_opt = physics.relative_entropy(f, opt_lut['f_M'])

        #
        # Save measured parameters
        #
        
        # Energy bins are corrected for S/C pot so are time and s/c dependent
        s_rel['f' + sc[3]] = f.rename({'phi': 'phi'+sc[3], 'energy': 'E'+sc[3], 'U': 'U'+sc[3]})
        s_rel['n' + sc[3]] = n
        s_rel['V' + sc[3]] = V
        s_rel['T' + sc[3]] = T
        s_rel['t' + sc[3]] = t
        s_rel['sV_rel' + sc[3]] = sV_rel_opt

        # Combine with optimized maxwellian parameters
        s_rel = xr.merge([s_rel,
                          opt_lut.rename({'f_M': 'f'+sc[3]+'_M',
                                          'phi': 'phi'+sc[3],
                                          'energy': 'E'+sc[3],
                                          'U': 'U'+sc[3],
                                          'n_M': 'n'+sc[3]+'_M',
                                          't_M': 't'+sc[3]+'_M',
                                          's_M': 's'+sc[3]+'_M',
                                          'sV_M': 'sV'+sc[3]+'_M'})])

    # Save as dataset
    fname = filename('mms', mode, start_date, end_date, optdesc=optdesc+'-s')
    s_rel.to_netcdf(fname)

    return fname


def load_max_lut(sc, mode, optdesc, start_date, end_date,
                 resolution=(100,100)):
    '''
    Create a Maxwellian Look-Up Table (LUT).

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
    
    Returns
    -------
    lut_file : path-like
        Path to the Maxwellian Look-Up Table
    '''
    
    instr = 'fpi'
    level = 'l2'

    lut_file = max_lut_filename(sc, mode, start_date, end_date,
                                resolution=resolution)
    if lut_file.exists():
        return lut_file

    # Precondition the distribution function
    f = max_lut_precond_f(sc, mode, optdesc, start_date, end_date)

    # Create the look-up table
    lut_file = physics.maxwellian_lut(f, filename=lut_file, dims=resolution)
    
    return lut_file


def max_lut_filename(*args, optdesc='des-dist', resolution=(100,100)):
    '''
    Create a filename for database files.

    Parameters
    ----------
    mode : str
        Data rate mode (srvy, brst)
    t0, t1 : `datetime.datetime`
        Start and end of the time of the data interval
    optdesc : str
        Optional filename descriptor
    
    Returns
    -------
    file_path : path-like
        Full file path.
    '''
    res_str = '{0}x{1}'.format(*resolution)
    lut_optdesc = optdesc + '-lut-' + res_str
    
    return filename(*args, optdesc=lut_optdesc)


def max_lut_precond_f(sc, mode, optdesc, start_date, end_date):
    '''
    Create a Maxwellian Look-Up Table (LUT).

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
    
    Returns
    -------
    lut_file : path-like
        Path to the Maxwellian Look-Up Table
    '''
    
    instr = 'fpi'
    level = 'l2'
    
    # Read the data
    #   - This includes removal of the photo-electron model
    fpi_dist = fpi.load_dist(sc=sc, mode=mode, optdesc=optdesc,
                            start_date=start_date, end_date=end_date)

    # Precondition the distributions
    #   - Wrap phi, extrapolate theta and E, remove s/c potential
    fpi_kwargs = fpi.precond_params(sc, mode, level, optdesc,
                                    start_date, end_date,
                                    time=fpi_dist['time'])
    f = fpi.precondition(fpi_dist['dist'], **fpi_kwargs)

    return f


def max_lut_optimize(lut, f, n, t, method='nt'):
    '''
    Determine the optimal Maxwellian distribution that best matches
    the density and temperature of the measured distribution function.

    Parameters
    ----------
    lut : `xarray.Dataset`
        Maxwellian look-up table
    n : (N,), array-like
        Number density calculated from the measured distribution
    t : (N,), array-like
        Scalar temperature calculated from the measured distribution
    f : (N,E,T,P), array-like
        Measured distribution function
    method : str
        Method of optimization:
            * 'n' : Minimize least square error (LSE) in density error
            * 't' : Minimize LSE in scalar temperature
            * 'nt' : Minimize LSE in both density and scalar temperature
            * 'interp': Interplate the density and temperature grid of the
                        look-up table onto the measured density and
                        temperature values
    
    Returns
    -------
    optimized_lut : `xarray.Dataset`
        Equivalent maxwellian distribution and associated density, temperature,
        entropy, and velocity-space entropy
    '''
    
    dims = (100, 100)
    
    n_M = np.zeros_like(n)
    t_M = np.zeros_like(t)
    s_M = np.zeros_like(n)
    sV_M = np.zeros_like(n)
    f_M = np.zeros_like(f)
    
    # Minimize density error
    if method == 'n':
        for idx, dens in enumerate(n):
            imin = np.argmin(np.abs(lut['N'].data - dens.item()))
            irow = imin // dims[1]
            icol = imin % dims[1]
            n_M[idx] = lut['N'][irow, icol]
            t_M[idx] = lut['t'][irow, icol]
            s_M[idx] = lut['s'][irow, icol]
            sV_M[idx] = lut['sv'][irow, icol]
            f_M[idx,...] = lut['f'][irow, icol, ...]

    # Minimize temperature error
    elif method == 't':
        for idx, temp in enumerate(t):
                imin = np.argmin(np.abs(lut['t'].data - temp.item()))
                irow = imin // dims[1]
                icol = imin % dims[1]
                n_M[idx] = lut['N'][irow, icol]
                t_M[idx] = lut['t'][irow, icol]
                s_M[idx] = lut['s'][irow, icol]
                sV_M[idx] = lut['sv'][irow, icol]
                f_M[idx,...] = lut['f'][irow, icol, ...]

    # Minimize error in both density and temperature
    elif method == 'nt':
        for idx, (dens, temp) in enumerate(zip(n, t)):
            imin = np.argmin(np.sqrt((lut['t'].data - temp.item())**2
                                     + (lut['N'].data - dens.item())**2
                                     ))
            irow = imin // dims[1]
            icol = imin % dims[1]
            n_M[idx] = lut['N'][irow, icol]
            t_M[idx] = lut['t'][irow, icol]
            s_M[idx] = lut['s'][irow, icol]
            sV_M[idx] = lut['sv'][irow, icol]
            f_M[idx,...] = lut['f'][irow, icol, ...]
    
    # Interpolate
    elif method == 'interp':
        lut_interp = lut.interp({'N_data': n, 't_data': t}, method='linear')
        n_M = lut_interp['N']
        t_M = lut_interp['t']
        s_M = lut_interp['s']
        sV_M = lut_interp['sv']
        f_M = lut_interp['f']
    
    return xr.Dataset({'time': ('time', n['time'].data),
                       'phi': (('time', 'phi_index'), f['phi'].data),
                       'theta': ('theta', f['theta'].data),
                       'energy': (('time', 'energy_index'), f['energy'].data),
                       'U': (('time', 'energy_index'), f['U'].data),
                       'n_M': ('time', n_M),
                       't_M': ('time', t_M),
                       's_M': ('time', s_M),
                       'sV_M': ('time', sV_M),
                       'f_M': (('time', 'phi_index', 'theta', 'energy_index'), f_M)
                       }).set_coords(['phi', 'energy', 'U'])
