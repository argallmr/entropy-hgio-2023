import datetime as dt
import xarray as xr
import numpy as np
from pathlib import Path

from pymms.data import util, fgm, edp, fpi
from pymms import config

import tools, physics

lut_dir = Path('~/data/fpi_fmax_lookup_tables/').expanduser()
output_dir = Path(config['dropbox_root'])

def load_data(t0, t1, dt_out=np.timedelta64(30, 'ms')):
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
    dt_out, `numpy.timedelta64`
        Sample interval to which all data is resampled. Default is the 30ms
        sample interval of DES
    
    Output
    ------
    filepath : `pathlib.Path`
        Path to netCDF file containing the data.
    '''

    # Get the data from each spacecraft
    mec_data = get_data('mec', t0, t1, dt_out)
    fgm_data = get_data('fgm', t0, t1, dt_out)
    edp_data = get_data('edp', t0, t1, dt_out)
    des_data = get_data('des', t0, t1, dt_out)
    dis_data = get_data('dis', t0, t1, dt_out)

    # Combine into a single dataset
    data = xr.merge([mec_data, fgm_data, edp_data, des_data, dis_data])

    # Create a file name
    filename = '_'.join(('mms', 'hgio',
                         t0.strftime('%Y%m%d%_H%M%S'),
                         t1.strftime('%Y%m%d%_H%M%S')))
    filepath = (output_dir / filename).with_suffix('.nc')

    # Save to data file
    data.to_netcdf(filepath)

    return filepath


def get_data(instr, t0, t1, dt_out=np.timedelta64(30, 'ms')):
    '''
    Get data from a single instrument. Data coordinates and time cadence
    are standardized.

    Parameters
    ----------
    instr : str
        Name of the instrument for which to load data
        ('fgm', 'edp', 'des', 'dis', 'mec')
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
    else:
        raise ValueError('Instrument {0} not recognized'.format(instr))
    
    # Download the data from each spacecraft
    spacecraft = ['mms1', 'mms2', 'mms3', 'mms4']

    # A place to store the data from each spacecraft
    ds = xr.Dataset()

    # Get the data from each spacecraft
    for sc in spacecraft:
        # Load the data
        data = func(sc, t0, t1)

        # Resample the data to a common time stamp
        data = tools.resample(data, t0, t1, dt_out,
                              extrapolate=extrapolate, method=method)

        # Store the data in the database
        try:
            ds[data.name] = data
        except AttributeError:
            ds = xr.merge([ds, data])
    
    return ds


def get_mec_data(sc, t0, t1):
    '''
    Get MEC data for a given spacecraft and time interval.

    Parameters
    ----------
    sc : str
        The spacecraft for which data is to be loaded
        ('mms1', 'mms2', 'mms3', 'mms4')
    t0, t1 : `datetime.datetime`
        Start and end of the time interval to be loaded
    
    Output
    ------
    r_data : `xarray.Dataset`
        The spacecraft position in GSE coordinates
    '''

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
    r_data = (r_data[sc+'_mec_r_gse']
              .rename({'Epoch': 'time',
                       sc+'_mec_r_gse_label': 'component'})
              .assign_coords({'component': ['x', 'y', 'z'],
                              'dt_plus': np.timedelta64(int(30e9), 'ns')})
              )

    # Name with the spacecraft number to make it unique
    r_data.name = 'r' + sc[-1]

    return r_data

def get_fgm_data(sc, t0, t1):
    '''
    Get FGM data for a given spacecraft and time interval.

    Parameters
    ----------
    sc : str
        The spacecraft for which data is to be loaded
        ('mms1', 'mms2', 'mms3', 'mms4')
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
    b_data = fgm.load_data(sc=sc, mode='brst', start_date=t0, end_date=t1)

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


def get_edp_data(sc, t0, t1):
    '''
    Get EDP data for a given spacecraft and time interval.

    Parameters
    ----------
    sc : str
        The spacecraft for which data is to be loaded
        ('mms1', 'mms2', 'mms3', 'mms4')
    t0, t1 : `datetime.datetime`
        Start and end of the time interval to be loaded
    
    Output
    ------
    e_data : `xarray.Dataset`
        The vector electric field in GSE coordinates
    '''

    # EDP
    #   - data rate is 32 S/s in survey mode and 4098 S/s in burst mode.
    #   - Since survey mode sampling rate is closer to the FGM burst mode sample rate, we will load EDP survey data.
    e_data = edp.load_data(sc=sc, mode='srvy', start_date=t0, end_date=t1)

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


def get_des_data(sc, t0, t1):
    '''
    Get DES data for a given spacecraft and time interval.

    Parameters
    ----------
    sc : str
        The spacecraft for which data is to be loaded
        ('mms1', 'mms2', 'mms3', 'mms4')
    t0, t1 : `datetime.datetime`
        Start and end of the time interval to be loaded
    
    Output
    ------
    des_data : `xarray.Dataset`
        The velocity, scalar pressure, and pressure tensor
    '''

    # DES
    des_data = fpi.load_moms(sc='mms1', mode='brst', optdesc='des-moms',
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


def get_dis_data(sc, t0, t1):
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
    dis_data = fpi.load_moms(sc='mms1', mode='brst', optdesc='dis-moms',
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


def max_lut_filename(sc, instr, mode, level, optdesc, start_date, end_date,
                     resolution=(100, 100)):
    '''
    Create a file name for the Maxwellian Look-Up Table

    Parameters
    ----------
    sc : str
        MMS spacecraft identifier ('mms1', 'mms2', 'mms3', 'mms4')
    instr : str
        Instrument short name ('fpi',)
    mode : str
        Operating mode ('brst', 'srvy', 'fast')
    level : str
        Data level ('l2,')
    optdesc : str
        Filename optional descriptor ('dis-dist', 'des-dist')
    start_date, end_date : `datetime.datetime`
        Start and end of the time interval
    
    Returns
    -------
    lut_file : path-like
        Path to the Maxwellian Look-Up Table
    '''
    res_str = '{0}x{1}'.format(*resolution)
    
    # Create a file name for the look-up table
    lut_file = lut_dir / '_'.join((sc, instr, mode, level,
                                   optdesc+'-lut-'+res_str,
                                   start_date.strftime('%Y%m%d_%H%M%S'),
                                   end_date.strftime('%Y%m%d_%H%M%S')))
    lut_file = lut_file.with_suffix('.ncdf')

    return lut_file


def max_lut_load(sc, mode, optdesc, start_date, end_date):
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

    lut_file = max_lut_filename(sc, instr, mode, level, optdesc,
                                start_date, end_date)

    # If the LUT does not exist, create it
    if not lut_file.exists():
        # Precondition the distribution function
        f = max_lut_precond_f(sc, mode, optdesc, start_date, end_date)

        # Create the look-up table
        lut_file = physics.maxwellian_lut(f, filename=lut_file)
    
    return lut_file


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