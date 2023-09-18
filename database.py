import datetime as dt
import xarray as xr
import numpy as np
from pymms.data import util, fgm, edp, fpi
import tools


def load_data(t0, t1, dt_out=dt.timedelta(milliseconds=30)):

    # Get the data from each spacecraft
    mec_data = get_data('mec', t0, t1, dt_out)
    fgm_data = get_data('fgm', t0, t1, dt_out)
    edp_data = get_data('edp', t0, t1, dt_out)
    des_data = get_data('des', t0, t1, dt_out)

    # Combine into a single dataset
    data = xr.merge([mec_data, fgm_data, edp_data, des_data])


def get_data(instr, t0, t1, dt_out=np.timedelta64(30, 'ms')):

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

    # FGM
    #   - sampled at 4 S/s in survey mode and 128 S/s in burst mode.
    #   - Select burst mode for its higher time resolution.
    b_data = fgm.load_data(sc=sc, mode='brst', start_date=t0, end_date=t1)

    # Select the magnetic field in GSE coordinates
    #   - Remove the total magnetic field
    #   - NOTE: These are the center times
    #   - NOTE: in Survey mode there can be a mix of sample rates (fast/slow)
    b_data = (b_data['B_GSE'][:,0:3]
              .rename({'b_index': 'component'})
              .assign_coords({'dt_plus': (b_data['time_delta'].mean() * 1e9).astype('timedelta64[ns]')})
              )

    # Name with the spacecraft number to make it unique
    b_data.name = 'B' + sc[-1]

    return b_data


def get_edp_data(sc, t0, t1):

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

    # Name with the spacecraft number to make it unique
    e_data.name = 'E' + sc[-1]
    
    return e_data


def get_des_data(sc, t0, t1):

    # DES
    des_data = fpi.load_moms(sc='mms1', mode='brst', optdesc='des-moms', start_date=t0, end_date=t1)

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
              .rename({'cart_index_dim1': 'comp_1', 'cart_index_dim2': 'comp_2'})
              )
    
    # Combine each into a single dataset
    return xr.Dataset({'V'+sc[-1]: v_data,
                       'p'+sc[-1]: p_data,
                       'P'+sc[-1]: P_data}).assign_coords({'dt_plus': des_data['Epoch_plus_var'].data})
