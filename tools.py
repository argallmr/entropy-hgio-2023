import numpy as np
import xarray as xr
from scipy.stats import binned_statistic
from warnings import warn


def barycentric_avg(v):
    '''
    Calculate the barycentric average

    Parameters
    ----------
    v : `xarray.Dataset`
        Measurements taken at each vertex of a tetrahedron
    
    Returns
    -------
    v_bary : `xarray.DataArray`
        Barycentric average of the measurements
    '''
    v_bary = 0
    for v_name in v:
        v_bary += v[v_name]
    
    v_bary /= len(v)

    return v_bary


def binned_avg_ds(ds, t_out):
    
    def ds_bin_avg(data_to_avg):
        avg, bin_edges, binnum = binned_statistic(t_in, data_to_avg,
                                                  statistic='mean',
                                                  bins=t_bins)
        return avg
    
    vars_out = {}
    
    # scipy does not like datetime64 so time has to be converted to floats
    #   - Subtracting lots of datetime64 can take a long time! (???)
    t_ref =  np.min([ds['time'][0].data.astype('datetime64[D]'),
                     t_out[0].astype('datetime64[D]')])
    t_bins = (t_out - t_ref).astype('float')
    t_in = (ds['time'].data - t_ref).astype('float')
    
    # Step through each variable in the dataset
    for name, var in ds.data_vars.items():

        # Remember what the variable looked like
        var_shape = (len(t_bins)-1, *var.shape[1:])
        var_ndim = var.ndim
        var_dims = var.dims
        var_dtype = var.dtype
        
        # Reshape the variable to be 2D
        if var_ndim == 1:
            var = var.expand_dims('temp', axis=1)
        elif var_ndim > 2:
            var = var.stack(temp=var_dims[1:])
        
        # scipy does not like datetimes/timedelta of any type so convert to float
        if np.issubdtype(var_dtype, np.timedelta64):
            var = var.data.astype(float)

        # Bin-average each component of the vector
        temp_vars = []
        for idx in range(var.shape[1]):
            temp_vars.append(ds_bin_avg(var.data[:,idx]))
        
        # Reshape the data back to how it was
        temp_vars = np.column_stack(temp_vars)
        if var_ndim == 1:
            temp_vars = temp_vars.squeeze()
        elif var_ndim > 2:
            temp_vars = temp_vars.reshape(var_shape)
            var = var.unstack()
        
        # Convert back to datetime/timedelta
        if np.issubdtype(var_dtype, np.timedelta64):
            temp_vars = temp_vars.astype(np.timedelta64)

        # Combine bin-averaged components into an array
        temp = xr.DataArray(temp_vars,
                            dims=var_dims,
                            coords={key: value if key != var_dims[0] else t_out[:-1]
                                    for key, value in var.coords.items()},
                            name=name)
        
        # Save the variables in the output dictionary
        vars_out[name] = temp
    
    # Create a new dataset
    return xr.Dataset(vars_out)
 

def binned_avg(data, t_out):
    '''
    Resample data by averaging into temporal bins
    
    Parameters
    ----------
    data : `numpy.array`
        Data to be averaged
    t_out : `numpy.datetime64`
        Leading edge of bins into which data should be averaged
    
    Returns
    -------
    avg : `xarray.DataArray`
        Resampled data
    '''
    
    # scipy does not like datetime64 so time has to be converted to floats
    t_ref =  np.min([data['time'][0].data.astype('datetime64[D]'),
                     t_out[0].astype('datetime64[D]')])
    t_bins = (t_out - t_ref).astype('float')
    t_in = (data['time'].data - t_ref).astype('float')

    # Calculate the average
    def bin_avg(data_to_avg):
        avg, _, _ = binned_statistic(t_in, data_to_avg,
                                     statistic='mean',
                                     bins=t_bins)
        return avg
    
    # A time series
    if data.ndim == 1:
        if np.issubdtype(data.dtype, np.timedelta64):
            temp = bin_avg(data.data.astype(float))
            temp = temp.astype(np.timedelta64)
        else:
            temp = bin_avg(data.data)
        
        # The final time in t_out is the right-most bin edge of the final bin. Remove it.
        temp = xr.DataArray(temp,
                            dims=('time',),
                            coords={'time': t_out[:-1]},
                            name=data.name)
    
    # A vector of time series
    elif data.ndim == 2:
        temp_vars = []
        
        # Bin each component of the vector
        for idx in range(data.shape[1]):
            temp_vars.append(bin_avg(data.data[:,idx]))
        
        # Combine bin-averaged components into an array
        # The final time in t_out is the right-most bin edge of the final bin. Remove it.
        temp = xr.DataArray(np.column_stack(temp_vars),
                            dims=('time', data.dims[1]),
                            coords={'time': t_out[:-1],
                                    data.dims[1]: data.coords[data.dims[1]]},
                            name=data.name)

    else:
        warn('Variable {0} has > 2 dimensions. Skipping'
                .format(data.name))

    return temp


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


def expand_times(da, t_out):
    
    # Locate the data times within the packet times by using the packet times
    # as bin edges
    #   - scipy does not like datetime64 so time has to be converted to floats
    #   - Subtracting lots of datetime64 can take a long time! (???)
    t0 =  np.min([da['time'][0].data.astype('datetime64[D]'),
                     t_out[0].astype('datetime64[D]')])
    dt_out = (t_out - t0).astype('float')
    dt_in = (da['time'].data - t0).astype('float')
    
    # Bin the data. The key here will be the bin number
    cnt, bin_edges, binnum = binned_statistic(dt_out, dt_out,
                                              statistic='count',
                                              bins=dt_in)

    # Test bins
    #   - Bin 0 is for data before the first packet time
    #   - Bin len(t_out) is for all data past the last packet time. There
    #     should be `sample_rate` points after the last packet time. If
    #     there are more, there is a problem: TODO - test
    if binnum[0] == 0:
        raise ValueError('Data times start before output times.')
    
    # Expand the input data
    result = da[binnum-1].assign_coords({'time': t_out})
    
    return result


def generate_time_stamps(t_start, t_stop, t_res=np.timedelta64(30, 'ms')):
    '''
    Create an array of times spanning a time interval with a specified resolution.
    
    Parameters
    ----------
    t_start, t_stop : `numpy.datetime64`
        Start and end of the time interval, given as the begin times of the first
        and last samples
    t_res : `numpy.timedelta64`
        Sample interval
    
    Returns
    -------
    t_stamps : `numpy.datetime64`
        Timestamps spanning the given time interval and with the given resolution. Note
        that the last point in the array is the end time of the last sample. This is to
        work better with `scipy.binned_statistic`.
    '''
    # Find the start time
    t_ref = t_start.astype('datetime64[m]').astype('datetime64[ns]')
    t_start = t_start - ((t_start - t_ref) % t_res)

    # Find the end time -- it should be after the final time
    t_ref = t_stop.astype('datetime64[m]').astype('datetime64[ns]')
    dt_round = (t_stop - t_ref) % t_res
    t_stop += (t_res - dt_round)

    # Time at five second intervals.
    #  - We want t_end to be included in the array as the right-most edge of the time interval
    t_stamps = np.arange(t_start, t_stop + t_res, t_res)
    
    return t_stamps


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


def interp(ds, t_out, method='linear', extrapolate=False):
    if extrapolate:
        kwargs = {'fill_value': 'extrapolate'}
    else:
        kwargs = None
    
    return ds.interp(time=t_out, method=method, kwargs=kwargs)


def interp_over_gaps(data, t_out, extrapolate=False):
    '''
    Interpolate data being careful to avoid interpolating over data gaps
    
    Parameters
    ----------
    t_out : `numpy.datetime64`
        Times to which the data should be interpolated
    data : `xarray.DataArray`
        Data to be interpolated with coordinate 'time'
    t_delta : `numpy.timedelta64`
        Sampling interval of the data
    
    Returns
    -------
    result : `xarray.DataArray`
        Data interpolated to the given timestamps via nearest neighbor averaging
    '''
    if extrapolate:
        kwargs = {'fill_value': 'extrapolate'}
    else:
        kwargs = None

    # Data gaps are any time interval larger than a sampling interval
    N_dt = np.round(data['time'].diff(dim='time') / data['dt_plus']).astype(int)

    # Find the last sample in each contiguous segment
    #   - The last data point in the array is the last sample in the last contiguous segment
    idx_gaps = np.argwhere(N_dt.data > 1)[:,0]
    idx_gaps = np.append(idx_gaps, len(data['time']))

    # Number of data gaps
    N_gaps = len(idx_gaps)
    if (N_gaps - 1) > 0:
        warn('{0} data gaps found in dataset. Gap size: {1}.'
             .format(N_gaps - 1, N_dt[idx_gaps[:-1]]))

    # Interpolate each contiguous segment
    temp = []
    istart = 0
    for igap in idx_gaps:
        # If no extrapolation, the first and last data points turn into NaNs
        # or are dropped because one of the nearest neighbors is beyond the
        # data interval
        #   - d_in:     *   *   *   *   *
        #   - t_in:     *   *   *   *   *
        #   - t_out:  *   *   *   *   *   *
        #   - d_out: NaN  *   *   *   *  NaN
        #   - len(data) = len(t_in) - 1 for each gap
        temp.append(data[dict(time=slice(istart, igap+1))]
                    .interp({'time': t_out}, method='nearest', kwargs=kwargs)
                    .dropna(dim='time')
                    )
        istart = igap+1

    # Combine each contiguous segment into a single array
    return xr.concat(temp, dim='time')


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


def resample(data, t0, t1, dt_out, extrapolate=False, method='nearest'):

    # Generate a set of timestamps
    #   - The end time will be the next sample after t1
    #   - For binned averages, we want the closed interval [t_out[0], t_out[-1]]
    #   - For all other resample methods, we want the half-open interval [t_out[0], t_out[-1])
    t_out = generate_time_stamps(np.datetime64(t0),
                                 np.datetime64(t1),
                                 np.timedelta64(dt_out))

    # Align the time
    #   - Same sample rate but different time stamps
    #   - Assumes sample time is at beginning of sample interval (dt_minus = 0)
    #   - Want to bring the ratio to 1
    dt_ratio = np.timedelta64(dt_out) / data['dt_plus']
    
    # Upsample / Interpolate
    #   - Less than two samples per target sampling interval
    if dt_ratio <= 2:
        
        # Repeat values
        if method == 'repeat':
            data = expand_times(data, t_out[:-1])
        
        elif method in ('nearest', 'linear'):
            data = data.interp({'time': t_out[:-1]}, method=method)

        # Interpolate
        elif method == 'interp_gaps':
            data = interp_over_gaps(data, t_out[:-1],
                                    extrapolate=extrapolate)
        
        else:
            raise ValueError('Method {0} not recognized. Pick one of '
                             '(repeat, nearest, linear, interp_gaps).'
                             .format(method))

    # Downsample
    #   - Two or more samples per target sampling interval
    elif dt_ratio > 2:
        if isinstance(data, xr.DataArray):
            func = binned_avg
        else:
            func = binned_avg_ds
        
        # Simple average
        data = func(data, t_out)

    # Suppress annoying xarray warning about converting to ns precision by doing the conversion ourselves
    return data.assign_coords({'dt_plus': dt_out.astype('timedelta64[ns]')})


def smooth(data, n, step=1, fs=None, weights=1, end_points=None):
    '''
    Smooth an array.

    Parameters
    ----------
    data : (N,...), `xarray.DataArray`
        Data to be smoothed. Smoothing is performed along the "time" dimension
    n : int or float
        Number of points to smooth. If a float, this is the time interval over
        which to smooth
    step : int
        Number of points to skip between smoothing windows
    fs : float
        Sample rate. Used if `n` is a time interval.
    end_points : str
        How to treat end points. (None, "copy", "wrap", "repeat")
          * None: Skip over end points (output is 0)
          * copy: Copy points from the input array
          * wrap: Wrap the smoothing window to the other end of the array
          * repeat: Repeat the first or last data point
    
    Returns
    -------
    out : (N, ...), `xarray.DataArray`
        The smoothed input array
    '''
    if isinstance(n, float) or isinstance(step, float):
        # Calculate the sample rate
        if fs is None:
            fs = 1.0 / (np.diff(data['time']).mean().astype(float) * 1e-9)
    
    # Number of points in the smoothing window is given as a time interval
    if isinstance(n, float):
        n = np.int64(np.round(n * fs))

    # Number of points to shift is given as a time interval
    if isinstance(step, float):
        step = np.int64(np.round(step * fs))
    
    # Need to average at least two points
    if n < 2:
        raise ValueError('Too few puts (n={0}). Must be at least 2.'.format(n))
    
    # Need to shift by at least one point
    if step < 1:
        raise ValueError('Too few points to shift (step={0}). '
                         'Must be at least 1.'.format(step))
    
    # Weights must be scalar or an array the same length as data
    if np.isscalar(weights):
        weights = np.repeat(weights, n)
    if len(weights) != n:
        raise ValueError('Weights must be scalar or have length n.')
    if data.ndim > 1:
        weights = np.expand_dims(weights, np.arange(1,data.ndim).tolist())
    
    # Number of windows that fit within the data
    N = len(data)
    N_out = int(np.floor((N-n)/step + 1))

    # Allocate memory to the output array
    out = np.zeros((N_out, *data.shape[1:]), dtype=data.dtype)
    t_out = np.zeros(N_out, dtype='datetime64[ns]')

    # Initial loop conditions
    i0 = 0
    i1 = n
    # idx = np.int64((i0 + i1) / 2)
    idx = 0
    while i1 <= N:
        out[idx,...] = np.sum(weights * data[i0:i1,...], axis=0) / np.sum(weights, axis=0)
        t_out[idx] = data['time'][i0:i1].mean(dim='time').item()

        i0 += step
        i1 += step
        idx += 1
    
    # Turn into a DataArray
    out = (xr.DataArray(out,
                        dims=('time', *data.dims[1:]),
                        coords={key: data[key] for key in data.coords if key != 'time'})
           .assign_coords({'time': t_out})
           )
    
    '''
    imid0 = np.int64(n / 2)
    imid1 = N - (n - imid0)
    i0 = -imid0
    i1 = i0 + n
    for ii in range(N):

        # Skip the end points (copied later)
        if end_points in (None, 'copy'):
            if (i0 >= 0) & (i1 < N):
                idx = slice(i0, i1)
            else:
                i0 += step
                i1 += step
                continue
        
        # Repeat the first/last point in the array
        elif end_points == 'repeat':
            if i0 < 0:
                idx = np.append(np.repeat(0, -i0), np.arange(0, i1))
            elif i1 >= N:
                idx = np.append(np.arange(i0, N), np.repeat(-1, i1-N))
            else:
                idx = slice(i0, i1)
        
        # Wrap around to the beginning of the array
        elif end_points == 'wrap':
            if i0 < 0:
                idx = np.append(np.arange(i0, 0), np.arange(i1))
            elif i1 >= N:
                print(i0, N, 0, i1-N)
                idx = np.append(np.arange(i0, N), np.arange(0, i1-N))
            else:
                idx = slice(i0, i1)
        
        else:
            raise ValueError('end_points must be in (None, "repeat", "wrap", "copy").')

        out[ii,...] = data[idx,...].mean(dim='time')
        i0 += step
        i1 += step
    
    # Copy values to end-points
    if end_points == 'copy':
        out[0:imid0,...] = data[0:imid0,...]
        out[imid1:,...] = data[imid1:,...]
    '''

    return out