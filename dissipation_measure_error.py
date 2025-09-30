#sys.path.insert(0, '/Users/krmhanieh/pymms/pymms')
#os.chdir('/Users/krmhanieh/entropy-hgio-2023')
import database as db, plots


import plots, maxlut, physics, database, tools
from maxlut import Lookup_Table


import os
import sys
import re
import warnings
import datetime as dt

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt, dates as mdates


#sys.path.insert(0, '/Users/krmhanieh/pymms/pymms')
#os.chdir('/Users/krmhanieh/entropy-hgio-2023')
import database as db


from pymms import config
from pymms.data import fpi
from scipy import constants

import plots, maxlut, physics, database, tools
from maxlut import Lookup_Table


def format_axes(ax, xaxis='on'):
    """Minimal formatting helper."""
    ax.grid(True, which='major', alpha=0.3)
    ax.grid(True, which='minor', alpha=0.15)
    try:
        ax.minorticks_on()
    except Exception:
        pass
    if str(xaxis).lower() == 'off':
        ax.tick_params(axis='x', which='both', labelbottom=False)

def get_des_errors(sc, mode, t0, t1):
    """Return UNSMOOOTHED error arrays for electrons for one spacecraft."""
    moms = fpi.load_moms(sc=sc, mode=mode, optdesc='des-moms',
                         start_date=t0, end_date=t1, center_times=False)
    v_err_name = next((n for n in moms.data_vars if 'bulkv_err' in n), None)
    if v_err_name is None: raise KeyError(f"No bulkv_err variable for electrons in {sc} {mode}")
    v_err = moms[v_err_name]
    comp_dim0 = [d for d in v_err.dims if d != 'time'][0]
    lab_name = next((c for c in moms.coords if 'bulkv_err_label' in c), None)
    comp_labels = [str(s)[1].lower() if str(s).startswith('V') else str(s) for s in moms[lab_name].values] if lab_name else ['x','y','z']
    v_err = v_err.rename({comp_dim0: 'component'}).assign_coords(component=comp_labels)
    
    p_err_name = next((n for n in moms.data_vars if 'prestensor_err' in n), None)
    if p_err_name is None: raise KeyError(f"No prestensor_err variable for electrons in {sc} {mode}")
    p_err = moms[p_err_name]
    d1, d2 = [d for d in p_err.dims if d != 'time']
    p_err = p_err.rename({d1: 'comp1', d2: 'comp2'}).assign_coords(comp1=['x','y','z'], comp2=['x','y','z'])
    
    scn = sc[-1]
    return xr.Dataset({f'Ve{scn}_err': v_err, f'Pe{scn}_err': p_err})

def _coerce_vec3(da, new_dim_name='component'):
    comp_dim = next((d for d in da.dims[::-1] if da.sizes[d] == 3), None)
    if comp_dim is None:
        raise ValueError(f"Expected a 3-component vector, got dims {da.dims}")
    if comp_dim != new_dim_name:
        da = da.rename({comp_dim: new_dim_name})
    if new_dim_name not in da.coords or len(da[new_dim_name]) != 3:
        da = da.assign_coords({new_dim_name: ['x','y','z']})
    return da

def _coerce_tensor(da, new_dims=('comp1', 'comp2')):
    other_dims = [d for d in da.dims if d != 'time']
    if len(other_dims) != 2:
        raise ValueError(f"Expected a 3x3 tensor with 2 non-time dims, but got dims {da.dims}")
    rename_map = {}
    if other_dims[0] != new_dims[0]: rename_map[other_dims[0]] = new_dims[0]
    if other_dims[1] != new_dims[1]: rename_map[other_dims[1]] = new_dims[1]
    da_renamed = da.rename(rename_map) if rename_map else da
    for dim_name in new_dims:
        if dim_name not in da_renamed.coords or len(da_renamed[dim_name]) != 3:
            da_renamed = da_renamed.assign_coords({dim_name: ['x', 'y', 'z']})
    return da_renamed

def _resolve_sc4_positions(data):
    r_gse = [f"r{i}_GSE" for i in range(1, 5)]
    if all(name in data for name in r_gse):
        return xr.Dataset({f"r{i}": _coerce_vec3(data[f"r{i}_GSE"]) for i in range(1, 5)})
    r_plain = [f"r{i}" for i in range(1, 5)]
    if all(name in data for name in r_plain):
        return xr.Dataset({f"r{i}": _coerce_vec3(data[f"r{i}"]) for i in range(1, 5)})
    name_re = re.compile(r"^(r|pos|position)([_\\-]?(gse|gsm))?$", re.IGNORECASE)
    for vname, da in data.data_vars.items():
        if name_re.match(vname):
            try:
                sc_dim = next(d for d in da.dims if da.sizes[d] == 4)
                return xr.Dataset({f"r{i+1}": _coerce_vec3(da.isel({sc_dim: i})) for i in range(4)})
            except (StopIteration, ValueError):
                continue
    raise KeyError("Could not resolve positions r1..r4.")

def _reciprocal_vectors(R):
    if 'time' not in R.dims: R = R.expand_dims(time=[0])
    idx = [(0,1,2,3), (1,2,3,0), (2,3,0,1), (3,0,1,2)]
    Rv = R.transpose('sc', 'time', 'comp').values
    k = np.zeros_like(Rv)
    for a,(α,β,γ,λ) in enumerate(idx):
        r_ab, r_ag, r_al = Rv[β]-Rv[α], Rv[γ]-Rv[α], Rv[λ]-Rv[α]
        r_bg, r_bl = Rv[γ]-Rv[β], Rv[λ]-Rv[β]
        num = np.cross(r_bg, r_bl)
        den = np.einsum('ti,ti->t', r_ab, np.cross(r_ag, r_al))
        den[den == 0] = np.nan
        k[a] = (num.T / den).T
    return xr.DataArray(k, coords=R.coords, dims=R.dims)

def p_theta_error(R, U, U_err, p, P, P_err):
    k = _reciprocal_vectors(R)
    theta = (k * U).sum(['sc', 'comp'])
    sigma_theta = np.sqrt(((k**2) * (U_err**2)).sum(['sc', 'comp']))
    sigma_P_av = 0.25 * np.sqrt((P_err**2).sum('sc'))
    diag = [sigma_P_av.sel(comp1=i, comp2=i) for i in sigma_P_av.coords['comp1'].values]
    sigma_p = (1/3) * np.sqrt(sum(d**2 for d in diag))
    P_av = P.mean('sc')
    p_av = (1/3) * xr.apply_ufunc(np.trace, P_av, input_core_dims=[['comp1','comp2']], vectorize=True)
    return np.sqrt((theta**2)*(sigma_p**2) + (p_av**2)*(sigma_theta**2))

def PiD_error(R, U, U_err, p, P, P_err, D):
    k = _reciprocal_vectors(R)
    def _sigma_duj_dxi(k, U_err, i, j):
        comp_vals = k.coords['comp'].values
        term = k.sel(comp=comp_vals[i])**2 * U_err.sel(comp=comp_vals[j])**2
        return np.sqrt(term.sum('sc'))
    comps, T = list(k.coords['comp'].values), k.sizes['time']
    sig_D = np.zeros((T,3,3))
    for i in range(3):
        for j in range(i + 1, 3):
            s1, s2 = _sigma_duj_dxi(k, U_err, i, j), _sigma_duj_dxi(k, U_err, j, i)
            val = 0.5 * np.sqrt(s1**2 + s2**2)
            sig_D[:, i, j] = sig_D[:, j, i] = val.values
    sxx, syy, szz = [_sigma_duj_dxi(k, U_err, i, i) for i in range(3)]
    sig_D[:,0,0] = np.sqrt((4/9)*sxx**2 + (1/9)*syy**2 + (1/9)*szz**2).values
    sig_D[:,1,1] = np.sqrt((1/9)*sxx**2 + (4/9)*syy**2 + (1/9)*szz**2).values
    sig_D[:,2,2] = np.sqrt((1/9)*sxx**2 + (1/9)*syy**2 + (4/9)*szz**2).values
    sigma_D = xr.DataArray(sig_D, coords={'time':k['time'], 'comp1':comps, 'comp2':comps}, dims=('time','comp1','comp2'))
    sigma_P_av = 0.25 * np.sqrt((P_err**2).sum('sc'))
    sig_Pi = np.zeros((T,3,3))
    for i in range(3):
        for j in range(i + 1, 3):
            sig_Pi[:, i, j] = sig_Pi[:, j, i] = sigma_P_av.sel(comp1=comps[i], comp2=comps[j]).values
    sxx_p, syy_p, szz_p = [sigma_P_av.sel(comp1=c, comp2=c) for c in comps]
    sig_Pi[:,0,0] = np.sqrt((4/9)*sxx_p**2 + (1/9)*syy_p**2 + (1/9)*szz_p**2).values
    sig_Pi[:,1,1] = np.sqrt((1/9)*sxx_p**2 + (4/9)*syy_p**2 + (1/9)*szz_p**2).values
    sig_Pi[:,2,2] = np.sqrt((1/9)*sxx_p**2 + (1/9)*syy_p**2 + (4/9)*szz_p**2).values
    sigma_Pi = xr.DataArray(sig_Pi, coords={'time':sigma_P_av['time'], 'comp1':comps, 'comp2':comps}, dims=('time','comp1','comp2'))
    P_bary = P.mean('sc')
    p_bary = p.mean('sc')
    I = xr.DataArray(np.broadcast_to(np.eye(3), (P_bary.sizes['time'], 3, 3)), dims=('time','comp1','comp2'), coords=P_bary.coords)
    Pi = P_bary - p_bary * I
    sigma_comp = np.sqrt((Pi**2)*(sigma_D**2) + (D**2)*(sigma_Pi**2))
    return np.sqrt(sigma_comp.sel(comp1='x',comp2='x')**2 + sigma_comp.sel(comp1='y',comp2='y')**2 + sigma_comp.sel(comp1='z',comp2='z')**2 +
                   4 * (sigma_comp.sel(comp1='x',comp2='y')**2 + sigma_comp.sel(comp1='x',comp2='z')**2 + sigma_comp.sel(comp1='y',comp2='z')**2))

# Main Plotting Function
def plot_electron_data(t0, t1, mode='brst', verbose=True):
    """
    Calculates and plots electron data with a publication-style design.
    """
    # Load Data
    fname = database.load_data(t0, t1, mode=mode)
    data = xr.load_dataset(fname)
    
    try:
        R4 = _resolve_sc4_positions(data)
        data = xr.merge([data, R4])
    except KeyError as e:
        raise KeyError(f"Positions not found / normalized: {e}")

    data = data.assign_coords(component=['x','y','z'])
    
    ptheta_e = physics.pressure_dilatation(
        data[['r1','r2','r3','r4']], 
        data[['Ve1','Ve2','Ve3','Ve4']], 
        xr.Dataset({f'p{i}': data[f'pe{i}'] for i in range(1,5)})
    )
    PiD_e = physics.PiD(
        data[['r1','r2','r3','r4']], 
        data[['Ve1','Ve2','Ve3','Ve4']], 
        xr.Dataset({f'p{i}': data[f'pe{i}'] for i in range(1,5)}), 
        xr.Dataset({f'P{i}': data[f'Pe{i}'] for i in range(1,5)})
    )
    
    # Electron Error Calculation 
    sigma_p_theta_e, sigma_PiD_e = None, None
    try:
        sc_coord = xr.DataArray([1,2,3,4], dims='sc', name='sc')
        R_sc = xr.concat([_coerce_vec3(data[f'r{i}'], 'comp') for i in range(1,5)], dim=sc_coord)
        U_sc_e = xr.concat([_coerce_vec3(data[f'Ve{i}'], 'comp') for i in range(1,5)], dim=sc_coord)
        p_sc_e = xr.concat([data[f'pe{i}'] for i in range(1,5)], dim=sc_coord)
        P_sc_e = xr.concat([_coerce_tensor(data[f'Pe{i}']) for i in range(1,5)], dim=sc_coord)
        R_sc, U_sc_e, p_sc_e, P_sc_e = xr.align(R_sc, U_sc_e, p_sc_e, P_sc_e, join='inner')

        errs_e = xr.merge([get_des_errors(f'mms{i}', mode, t0, t1) for i in range(1, 5)])
        U_err_e = xr.concat([_coerce_vec3(errs_e[f'Ve{i}_err'], 'comp') for i in range(1,5)], dim=sc_coord)
        P_err_e = xr.concat([_coerce_tensor(errs_e[f'Pe{i}_err']) for i in range(1,5)], dim=sc_coord)
        
        U_err_sc_e = U_err_e.sortby('time').interp(time=U_sc_e.time, method='nearest')
        P_err_sc_e = P_err_e.sortby('time').interp(time=P_sc_e.time, method='nearest')

        sigma_p_theta_e = p_theta_error(R_sc, U_sc_e, U_err_sc_e, p_sc_e, P_sc_e, P_err_sc_e)
        D_val_e = physics.devoriak_pressure(
            data[['r1','r2','r3','r4']], data[['Ve1','Ve2','Ve3','Ve4']],
            xr.Dataset({f'p{i}': data[f'pe{i}'] for i in range(1,5)}), xr.Dataset({f'P{i}': data[f'Pe{i}'] for i in range(1,5)})
        )
        sigma_PiD_e = PiD_error(R_sc, U_sc_e, U_err_sc_e, p_sc_e, P_sc_e, P_err_sc_e, D=D_val_e)
    except Exception as exc:
        if verbose: print(f"[warn] Could not compute electron uncertainties: {exc}")

    # --- Plotting ---
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(9, 7), sharex=True)
    plt.subplots_adjust(hspace=0.1)

    # Panel (a): Magnetic Field
    ax_a = axes[0]
    B_avg = data[['B1','B2','B3','B4']].to_array(dim='sc').mean('sc')
    B_mag = np.linalg.norm(B_avg, axis=B_avg.get_axis_num('component'))
    ax_a.plot(B_avg.time.values, B_avg.sel(component='x').values, color='red', label='Bx')
    ax_a.plot(B_avg.time.values, B_avg.sel(component='y').values, color='green', label='By')
    ax_a.plot(B_avg.time.values, B_avg.sel(component='z').values, color='blue', label='Bz')
    ax_a.plot(B_avg.time.values, B_mag, color='black', label='|B|', lw=1.5)
    ax_a.set_ylabel('B (nT)')
    ax_a.legend(loc='upper right', frameon=False, ncol=4)
    format_axes(ax_a, xaxis='off')

    
    ax_b = axes[1]
    plot_val = -ptheta_e
    
    
    ax_b.plot(plot_val.time.values, plot_val.values, color='gray', lw=1.5, label=r'-p$\theta$ Electrons')
    
    # Plot the shaded error band
    if sigma_p_theta_e is not None:
        err = sigma_p_theta_e.interp(time=plot_val.time, method='nearest')
        ax_b.fill_between(plot_val.time.values, 
                          (plot_val - err).values, 
                          (plot_val + err).values, 
                          color='cornflowerblue', alpha=0.4, label=r'$\pm \sigma$')
    
    ax_b.set_ylabel(r'-p$\theta_e$ (nW/m$^3$)')
    ax_b.legend(loc='upper left', frameon=False)
    format_axes(ax_b, xaxis='off')

    
    ax_c = axes[2]
    plot_val_c = -PiD_e
    
    # Plot the central line
    ax_c.plot(plot_val_c.time.values, plot_val_c.values, color='gray', lw=1.5, label=r'-$\Pi$D Electrons')
    
    # Plot the shaded error band
    if sigma_PiD_e is not None:
        err_c = sigma_PiD_e.interp(time=plot_val_c.time, method='nearest')
        ax_c.fill_between(plot_val_c.time.values, 
                          (plot_val_c - err_c).values, 
                          (plot_val_c + err_c).values, 
                          color='cornflowerblue', alpha=0.4, label=r'$\pm \sigma$')

    ax_c.set_ylabel(r'-$\Pi_e$:D (nW/m$^3$)')
    ax_c.legend(loc='upper left', frameon=False)
    format_axes(ax_c, xaxis='on')
    
    
    ax_c.set_xlabel(f'Time after {t0.strftime("%Y-%m-%d %H:%M:%S")} (s)')
    t0_naive = t0.replace(tzinfo=None)
    def time_to_seconds(x, pos):
        time_val = mdates.num2date(x)
        time_val_naive = dt.datetime(time_val.year, time_val.month, time_val.day, 
                                      time_val.hour, time_val.minute, time_val.second, 
                                      time_val.microsecond)
        return f'{(time_val_naive - t0_naive).total_seconds():.0f}'
        
    ax_c.xaxis.set_major_formatter(plt.FuncFormatter(time_to_seconds))
    plt.setp(axes, xlim=(t0, t1))

    return fig, axes

#t0 = dt.datetime(2017, 6, 17, 20, 24, 3)
#t1 = dt.datetime(2017, 6, 17, 20, 24, 11) 
#t0 = dt.datetime(2016, 10, 22, 12, 59, 13)
#t1 = dt.datetime(2016, 10, 22, 12, 59, 15)
t0 = dt.datetime(2017, 7, 11, 22, 34, 0)
t1 = dt.datetime(2017, 7, 11, 22, 34, 5)
#t0 = dt.datetime(2015, 12, 14, 1, 17, 38)
#t1 = dt.datetime(2015, 12, 14, 1, 17, 39)
warnings.filterwarnings("ignore", message=".*rename.*does not create an index anymore.*")

print("Preparing electron data files...")
maxlut.main_ts('mms1', 'brst', 'des-dist', t0, t1)
maxlut.main_ts('mms2', 'brst', 'des-dist', t0, t1)
maxlut.main_ts('mms3', 'brst', 'des-dist', t0, t1)
maxlut.main_ts('mms4', 'brst', 'des-dist', t0, t1)

print("Loading combined data...")
db.load_data(t0, t1, mode='brst')
db.load_entropy('brst', t0, t1)

print("Generating plot...")
fig, axes = plot_electron_data(t0, t1, mode='brst')
#fig.savefig("electron_plot_styled.png", dpi=200)
fig.savefig("/Users/krmhanieh/Desktop/los alamos/Shield/paper/electron_plot_styled3.png",dpi=200)
print("Plot saved as electron_plot_styled.png")
plt.show()

