import numpy as np
import datetime as dt
from scipy import constants
from pymms.data import fgm, fpi
from matplotlib import pyplot as plt
import xarray as xr
import os

def _calculate_1d_dist(dist_obj, component='x'):
    
    dist_obj.precondition()
    f_data = dist_obj._f
    phi = dist_obj._phi
    theta = dist_obj._theta
    energy = dist_obj._energy
    mass = dist_obj.mass
    phi_grid, theta_grid = np.meshgrid(phi, theta, indexing='ij')
    with np.errstate(divide='ignore', invalid='ignore'):
        speed = np.sqrt(2.0 * constants.eV * energy / mass)
    speed[~np.isfinite(speed)] = 0
    vx = -speed * np.sin(theta_grid[..., np.newaxis]) * np.cos(phi_grid[..., np.newaxis])
    vy = -speed * np.sin(theta_grid[..., np.newaxis]) * np.sin(phi_grid[..., np.newaxis])
    vz = -speed * np.cos(theta_grid[..., np.newaxis])
    if component == 'x':
        v_reduce = vx
    elif component == 'y':
        v_reduce = vy
    else: # component == 'z'
        v_reduce = vz
    v_max = np.max(speed) * 1e-3
    v_bins = np.linspace(-v_max, v_max, 128)
    f_1d, _ = np.histogram(v_reduce.flatten() * 1e-3, bins=v_bins, weights=f_data.flatten())
    bin_counts, _ = np.histogram(v_reduce.flatten() * 1e-3, bins=v_bins)
    with np.errstate(divide='ignore', invalid='ignore'):
        f_1d = f_1d / bin_counts
    f_1d[np.isnan(f_1d)] = 0
    v_centers = (v_bins[:-1] + v_bins[1:]) / 2
    return xr.DataArray(f_1d, coords={'velocity': v_centers}, dims=['velocity'])

def plot_1d_distributions(sc, instr, mode, time, vlim=None, vscale=None, n_steps=1):
    """
    Loads, optionally averages, rotates, and plots 1D FPI distributions,
    generating four separate comparison plots with lines at the calculated bulk velocity.
    """
    optdesc = instr+'-dist'
    time_delta = dt.timedelta(seconds=2) if mode == 'brst' else dt.timedelta(seconds=30)
    t0 = time - time_delta
    t1 = time + time_delta

    b_data = fgm.load_data(sc=sc, mode=mode, start_date=t0, end_date=t1)
    moms_data = fpi.load_moms(sc=sc, mode=mode, optdesc=instr+'-moms', start_date=t0, end_date=t1)
    dist_data = fpi.load_dist(sc=sc, mode=mode, optdesc=optdesc, start_date=t0, end_date=t1)
    fpi_kwargs = fpi.precond_params(sc=sc, mode=mode, level='l2', optdesc=optdesc,
                                    start_date=t0, end_date=t1, time=dist_data['time'])
    dist_data['dist'].data = dist_data['dist'].where(dist_data['dist'] >= 0, 0)
    scpot = fpi_kwargs.pop('scpot')
    
    print(f"Processing {n_steps} time step(s)...")
    center_idx = np.abs(dist_data['time'] - np.datetime64(time)).argmin().item()
    t_fpi_center = dist_data['time'][center_idx]
    
    if n_steps > 1:
        half_window = n_steps // 2
        start_idx = max(0, center_idx - half_window)
        end_idx = min(len(dist_data['time']), center_idx + half_window + 1)
        time_slice = slice(start_idx, end_idx)
        t_start = dist_data['time'][start_idx].data
        t_end = dist_data['time'][end_idx-1].data
        time_range_slice = slice(t_start, t_end)
        f_dist_slice = dist_data['dist'].isel(time=time_slice)
        f_fpi_averaged_data = f_dist_slice.mean(dim='time')
        phi_vals = dist_data['dist']['phi'].isel(time=center_idx).data if 'time' in dist_data['dist']['phi'].dims else dist_data['dist']['phi'].data
        theta_vals = dist_data['dist']['theta'].isel(time=center_idx).data if 'time' in dist_data['dist']['theta'].dims else dist_data['dist']['theta'].data
        energy_vals = dist_data['dist']['energy'].isel(time=center_idx).data if 'time' in dist_data['dist']['energy'].dims else dist_data['dist']['energy'].data
        f_fpi = xr.DataArray(
            f_fpi_averaged_data.data,
            coords={'phi': ('phi', phi_vals), 'theta': ('theta', theta_vals), 'energy': ('energy', energy_vals)},
            dims=['phi', 'theta', 'energy'],
            attrs=dist_data['dist'].attrs
        )
        b_vec = b_data['B_GSE'].sel(time=time_range_slice).mean(dim='time').data[0:3]
        v_bulk_vec = moms_data['velocity'].sel(time=time_range_slice).mean(dim='time').data
        Vsci = scpot.sel(time=time_range_slice).mean(dim='time').data
    else:
        f_fpi = dist_data['dist'].sel(time=time, method='nearest')
        b_vec = b_data['B_GSE'].sel(time=time, method='nearest').data[0:3]
        v_bulk_vec = moms_data['velocity'].sel(time=time, method='nearest').data
        Vsci = scpot.sel(time=time, method='nearest').data
        
    f = fpi.Distribution_Function.from_fpi(f_fpi, scpot=Vsci, time=t_fpi_center.data, **fpi_kwargs)
    
    f_rotated_list = f.rotate(b_vec, v_bulk_vec, cs='vxb', orientations=['pp1'])
    f_rotated_obj = f_rotated_list[0]
    
    print("Calculating all 1D distribution components and Maxwellian fit...")
    f_par = _calculate_1d_dist(f_rotated_obj, component='x')
    f_perp1 = _calculate_1d_dist(f_rotated_obj, component='y')
    f_perp2 = _calculate_1d_dist(f_rotated_obj, component='z')
    
    f_max = f_rotated_obj.maxwellian()
    f_max_par = _calculate_1d_dist(f_max, component='x')

    # Construct the field-aligned basis vectors from the original GSE vectors
    b_mag = np.linalg.norm(b_vec)
    e_parallel = b_vec / b_mag if b_mag > 0 else np.array([1., 0., 0.])
    
    b_cross_v = np.cross(b_vec, v_bulk_vec)
    b_cross_v_mag = np.linalg.norm(b_cross_v)
    e_perp2 = b_cross_v / b_cross_v_mag if b_cross_v_mag > 0 else np.cross(e_parallel, [0,0,1])
    
    e_perp1 = np.cross(e_perp2, e_parallel)

    # Project the total bulk velocity onto these basis vectors
    v_parallel_kms = np.dot(v_bulk_vec, e_parallel)
    v_perp1_kms = np.dot(v_bulk_vec, e_perp1)
    v_perp2_kms = np.dot(v_bulk_vec, e_perp2)
    
    # Scale the physical velocities for plotting
    v_bulk_par_plot = v_parallel_kms / (vscale if vscale else 1)
    v_bulk_perp1_plot = v_perp1_kms / (vscale if vscale else 1)
    v_bulk_perp2_plot = v_perp2_kms / (vscale if vscale else 1)
    
    v_unit = 'km/s'
    if vscale is not None:
        v_unit = f'x$10^{{{int(np.log10(vscale))}}}$ km/s'
    
    v_par_scaled = f_par['velocity'].data / (vscale if vscale else 1)
    f_par_data = f_par.data
    valid_par = f_par_data > 0
    
    v_perp1_scaled = f_perp1['velocity'].data / (vscale if vscale else 1)
    f_perp1_data = f_perp1.data
    valid_perp1 = f_perp1_data > 0
    
    v_perp2_scaled = f_perp2['velocity'].data / (vscale if vscale else 1)
    f_perp2_data = f_perp2.data
    valid_perp2 = f_perp2_data > 0
    
    f_max_par_data = f_max_par.data
    
    t_fpi_dt = t_fpi_center.data.astype('datetime64[us]').astype(dt.datetime).item()
    avg_str = f"\n(Averaged over {n_steps} steps)" if n_steps > 1 else ""
    time_subtitle = f"Center Time: {t_fpi_dt.strftime('%Y-%m-%d %H:%M:%S.%f')}{avg_str}"

    # --- PLOT 1: Parallel vs Perpendicular 1 ---
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    ax1.plot(v_par_scaled[valid_par], f_par_data[valid_par], 'D-', color='red', markersize=4, label='Parallel ($f_{\\parallel}$)')
    ax1.plot(v_perp1_scaled[valid_perp1], f_perp1_data[valid_perp1], 'o-', color='blue', markersize=4, label='Perpendicular 1 ($f_{\\perp,1}$)')
    ax1.axvline(v_bulk_par_plot, color='gray', linestyle='--', linewidth=1.5, label=f'Bulk V = {v_parallel_kms:.1f} km/s')
    ax1.set(yscale='log', xlabel=f'Velocity [{v_unit}]', ylabel='f ($s^3/m^6$)')
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax1.legend()
    combined_data1 = np.concatenate((f_par_data[valid_par], f_perp1_data[valid_perp1]))
    if combined_data1.size > 0:
        ax1.set_ylim(np.max(combined_data1) / 1e6, np.max(combined_data1) * 10)
    if vlim is not None:
        ax1.set_xlim(-(vlim / (vscale or 1)), vlim / (vscale or 1))
    ax1.set_title(f"{sc.upper()} {instr.upper()} | Parallel vs. Perp 1\n{time_subtitle}", fontsize=12)
    fig1.tight_layout()

    # --- PLOT 2: Parallel vs Perpendicular 2 ---
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    ax2.plot(v_par_scaled[valid_par], f_par_data[valid_par], 'D-', color='red', markersize=4, label='Parallel ($f_{\\parallel}$)')
    ax2.plot(v_perp2_scaled[valid_perp2], f_perp2_data[valid_perp2], 's-', color='green', markersize=4, label='Perpendicular 2 ($f_{\\perp,2}$)')
    ax2.axvline(v_bulk_par_plot, color='gray', linestyle='--', linewidth=1.5, label=f'Bulk V = {v_parallel_kms:.1f} km/s')
    ax2.set(yscale='log', xlabel=f'Velocity [{v_unit}]', ylabel='f ($s^3/m^6$)')
    ax2.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax2.legend()
    combined_data2 = np.concatenate((f_par_data[valid_par], f_perp2_data[valid_perp2]))
    if combined_data2.size > 0:
        ax2.set_ylim(np.max(combined_data2) / 1e6, np.max(combined_data2) * 10)
    if vlim is not None:
        ax2.set_xlim(-(vlim / (vscale or 1)), vlim / (vscale or 1))
    ax2.set_title(f"{sc.upper()} {instr.upper()} | Parallel vs. Perp 2\n{time_subtitle}", fontsize=12)
    fig2.tight_layout()

    # --- PLOT 3: Perpendicular 1 vs Perpendicular 2 ---
    fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    ax3.plot(v_perp1_scaled[valid_perp1], f_perp1_data[valid_perp1], 'o-', color='blue', markersize=4, label='Perpendicular 1 ($f_{\\perp,1}$)')
    ax3.plot(v_perp2_scaled[valid_perp2], f_perp2_data[valid_perp2], 's-', color='green', markersize=4, label='Perpendicular 2 ($f_{\\perp,2}$)')
    ax3.axvline(v_bulk_perp1_plot, color='gray', linestyle='--', linewidth=1.5, label=f'Bulk V_perp1 = {v_perp1_kms:.1f} km/s')
    ax3.axvline(v_bulk_perp2_plot, color='purple', linestyle=':', linewidth=2, label=f'Bulk V_perp2 = {v_perp2_kms:.1f} km/s') # <-- ADDED THIS LINE
    ax3.set(yscale='log', xlabel=f'Velocity [{v_unit}]', ylabel='f ($s^3/m^6$)')
    ax3.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax3.legend()
    combined_data3 = np.concatenate((f_perp1_data[valid_perp1], f_perp2_data[valid_perp2]))
    if combined_data3.size > 0:
        ax3.set_ylim(np.max(combined_data3) / 1e6, np.max(combined_data3) * 10)
    if vlim is not None:
        ax3.set_xlim(-(vlim / (vscale or 1)), vlim / (vscale or 1))
    ax3.set_title(f"{sc.upper()} {instr.upper()} | Perp 1 vs. Perp 2\n{time_subtitle}", fontsize=12)
    fig3.tight_layout()
    
    # --- PLOT 4: Comprehensive Summary Plot ---
    fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(6, 5))
    ax4.plot(v_par_scaled[valid_par], f_par_data[valid_par], 'D-', color='red', markersize=4, label='Parallel ($f_{\\parallel}$)')
    ax4.plot(v_perp1_scaled[valid_perp1], f_perp1_data[valid_perp1], 'o-', color='blue', markersize=4, label='Perp 1 ($f_{\\perp,1}$)')
    ax4.plot(v_perp2_scaled[valid_perp2], f_perp2_data[valid_perp2], 's-', color='green', markersize=4, label='Perp 2 ($f_{\\perp,2}$)')
    ax4.plot(v_par_scaled, f_max_par_data, '--', color='cyan', linewidth=2, label='Maxwellian Fit')
    ax4.axvline(v_bulk_par_plot, color='gray', linestyle='--', linewidth=1.5, label=f'Bulk V_par = {v_parallel_kms:.1f} km/s')
    ax4.set(yscale='log', xlabel=f'Velocity [{v_unit}]', ylabel='f ($s^3/m^6$)')
    ax4.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax4.legend()
    combined_data4 = np.concatenate((f_par_data[valid_par], f_perp1_data[valid_perp1], f_perp2_data[valid_perp2]))
    if combined_data4.size > 0:
        ax4.set_ylim(np.max(combined_data4) / 1e7, np.max(combined_data4) * 10)
    if vlim is not None:
        ax4.set_xlim(-(vlim / (vscale or 1)), vlim / (vscale or 1))
    ax4.set_title(f"{sc.upper()} {instr.upper()} | Comprehensive Plot\n{time_subtitle}", fontsize=12)
    fig4.tight_layout()
    
    return [fig1, fig2, fig3, fig4], [ax1, ax2, ax3, ax4]

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Plot 1D FPI distributions, with optional averaging.')
    
    parser.add_argument('sc', type=str, help='Spacecraft Identifier (e.g., mms1)')
    parser.add_argument('instr', type=str, help='Instrument (des or dis)')
    parser.add_argument('mode', type=str, help='Data rate mode (brst or fast)')
    parser.add_argument('time', type=str, help='Center time in "YYYY-MM-DDTHH:MM:SS.ffffff" format')
    
    parser.add_argument('--filename', type=str, help='Base output file name to save the plots.')
    parser.add_argument('--vlim', type=float, default=40000, help='Max velocity limit for display on the x-axis.')
    parser.add_argument('--vscale', type=float, default=1000, help='Velocity scaling factor.')
    parser.add_argument('--no-show', action='store_true', help='Do not show the plot on screen.')
    parser.add_argument('--n_steps', type=int, default=1, 
                        help='Number of adjacent time steps to average together. Default is 1 (no averaging).')

    args = parser.parse_args()
    time = dt.datetime.strptime(args.time, '%Y-%m-%dT%H:%M:%S.%f')

    figs, axes = plot_1d_distributions(args.sc, args.instr, args.mode, time,
                                      vlim=args.vlim, vscale=args.vscale, 
                                      n_steps=args.n_steps)
    
    if args.filename:
        base, ext = os.path.splitext(args.filename)
        fname1 = base + '_par-perp1' + ext
        fname2 = base + '_par-perp2' + ext
        fname3 = base + '_perp1-perp2' + ext
        fname4 = base + '_comprehensive' + ext
        
        figs[0].savefig(fname1)
        figs[1].savefig(fname2)
        figs[2].savefig(fname3)
        figs[3].savefig(fname4)
        print(f"Plots saved to {fname1}, {fname2}, {fname3}, and {fname4}")
    
    if not args.no_show:
        plt.show()
