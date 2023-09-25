import xarray as xr
from pathlib import Path
from matplotlib import pyplot as plt


from pymms import config
import database
import physics

data_path = Path(config['dropbox_root'])

def dissipation_measures(t0, t1):

    # Create a file name
    filename = '_'.join(('mms', 'hgio',
                         t0.strftime('%Y%m%d%_H%M%S'),
                         t1.strftime('%Y%m%d%_H%M%S')))
    file_path = (data_path / filename).with_suffix('.nc')
    if ~file_path.exists():
        file_path = database.load_data(t0, t1)

    # Load the data
    data = xr.load_dataset(file_path)
    
    # Calculate electron frame dissipation measure
    De_moms = xr.Dataset()
    for idx in range(1, 5):
        sc = str(idx)
        De_moms['De'+sc] = physics.De_moms(data['E'+sc], data['B'+sc],
                                           data['ne'+sc], data['Vi'+sc],
                                           data['Ve'+sc])
    
    De_curl = physics.De_curl(data[['E1', 'E2', 'E3', 'E4']],
                              data[['B1', 'B2', 'B3', 'B4']],
                              data[['Ve1', 'Ve2', 'Ve3', 'Ve4']],
                              data[['r1', 'r2', 'r3', 'r4']])

    # Plot the data
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False)
    
    # Electron Frame Dissipation Measure
    ax = axes[0,0]
    De_moms['De1'].plot(ax=ax, color='Black', label='MMS1')
    De_moms['De2'].plot(ax=ax, color='Blue', label='MMS1')
    De_moms['De3'].plot(ax=ax, color='Green', label='MMS1')
    De_moms['De4'].plot(ax=ax, color='Red', label='MMS1')
    De_curl.plot(ax=ax, color='magenta', label='Curl')
    ax.set_title('')
    ax.set_ylabel('De [$nW/m^{3}$]')
    ax.legend()

    plt.show()
