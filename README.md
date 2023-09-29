# entropy-hgio-2023
The evolution of kinetic entropy and its connection to energy transfer in fundamental processes

## Create a Database
Load all of the data required for analyzing entropy and other dissipation parameters for a given time interval. Data is resample to the 30ms time cadence of the DES instrument.

```python
import datetime as dt
import database

# Define the time interval
t0 = dt.datetime(2017, 7, 11, 22, 34, 0)
t1 = dt.datetime(2017, 7, 11, 22, 34, 5)

# Load data into file
file = database.load_data(t0, t1)
```
## Plot Dissipation Measures
Compare entropy measures to other indicators of energy dissipation.

```python
import datetime as dt
import plots

# Define the time interval
t0 = dt.datetime(2017, 7, 11, 22, 34, 0)
t1 = dt.datetime(2017, 7, 11, 22, 34, 5)

# Create the plot
plots.dissipation_measures(t0, t1)
```

## Plot Maxwellian Look-Up Table
Plot (and create, if not already created) a Maxwellian distribution Look-Up Table (LUT).

```Python
import datetime as dt
import plots

# Define the time interval
t0 = dt.datetime(2017, 7, 11, 22, 34, 0)
t1 = dt.datetime(2017, 7, 11, 22, 34, 5)

# Create the plot
plots.max_lut('mms3', 'brst', 'des-dist', t0, t1)
```