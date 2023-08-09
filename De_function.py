from pymms.data import fgm, edp, fpi, util
import datetime as dt
import xarray as xr
import numpy as np
from scipy import constants as c
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt, dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
import time
import os

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

