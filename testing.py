from datetime import datetime, timedelta

import xarray as xr
import numpy as np
import pandas as pd
from glob import glob

import plotting
import utils
import tracking
import density_map_testing

# data = xr.open_dataset("/scratch/ll44/sc6160/data/2022-03/source-era5_data-2022-3_res-0.25_levels-37_ausonly_time-6h.nc")
# print(data['relative_vorticity'])
# print(data['relative_vorticity'].values)

vort = xr.open_dataset(glob("/g/data/rt52/era5/pressure-levels/reanalysis/vo/2022/vo_era5_oper_pl_20220301-*.nc")[0]) \
    .sel(latitude=slice(utils.AUS_LAT_BOUNDS[1], utils.AUS_LAT_BOUNDS[0]), \
            longitude=slice(utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1])) \
    .resample(time='6h').nearest()

# # print(vort)
# print(vort['vo'].sel(time=datetime(2023, 3, 1, 0), level=850, method='nearest').values[:5])
# print(vort['vo'].sel(time=datetime(2023, 3, 1, 0), level=850, method='nearest'))