import os, sys
from datetime import datetime, timedelta
import src.utils.utils as utils
import tracking
import xarray as xr
import numpy as np

d = xr.open_dataset("/scratch/ll44/sc6160/data/obs/Billy_2022072S11107_obs_data.nc")

frame = d.sel(time=d['time'].values[0])['land_sea_mask']

frame.to_netcdf("/scratch/ll44/sc6160/data/land_sea_mask.nc")