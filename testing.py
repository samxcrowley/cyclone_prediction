from datetime import datetime, timedelta
import os
from glob import glob

import xarray as xr
import numpy as np

import plotting, utils

preds = xr.open_dataset("/scratch/ll44/sc6160/data/2023-04/source-era5_data-2023-4_res-0.25_levels-37_tc-ilsa.nc")
print(preds['mean_sea_level_pressure'].values[:10])