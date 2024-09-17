from datetime import datetime, timedelta
import os
from glob import glob

import xarray as xr
import numpy as np

import plotting, utils

# TC Olga
obs = xr.open_dataset("/scratch/ll44/sc6160/data/2024-04/source-era5_data-2024-4_res-0.25_levels-37.nc")
pred = xr.open_dataset("/scratch/ll44/sc6160/out/preds_olga.nc")

