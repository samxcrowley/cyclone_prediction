import sys
from datetime import datetime

import xarray as xr
import numpy as np
import pandas as pd

import utils, tracking

# change the below fields for each TC
ibtracs = xr.open_dataset("/scratch/ll44/sc6160/data/IBTrACS/IBTrACS.ALL.v04r01.nc")
obs = xr.open_dataset("/scratch/ll44/sc6160/data/2023-04/source-era5_data-2023-4_res-0.25_levels-37_tc-ilsa.nc") \
        .isel(time=slice(1, -1))
preds = xr.open_dataset("/scratch/ll44/sc6160/out/preds_ilsa.nc")

tc_name = "ilsa"
tc_id = "2023096S08133".encode("utf-8")
tc_start_time = datetime(2023, 4, 6, 6, 0, 0)
tc_end_time = datetime(2023, 4, 17, 0, 0, 0)

# IBTrACS track
tc_data = ibtracs.where(ibtracs['sid'] == tc_id, drop=True)
tc_lats = tc_data['lat'].values[0]
tc_lats = tc_lats[~np.isnan(tc_lats)]
tc_lons = tc_data['lon'].values[0]
tc_lons = tc_lons[~np.isnan(tc_lons)]
tc_start_lat = tc_lats[0]
tc_start_lon = tc_lons[0]

# OBS track
obs_track_lats, obs_track_lons, _ = tracking.gc_track(obs, tc_start_lat, tc_start_lon, tc_start_time)

# GC track
gc_track_lats, gc_track_lons, _ = tracking.gc_track(preds, tc_start_lat, tc_start_lon, tc_start_time)

# save lists in dataset
lists = [tc_lats, tc_lons, obs_track_lats, obs_track_lons, gc_track_lats, gc_track_lons]

ds = xr.Dataset(
    data_vars={
        'ibtracs_lats': (['index1'], tc_lats),
        'ibtracs_lons': (['index2'], tc_lons),
        'obs_lats': (['index3'], obs_track_lats),
        'obs_lons': (['index4'], obs_track_lons),
        'gc_lats': (['index5'], gc_track_lats),
        'gc_lons': (['index6'], gc_track_lons)
    }
)

ds.to_netcdf(f"/scratch/ll44/sc6160/out/tracks/{tc_name}.nc")