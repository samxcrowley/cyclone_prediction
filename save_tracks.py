import sys
from datetime import datetime

import xarray as xr
import numpy as np
import pandas as pd

import utils, tracking as tracking

tc_file = sys.argv[1]
tc_name, tc_id, start_time, end_time, tc_dir = utils.load_tc_data(tc_file)

ibtracs = xr.open_dataset("/scratch/ll44/sc6160/data/IBTrACS/IBTrACS.ALL.v04r01.nc")

obs = None
try:
    obs = xr.open_dataset(f"/scratch/{tc_dir}/sc6160/data/obs/{tc_name}_{tc_id}_obs_data.nc")
except:
    print(f"No obs data for {tc_name} yet.")
    sys.exit(0)
preds = xr.open_dataset(f"/scratch/{tc_dir}/sc6160/out/pred/{tc_name}_{tc_id}_pred_data.nc")

timestep_offset = 0
offset_start_time = start_time + (utils.TIME_STEP * timestep_offset)

# IBTrACS track
tc_data = ibtracs.where(ibtracs['sid'] == tc_id.encode('utf-8'), drop=True) \
                .isel(date_time=slice(0, None, 2)) # change from 6-hourly to 3-hourly
tc_lats = tc_data['lat'].values[0]
tc_lats = tc_lats[~np.isnan(tc_lats)]
tc_lons = tc_data['lon'].values[0]
tc_lons = tc_lons[~np.isnan(tc_lons)]
tc_start_lat = tc_lats[timestep_offset]
tc_start_lon = tc_lons[timestep_offset]

# OBS track
obs_track_lats, obs_track_lons, _ = tracking.gc_track(obs, tc_start_lat, tc_start_lon, offset_start_time, start_time)

# GC track
gc_track_lats, gc_track_lons, _ = tracking.gc_track(preds, tc_start_lat, tc_start_lon, offset_start_time, start_time)

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

ds.to_netcdf(f"/scratch/ll44/sc6160/out/tracks/{tc_name}_{tc_id}_tracks.nc")