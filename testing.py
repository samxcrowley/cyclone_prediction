from datetime import datetime, timedelta

import xarray as xr
import numpy as np
import pandas as pd

import plotting
import utils
import tracking
import density_map_testing

# merged = xr.open_dataset("/scratch/ll44/sc6160/data/2022-01/merged_resampled_6h_full.nc")

# pred_file_path = "/scratch/ll44/sc6160/out/preds.nc"
# eval_file_path = "/scratch/ll44/sc6160/out/evals.nc"

# preds = xr.open_dataset(pred_file_path)
# evals = xr.open_dataset(eval_file_path)

# ref_time = datetime(2022, 1, 1, 0, 0, 0)
ibtracs = xr.open_dataset("/scratch/ll44/sc6160/data/IBTrACS/IBTrACS.last3years.v04r01.nc")

tcs = ["2022008S13148", "2022054S13100", "2022055S13129",
       "2022127S07088", "2022038S19164"]

tracks = []
for tc in tcs:
    
    tc_id = tc.encode("utf-8")
    tc_data = ibtracs.where(ibtracs["sid"] == tc_id, drop=True)

    tc_lons = tc_data['lon']
    tc_lons_np = tc_lons.values[0]
    tc_lats = tc_data['lat']
    tc_lats_np = tc_lats.values[0]

    times = tc_lons['time'][0]
    times = times[~pd.isna(times)]
    tc_start_time = times.min().values
    tc_start_time = pd.to_datetime(tc_start_time).to_pydatetime()
    tc_end_time = times.max().values
    tc_end_time = pd.to_datetime(tc_end_time).to_pydatetime()

    track = {'start_time': tc_start_time,
             'end_time': tc_end_time,
             'lats': tc_lats_np,
             'lons': tc_lons_np}
    
    tracks.append(track)

density_map_testing.plot_density_map(tracks)

# plotting.plot_tc_track(tc_id, tc_name, tc_lons, tc_lats)

# track_lons, track_lats = tracking.track_from_start_and_end(preds, tc_lons_np[0][0], tc_lats_np[0][0], \
#                                   tc_start_time, tc_end_time, ref_time)

# plotting.plot_tc_track(tc_id, f"{tc_name}_prediction_track", track_lons, track_lats, aus_bounds=True)