from datetime import datetime, timedelta

import xarray as xr
import numpy as np

import plotting
import utils
import tracking

pred_file_path = "/scratch/ll44/sc6160/out/preds.nc"
eval_file_path = "/scratch/ll44/sc6160/out/evals.nc"

preds = xr.open_dataset(pred_file_path)
evals = xr.open_dataset(eval_file_path)

ref_time = datetime(2022, 1, 1, 0, 0, 0)
ibtracs = xr.open_dataset("/scratch/ll44/sc6160/data/IBTrACS/IBTrACS.last3years.v04r01.nc")

tc_id = "2022008S13148".encode("utf-8")
tc_name = "TIFFANY"
tc_season = 2022

# # locate TC id
# matching_ids = []

# for i in range(ibtracs.sizes["storm"]):

#     name = ibtracs["name"].isel(storm=i).item().decode("utf-8")
#     season = ibtracs["season"].isel(storm=i).item()
#     sid = ibtracs["sid"].isel(storm=i).item()

#     if name == tc_name and season == tc_season:
        # matching_ids.append(sid)

tc_start_time = datetime(2022, 1, 8, 0, 0, 0)
tc_end_time = datetime(2022, 1, 17, 12, 0, 0)

tc_data = ibtracs.where(ibtracs["sid"] == tc_id, drop=True)
tc_lons = tc_data['lon']
tc_lons_np = tc_lons.values
tc_lats = tc_data['lat']
tc_lats_np = tc_lats.values

# plotting.plot_tc_track(tc_id, tc_name, tc_lons, tc_lats)

track_lons, track_lats = tracking.track_from_start_and_end(preds, tc_lons_np[0][0], tc_lats_np[0][0], \
                                  tc_start_time, tc_end_time, ref_time)

plotting.plot_tc_track(tc_id, tc_name, track_lons, track_lats)