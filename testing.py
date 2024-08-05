from datetime import datetime, timedelta

import xarray as xr
import numpy as np

import plotting
import data_utils
import tracking

pred_file_path = "out/preds.nc"
eval_file_path = "out/evals.nc"

preds = xr.open_dataset(pred_file_path)
evals = xr.open_dataset(eval_file_path)

ibtracs = xr.open_dataset("/scratch/ll44/sc6160/data/IBTrACS/IBTrACS.last3years.v04r01.nc")

# TO-DO: use ID

tc_id = "2022055S13129"

tc_name = "ANIKA"
tc_season = 2022

# locate TC id
matching_ids = []

for i in range(ibtracs.sizes["storm"]):

    name = ibtracs["name"].isel(storm=i).item().decode("utf-8")
    season = ibtracs["season"].isel(storm=i).item()
    sid = ibtracs["sid"].isel(storm=i).item()

    if name == tc_name and season == tc_season:
        matching_ids.append(sid)

if len(matching_ids) == 0:
    raise ValueError(f"No cyclone found with name {tc_name} and year {tc_season}.")
elif len(matching_ids) > 1:
    raise ValueError(f"Multiple cyclones found with name {tc_name} and year {tc_season}: {matching_ids}")

tc_id = matching_ids[0]
print(tc_id)
tc_data = ibtracs.where(ibtracs["sid"] == tc_id, drop=True)
tc_lons = tc_data['lon']
tc_lons_np = tc_lons.values
tc_lats = tc_data['lat']
tc_lats_np = tc_lats.values

# plotting.plot_tc_track(tc_id, tc_name, tc_lons, tc_lats)

# test tracking
start_lon = tc_lons_np[0][0]
start_lat = tc_lats_np[0][0]
start_time = datetime(2022, 2, 25, 0, 0, 0) # 2022-02-25 00:00:00
end_time = datetime(2022, 3, 3, 12, 0, 0) # 2022-03-03 12:00:00

mslp_track = tracking.naive_mslp_track(preds, start_lon, start_lat, start_time, end_time)