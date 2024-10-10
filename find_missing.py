import sys, os
import xarray as xr
import utils

tc_names = utils.get_all_tc_names()

for name in tc_names:

    tc_file = f"/scratch/ll44/sc6160/tc_data/{name}.json"
    if utils.load_tc_data(tc_file) == None:
        continue

    tc_name, tc_id, start_time, end_time, tc_dir = utils.load_tc_data(tc_file)

    try:
        obs = xr.open_dataset(f"/scratch/{tc_dir}/sc6160/data/obs/{tc_name}_{tc_id}_obs_data.nc")
    except:
        print(f"No obs. data for {tc_name}")

    try:
        pred = xr.open_dataset(f"/scratch/{tc_dir}/sc6160/out/pred/{tc_name}_{tc_id}_pred_data.nc")
    except:
        print(f"No pred. data for {tc_name}")