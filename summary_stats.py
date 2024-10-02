import os, sys
import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import utils

tc_names = utils.get_all_tc_names()

obs_max_winds = []
pred_max_winds = []

obs_max_mslps = []
pred_max_mslps = []

for name in tc_names:

    tc_file = f"{name}.json"
    if utils.load_tc_data(tc_file) == None:
        continue
    tc_name, tc_id, start_time, end_time, tc_dir = utils.load_tc_data(tc_file)

    wind_ds = None
    mslp_ds = None

    try:
        wind_ds = xr.open_dataset(f"/scratch/ll44/sc6160/out/intensity/{tc_name}_{tc_id}_intensity_wind.nc")
    except:
        print(f"Couldn't find an intensity_wind file for {tc_name}")
        continue

    obs_max_wind = float(wind_ds['obs'].max().values)
    pred_max_wind = float(wind_ds['pred'].max().values)

    obs_max_winds.append(obs_max_wind)
    pred_max_winds.append(pred_max_wind)

    try:
        mslp_ds = xr.open_dataset(f"/scratch/ll44/sc6160/out/intensity/{tc_name}_{tc_id}_intensity_mslp.nc")
    except:
        print(f"Couldn't find an intensity_wind file for {tc_name}")

    obs_max_mslp = float(mslp_ds['obs'].min().values)
    pred_max_mslp = float(mslp_ds['pred'].min().values)

    obs_max_mslps.append(obs_max_mslp)
    pred_max_mslps.append(pred_max_mslp)

wind_data = pd.DataFrame({
    'Max. Wind Speed': obs_max_winds + pred_max_winds,
    'Type': ['Observed'] * len(obs_max_winds) + ['Predicted'] * len(pred_max_winds)
})

plt.figure(figsize=(8, 6))
sns.violinplot(x='Type', y='Max. Wind Speed', data=wind_data)
plt.title('Distribution of Max. Wind Speeds (Observed vs Predicted)')
plt.ylabel('Max. Wind Speed (m/s)')
plt.xlabel('Data Type')
plt.savefig("/scratch/ll44/sc6160/out/plots/summary/intensity_wind_violin.png")

mslp_data = pd.DataFrame({
    'Min. MSLP': obs_max_mslps + pred_max_mslps,
    'Type': ['Observed'] * len(obs_max_mslps) + ['Predicted'] * len(pred_max_mslps)
})

plt.figure(figsize=(8, 6))
sns.violinplot(x='Type', y='Min. MSLP', data=mslp_data)
plt.title('Distribution of Min. MSLP (Observed vs Predicted)')
plt.ylabel('Min. MSLP (hPa)')
plt.xlabel('Data Type')
plt.savefig("/scratch/ll44/sc6160/out/plots/summary/intensity_mslp_violin.png")