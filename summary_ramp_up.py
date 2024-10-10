import sys, os
import argparse
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils

parser = argparse.ArgumentParser(description='Process an ensemble of TCs.')
parser.add_argument('--good', action='store_true', help='Only process the "good" TCs')
args = parser.parse_args()

if args.good:
    tc_names = utils.get_all_good_tc_names()
else:
    tc_names = utils.get_all_tc_names()

obs_steps = []
pred_steps = []

threshold = 25

for name in tc_names:

    tc_file = f"/scratch/ll44/sc6160/data/tc_data/{name}.json"
    if utils.load_tc_data(tc_file) == None:
        continue
    tc_name, tc_id, start_time, end_time, tc_dir = utils.load_tc_data(tc_file)

    wind_ds = None
    try:
        wind_ds = xr.open_dataset(f"/scratch/ll44/sc6160/out/intensity/{tc_name}_{tc_id}_intensity_wind.nc")
    except:
        print(f"No wind intensity dataset for {tc_name}.")
        continue

    wind_obs = wind_ds['obs']
    wind_pred = wind_ds['pred']

    obs_n_steps = 0
    pred_n_steps = 0

    for v in wind_obs.values:
        if utils.ms_to_knots(v) < threshold:
            obs_n_steps += 1
        else:
            break

    for v in wind_pred.values:
        if utils.ms_to_knots(v) < threshold:
            pred_n_steps += 1
        else:
            break

    obs_steps.append(obs_n_steps)
    pred_steps.append(pred_n_steps)

# plotting
data = pd.DataFrame({
    'timesteps': obs_steps + pred_steps,
    'category': ['Observed'] * len(obs_steps) + ['Predicted'] * len(pred_steps)
})

plt.figure(figsize=(8, 6))
sns.violinplot(x='category', y='timesteps', data=data, cut=0)
plt.title('Ramp-Up Times to 25 Knots (Observed vs Predicted)')
plt.ylabel('Number of Timesteps')
plt.xlabel('Category')

if args.good:
    plt.savefig("/scratch/ll44/sc6160/out/plots/summary/ramp_up_violin_good.png")
else:
    plt.savefig("/scratch/ll44/sc6160/out/plots/summary/ramp_up_violin_all.png")

plt.figure(figsize=(8, 6))
sns.boxplot(x='category', y='timesteps', data=data)
plt.title('Ramp-Up Times to 25 Knots (Observed vs Predicted)')
plt.ylabel('Number of Timesteps')
plt.xlabel('Category')
plt.show()

if args.good:
    plt.savefig("/scratch/ll44/sc6160/out/plots/summary/ramp_up_boxplot_good.png")
else:
    plt.savefig("/scratch/ll44/sc6160/out/plots/summary/ramp_up_boxplot_all.png")