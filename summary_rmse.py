import os, sys
import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import utils
import argparse

parser = argparse.ArgumentParser(description='Process an ensemble of TCs.')
parser.add_argument('--good', action='store_true', help='Only process the "good" TCs')
args = parser.parse_args()

if args.good:
    tc_names = utils.get_all_good_tc_names()
else:
    tc_names = utils.get_all_tc_names()

obs_tracks = []
pred_tracks = []

for name in tc_names:

    tc_file = f"/scratch/ll44/sc6160/data/tc_data/{name}.json"
    if utils.load_tc_data(tc_file) == None:
        continue
    tc_name, tc_id, start_time, end_time, tc_dir = utils.load_tc_data(tc_file)

    track_ds = None
    try:
        track_ds = xr.open_dataset(f"/scratch/ll44/sc6160/out/tracks/{tc_name}_{tc_id}_tracks.nc")
    except:
        continue

    obs_tracks.append({'lats': track_ds['obs_lats'].values, 'lons': track_ds['obs_lons'].values})
    pred_tracks.append({'lats': track_ds['gc_lats'].values, 'lons': track_ds['gc_lons'].values})

# tracks
def track_rmse(obs_tracks, pred_tracks):

    # maximum length of tracks
    max_timesteps = max(len(track['lats']) for track in obs_tracks + pred_tracks)
    
    all_errors = [[] for _ in range(max_timesteps)]
    
    # errors for each pair of tracks
    for obs, pred in zip(obs_tracks, pred_tracks):
        for t in range(min(len(obs['lats']), len(pred['lats']))):
            error = geodesic((obs['lats'][t], obs['lons'][t]), (pred['lats'][t], pred['lons'][t])).km
            all_errors[t].append(error)
    
    rmse_over_time = []
    
    for errors in all_errors:
        if errors:
            rmse = np.sqrt(np.mean(np.array(errors)**2))
            rmse_over_time.append(rmse)
        else:
            break
    
    return rmse_over_time


track_rmse = track_rmse(obs_tracks, pred_tracks)
plt.figure(figsize=(10, 6))
plt.plot(track_rmse, marker='o', linestyle='-', color='b')
plt.title('RMSE Over Time for TC Tracks')
plt.xlabel('Timestep')
plt.ylabel('RMSE (km)')
plt.grid(True)

if args.good:
    plt.savefig("/scratch/ll44/sc6160/out/plots/summary/track_rmse_good.png")
else:
    plt.savefig("/scratch/ll44/sc6160/out/plots/summary/track_rmse_all.png")



# intensity (wind)
obs_winds = []
pred_winds = []

for name in tc_names:

    tc_file = f"/scratch/ll44/sc6160/data/tc_data/{name}.json"
    if utils.load_tc_data(tc_file) == None:
        continue
    tc_name, tc_id, start_time, end_time, tc_dir = utils.load_tc_data(tc_file)

    wind_ds = None
    try:
        wind_ds = xr.open_dataset(f"/scratch/ll44/sc6160/out/intensity/{tc_name}_{tc_id}_intensity_wind.nc")
    except:
        continue

    obs_winds.append(wind_ds['obs'].values)
    pred_winds.append(wind_ds['pred'].values)

def wind_rmse(obs_winds, pred_winds):

    num_timesteps = max(max(len(tc) for tc in obs_winds), max(len(tc) for tc in pred_winds))

    rmse_over_time = []

    for t in range(num_timesteps):
        
        obs = []
        preds = []
        for i in range(len(obs_winds)):
            o = obs_winds[i]
            p = pred_winds[i]
            if len(o) > t and len(p) > t:

                if np.isnan(o[t]) or np.isnan(pred_winds[i][t]):
                    continue

                obs.append(o[t])
                preds.append(pred_winds[i][t])

        rmse = np.sqrt(np.mean((np.array(obs) - np.array(preds)) ** 2))
        rmse_over_time.append(rmse)
    
    return rmse_over_time

wind_rmse = wind_rmse(obs_winds, pred_winds)
plt.figure(figsize=(10, 6))
plt.plot(wind_rmse, marker='o', linestyle='-', color='b')
plt.title('RMSE Over Time for Intensity (Max. Wind Speed)')
plt.xlabel('Timestep')
plt.ylabel('RMSE (m/s)')
plt.grid(True)

if args.good:
    plt.savefig("/scratch/ll44/sc6160/out/plots/summary/wind_rmse_good.png")
else:
    plt.savefig("/scratch/ll44/sc6160/out/plots/summary/wind_rmse_all.png")



# intensity (mslp)
obs_mslps = []
pred_mslps = []

for name in tc_names:

    tc_file = f"/scratch/ll44/sc6160/data/tc_data/{name}.json"
    if utils.load_tc_data(tc_file) == None:
        continue
    tc_name, tc_id, start_time, end_time, tc_dir = utils.load_tc_data(tc_file)

    mslp_ds = None
    try:
        mslp_ds = xr.open_dataset(f"/scratch/ll44/sc6160/out/intensity/{tc_name}_{tc_id}_intensity_mslp.nc")
    except:
        continue

    obs_mslps.append(mslp_ds['obs'].values)
    pred_mslps.append(mslp_ds['pred'].values)

def mslp_rmse(obs_mslps, pred_mslps):

    num_timesteps = max(max(len(tc) for tc in obs_mslps), max(len(tc) for tc in pred_mslps))

    rmse_over_time = []

    for t in range(num_timesteps):
        
        obs = []
        preds = []
        for i in range(len(obs_mslps)):
            o = obs_mslps[i]
            p = pred_mslps[i]
            if len(o) > t and len(p) > t:

                if np.isnan(o[t]) or np.isnan(pred_mslps[i][t]):
                    continue

                obs.append(o[t])
                preds.append(pred_mslps[i][t])

        rmse = np.sqrt(np.mean((np.array(obs) - np.array(preds)) ** 2))
        rmse_over_time.append(rmse)
    
    return rmse_over_time

mslp_rmse = mslp_rmse(obs_mslps, pred_mslps)
plt.figure(figsize=(10, 6))
plt.plot(mslp_rmse, marker='o', linestyle='-', color='b')
plt.title('RMSE Over Time for Intensity (Min. MSLP)')
plt.xlabel('Timestep')
plt.ylabel('RMSE (hPa)')
plt.grid(True)

if args.good:
    plt.savefig("/scratch/ll44/sc6160/out/plots/summary/mslp_rmse_good.png")
else:
    plt.savefig("/scratch/ll44/sc6160/out/plots/summary/mslp_rmse_all.png")