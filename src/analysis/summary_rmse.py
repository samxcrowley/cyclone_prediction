import os, sys
import xarray as xr
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import src.utils.utils as utils

obs_tracks = []
pred_tracks = []

tc_names = utils.get_all_tc_names()

for name in tc_names:

    tc_file = f"/scratch/ll44/sc6160/tc_data/{name}.json"
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
def track_rmse(observed_tracks, predicted_tracks):

    max_length = 0
    for track in observed_tracks:
        if len(track['lats']) > max_length:
            max_length = len(track['lats'])

    rmse_over_time = []

    for t in range(max_length):

        distances = []
    
        for obs_track, pred_track in zip(observed_tracks, predicted_tracks):

            obs_lats = obs_track['lats']
            obs_lons = obs_track['lons']
            pred_lats = pred_track['lats']
            pred_lons = pred_track['lons']

            if t < len(obs_lats) and t < len(pred_lats):

                for obs_lat, obs_lon, pred_lat, pred_lon in zip(obs_lats, obs_lons, pred_lats, pred_lons):

                    obs_point = (obs_lat, obs_lon)
                    pred_point = (pred_lat, pred_lon)

                    distance = geodesic(obs_point, pred_point).kilometers
                    distances.append(distance)

        if distances:
            rmse = np.sqrt(np.mean(np.square(distances)))
            rmse_over_time.append(rmse)
        else:
            rmse_over_time.append(np.nan)  # case where no TCs exist at this time

    return rmse_over_time


track_rmse = track_rmse(obs_tracks, pred_tracks)
plt.figure(figsize=(10, 6))
plt.plot(track_rmse, marker='o', linestyle='-', color='b')
plt.title('RMSE Over Time for TC Tracks')
plt.xlabel('Timestep')
plt.ylabel('RMSE (km)')
plt.grid(True)
plt.savefig("/scratch/ll44/sc6160/out/plots/summary/track_rmse.png")



# intensity
obs_winds = []
pred_winds = []

for name in tc_names:

    tc_file = f"/scratch/ll44/sc6160/tc_data/{name}.json"
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

# def wind_rmse(obs_winds, pred_winds):