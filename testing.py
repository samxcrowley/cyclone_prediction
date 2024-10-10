import os, sys
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import radians, sin, cos, sqrt, atan2
import seaborn as sns
from geopy.distance import great_circle
import utils

obs_tracks = []
pred_tracks = []

# tc_names = utils.get_all_tc_names()
tc_names = utils.get_all_good_tc_names()

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

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    r = 6371 # Radius of earth in kilometers
    return c * r

# tracks
def calculate_rmse_over_time(obs_tracks, pred_tracks):

    # Determine the maximum length of tracks
    max_timesteps = max(len(track['lats']) for track in obs_tracks + pred_tracks)
    
    # Initialize arrays to store errors
    all_errors = [[] for _ in range(max_timesteps)]
    
    # Calculate errors for each pair of tracks
    for obs, pred in zip(obs_tracks, pred_tracks):
        for t in range(min(len(obs['lats']), len(pred['lats']))):
            error = geodesic((obs['lats'][t], obs['lons'][t]), (pred['lats'][t], pred['lons'][t])).km
            all_errors[t].append(error)
    
    # Calculate RMSE for each timestep
    rmse_over_time = []
    num_tracks_over_time = []
    
    for errors in all_errors:
        if errors:  # Only calculate if we have errors for this timestep
            rmse = np.sqrt(np.mean(np.array(errors)**2))
            rmse_over_time.append(rmse)
            num_tracks_over_time.append(len(errors))
        else:
            break  # Stop when we run out of tracks
    
    return rmse_over_time, num_tracks_over_time

def plot_rmse_over_time(rmse_over_time, num_tracks_over_time):
    timesteps = range(len(rmse_over_time))
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot RMSE
    line1 = ax1.plot(timesteps, rmse_over_time, 'b-', label='RMSE')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('RMSE (km)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Plot number of tracks
    ax2 = ax1.twinx()
    line2 = ax2.plot(timesteps, num_tracks_over_time, 'r--', label='Number of Tracks')
    ax2.set_ylabel('Number of Tracks', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    
    plt.title('Track RMSE and Number of Tracks Over Time')
    plt.tight_layout()
    return fig

rmse_over_time, num_tracks = calculate_rmse_over_time(obs_tracks, pred_tracks)
plot_rmse_over_time(rmse_over_time, num_tracks)
plt.savefig("test_rmse.png")