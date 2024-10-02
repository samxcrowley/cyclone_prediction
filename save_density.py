from datetime import datetime, timedelta
import sys

import xarray as xr
import numpy as np
import pandas as pd
from geopy.distance import geodesic

import plotting, utils

obs_tracks = []
pred_tracks = []

tc_names = utils.get_all_tc_names()

for tc_name in tc_names:

    tc_file = f"/scratch/ll44/sc6160/tc_data/{tc_name}.json"
    if utils.load_tc_data(tc_file) == None:
        continue
    tc_name, tc_id, start_time, end_time, tc_dir = utils.load_tc_data(tc_file)

    ds = None

    try:
        ds = xr.open_dataset(f"/scratch/ll44/sc6160/out/tracks/{tc_name}_{tc_id}_tracks.nc")
    except:
        print(f"Couldn't find a track file for {tc_name}")
        continue

    obs_lats = ds['obs_lats'].values
    obs_lons = ds['obs_lons'].values
    pred_lats = ds['gc_lats'].values
    pred_lons = ds['gc_lons'].values
    
    obs_tracks.append({'lats': obs_lats.tolist(), 'lons': obs_lons.tolist()})
    pred_tracks.append({'lats': pred_lats.tolist(), 'lons': pred_lons.tolist()})

def get_density(tracks):

    grid_res = 0.25
    ri = 300

    lats = np.arange(utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1], grid_res)
    lons = np.arange(utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1], grid_res)

    mid_lats = lats + (grid_res / 2)
    mid_lons = lons + (grid_res / 2)

    density = np.zeros((len(lats), len(lons)))

    for track in tracks:

        tc_lats = np.array(track['lats'])
        tc_lons = np.array(track['lons'])

        valid_indices = ~np.isnan(tc_lats) & ~np.isnan(tc_lons)
        tc_lats = tc_lats[valid_indices]
        tc_lons = tc_lons[valid_indices]

        for lat, lon in zip(tc_lats, tc_lons):

            for i, grid_lat in enumerate(mid_lats):
                for j, grid_lon in enumerate(mid_lons):

                    if np.abs(lat - grid_lat) > (ri / 111) or np.abs(lon - grid_lon) > (ri / 111):
                        continue
                    
                    r = geodesic((lat, lon), (grid_lat, grid_lon)).km
                    
                    if r <= ri:
                        density[i, j] += utils.impact_factor(r, ri)

    return density

# save data
ds = xr.Dataset({"obs": get_density(obs_tracks)})
ds = xr.Dataset({"pred": get_density(pred_tracks)})
ds.to_netcdf("/scratch/ll44/sc6160/out/density/density.nc")