import sys
import platform
import os
import glob
import shutil
import cftime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic
import utils
from subprocess import Popen
from getpass import getpass

tc_file = sys.argv[1]
tc_name, tc_id, start_time, end_time, tc_dir = utils.load_tc_data(tc_file)

radius = 500
lat_radius_deg = radius / 111
latitudes = np.arange(utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1], 0.25)
longitudes = np.arange(utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1], 0.25)

tracks_ds = xr.open_dataset(f"/scratch/ll44/sc6160/out/tracks/{tc_name}_{tc_id}_tracks.nc")
ibt_lats = tracks_ds['ibtracs_lats']
ibt_lons = tracks_ds['ibtracs_lons']
obs_lats = tracks_ds['obs_lats']
obs_lons = tracks_ds['obs_lons']
pred_lats = tracks_ds['gc_lats']
pred_lons = tracks_ds['gc_lons']

time_steps = pd.date_range(start=start_time, end=end_time, freq='6h')

file_prefix = f"/scratch/ll44/sc6160/data/IMERG/{tc_name}/HTTP"
files = glob.glob(f"{file_prefix}*")

datasets = [xr.open_dataset(f) for f in files]
ds = xr.concat(datasets, dim='time')
ds = ds.interp(lat=latitudes, lon=longitudes)
ds = ds.sortby('time')
ds = ds.resample(time='6h').mean()

precip_density_imerg = xr.DataArray(np.zeros((len(latitudes), len(longitudes))),
                                    coords=[latitudes, longitudes],
                                    dims=["lat", "lon"])
precip_density_graphcast = xr.DataArray(np.zeros((len(latitudes), len(longitudes))),
                                    coords=[latitudes, longitudes],
                                    dims=["lat", "lon"])

# OBS
for i, (lat, lon) in enumerate(zip(ibt_lats, ibt_lons)):

    lat = float(lat.values)
    lon = float(lon.values)

    if i >= len(time_steps):
        break
    time = np.datetime64(time_steps[i])
    if time not in ds['time'].values:
        print(f"{time} not in IMERG dataset for TC {tc_name}.")
        continue

    lon_radius_deg = radius / (111 * np.cos(np.radians(lat)))

    # clip the latitude and longitude values to stay within the bounds
    lat_min = np.clip(lat - lat_radius_deg, utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1])
    lat_max = np.clip(lat + lat_radius_deg, utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1])
    lon_min = np.clip(lon - lon_radius_deg, utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1])
    lon_max = np.clip(lon + lon_radius_deg, utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1])

    rainfall = ds.sel(time=time) \
                .sel(lat=slice(lat_min, lat_max),
                     lon=slice(lon_min, lon_max)) \
                     ['precipitationCal'] * 6 # mm/hr -> mm
    total_rainfall = rainfall.fillna(0)

    result = precip_density_imerg.sel(lat=total_rainfall.lat, lon=total_rainfall.lon) + total_rainfall
    precip_density_imerg.loc[dict(lat=total_rainfall.lat, lon=total_rainfall.lon)] = result


# PRED
preds = xr.open_dataset(f"/scratch/{tc_dir}/sc6160/out/pred/{tc_name}_{tc_id}_pred_data.nc")
times = preds['time'].values
for i, (lat, lon) in enumerate(zip(pred_lats, pred_lons)):

    lat = float(lat.values)
    lon = float(lon.values)

    if i >= len(times):
        break

    time = times[i]

    lon_radius_deg = radius / (111 * np.cos(np.radians(lat)))

    lat_min = np.clip(lat - lat_radius_deg, utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1])
    lat_max = np.clip(lat + lat_radius_deg, utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1])
    lon_min = np.clip(lon - lon_radius_deg, utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1])
    lon_max = np.clip(lon + lon_radius_deg, utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1])

    rainfall_gc = preds.sel(time=time) \
                .squeeze('batch') \
                .sel(lat=slice(lat_min, lat_max),
                     lon=slice(lon_min, lon_max)) \
                     ['total_precipitation_6hr'] * 1000 # m -> mm
    rainfall_gc = xr.where(rainfall_gc < 0, 0, rainfall_gc)

    total_rainfall_gc = rainfall_gc.fillna(0)

    result = precip_density_graphcast.sel(lat=total_rainfall_gc.lat, lon=total_rainfall_gc.lon) + total_rainfall_gc
    precip_density_graphcast.loc[dict(lat=total_rainfall_gc.lat, lon=total_rainfall_gc.lon)] = result

# save data
ds = xr.Dataset({
    "obs": precip_density_imerg,
    "pred": precip_density_graphcast
})

ds.to_netcdf(f"/scratch/ll44/sc6160/out/rainfall/{tc_name}_rainfall.nc")