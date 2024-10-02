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

file_prefix = f"/scratch/ll44/sc6160/data/IMERG/{tc_name}/3B"
files = glob.glob(f"{file_prefix}*")

datasets = [xr.open_dataset(f) for f in files]
ds = xr.concat(datasets, dim='time')
ds = ds.interp(lat=latitudes, lon=longitudes)

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

    time = time_steps[i]
    time_cf = cftime.DatetimeJulian(time.year, time.month, time.day, time.hour, time.minute, time.second)

    next_time = time_steps[i] + pd.Timedelta(hours=3)
    next_time_cf = cftime.DatetimeJulian(next_time.year, next_time.month, next_time.day, next_time.hour, next_time.minute, next_time.second)

    lon_radius_deg = radius / (111 * np.cos(np.radians(lat)))

    rainfall = ds.sel(time=time_cf) \
                .sel(lat=slice(lat - lat_radius_deg, lat + lat_radius_deg),
                     lon=slice(lon - lon_radius_deg, lon + lon_radius_deg)) \
                     ['precipitationCal'] * 3 # mm/hr -> mm
    next_rainfall = ds.sel(time=next_time_cf) \
                .sel(lat=slice(lat - lat_radius_deg, lat + lat_radius_deg),
                     lon=slice(lon - lon_radius_deg, lon + lon_radius_deg)) \
                     ['precipitationCal'] * 3 # mm/hr -> mm

    total_rainfall = rainfall.fillna(0) + next_rainfall.fillna(0)

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

    rainfall_gc = preds.sel(time=time) \
                .squeeze('batch') \
                .sel(lat=slice(lat - lat_radius_deg, lat + lat_radius_deg),
                     lon=slice(lon - lon_radius_deg, lon + lon_radius_deg)) \
                     ['total_precipitation_6hr'] * 1000 # m -> mm

    total_rainfall_gc = rainfall_gc.fillna(0)

    result = precip_density_graphcast.sel(lat=total_rainfall_gc.lat, lon=total_rainfall_gc.lon) + total_rainfall_gc
    precip_density_graphcast.loc[dict(lat=total_rainfall_gc.lat, lon=total_rainfall_gc.lon)] = result


# save data
ds = xr.Dataset({
    "obs": precip_density_imerg,
    "pred": precip_density_graphcast
})

ds.to_netcdf(f"/scratch/ll44/sc6160/out/rainfall/{tc_name}_rainfall.nc")