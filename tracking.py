from datetime import datetime, timedelta

import xarray as xr
import numpy as np
import geopy.distance
from scipy.ndimage import minimum_filter, maximum_filter
from metpy.units import units, concatenate
import metpy.calc as mpcalc

import utils

def gc_track(preds, start_lat, start_lon, start_time, ref_time):

    mask_data = xr.open_dataset("/scratch/ll44/sc6160/data/land_sea_mask.nc")

    current_lat, current_lon = start_lat, start_lon
    track_lats, track_lons = [], []

    num_time_steps = 1
    max_time_steps = 60
    pred_end_time = None

    while True:

        track_lats.append(current_lat)
        track_lons.append(current_lon)

        time = start_time + (num_time_steps * utils.TIME_STEP)
        time_delta = utils.datetime_to_timedelta(time, ref_time)

        if time_delta not in preds['time'].values:
            break

        mslp_deg = 1

        mslp = utils.get_data_region(preds['mean_sea_level_pressure'], time_delta, current_lat, current_lon, mslp_deg)
        mslp_mask = (mslp == minimum_filter(mslp, size=5))
        mslp_minima_coords = np.column_stack(np.where(mslp_mask))
        mslp_minima_lats = mslp.lat[mslp_minima_coords[:, 0]]
        mslp_minima_lons = mslp.lon[mslp_minima_coords[:, 1]]
        
        # loop through locations by distance ascending
        mslp_distances = [geopy.distance.distance((current_lat, current_lon), \
                                                  (min_lat, min_lon))
                        for min_lat, min_lon in zip(mslp_minima_lats, mslp_minima_lons)]
        mslp_distances = np.argsort(mslp_distances)

        criteria_met = False

        for idx in mslp_distances:

            mslp_lat = mslp_minima_lats[idx]
            mslp_lon = mslp_minima_lons[idx]

            # vorticity check
            vort_deg = 2.5
            vort = calc_vort(utils.get_data_region(preds.sel(level=850), time_delta, mslp_lat, mslp_lon, vort_deg))
            
            vort_threshold = -3.5e-5
            vort_threshold_limit = -1e-5
            vort_threshold_decay = 0.5e-5

            if utils.is_on_land(mask_data, current_lat, current_lon):
                vort_threshold = -2e-5

            while np.all(vort >= vort_threshold):
                vort_threshold += vort_threshold_decay
                if vort_threshold > vort_threshold_limit:
                    continue

            # if all criteria met this is a valid position for the track
            criteria_met = True

            current_lat = mslp_lat
            current_lon = mslp_lon

            break

        pred_end_time = time
        num_time_steps += 1

        # if no locations meet the criteria, the TC is over
        if not criteria_met:
            print(f"No location met TC criteria at time {time}, tracking stopped.")
            break

        if num_time_steps >= max_time_steps:
            break

    return track_lats, track_lons, pred_end_time

def calc_vort(data):

    u = data['u_component_of_wind'].drop_vars(['level', 'time'])
    v = data['v_component_of_wind'].drop_vars(['level', 'time'])

    lat = data['lat']
    lon = data['lon']

    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
    
    dudy = mpcalc.first_derivative(u, delta=dy, axis=0)
    dvdx = mpcalc.first_derivative(v, delta=dx, axis=1)

    vort = dvdx - dudy

    return xr.DataArray(
        vort.magnitude,
        coords=[lat, lon],
        dims=['lat', 'lon']
    )