from datetime import datetime, timedelta

import xarray as xr
import numpy as np
import geopy.distance
from scipy.ndimage import minimum_filter, maximum_filter
from metpy.units import units, concatenate
import metpy.calc as mpcalc

import utils

def gc_track_multiple_tcs(preds, start_lat, start_lon, start_time, end_time, initial_radius=5.0):

    tc_tracks = {}  # each TC will be tracked as {tc_id: {'lats': [], 'lons': []}}
    active_tcs = []  # list of currently active TCs by their tc_id
    next_tc_id = 0

    n_steps = int((end_time - start_time) / utils.TIME_STEP)
    time_steps = [start_time + i * utils.TIME_STEP for i in range(1, n_steps)]

    for time in time_steps:

        time_delta = utils.datetime_to_timedelta(time, start_time)

        # check for new TCs within the search region at this time step
        mslp_deg = initial_radius # adjustable search radius for detecting new TCs
        mslp = get_data_region(preds['mean_sea_level_pressure'], time_delta, start_lat, start_lon, mslp_deg)
        mslp_mask = (mslp == minimum_filter(mslp, size=5))
        mslp_minima_coords = np.column_stack(np.where(mslp_mask))
        mslp_minima_lats = mslp.lat[mslp_minima_coords[:, 0]]
        mslp_minima_lons = mslp.lon[mslp_minima_coords[:, 1]]

        # loop through each detected local minima and attempt to track it as a new TC if it meets criteria
        for lat, lon in zip(mslp_minima_lats, mslp_minima_lons):

            # vorticity check
            vort_deg = 2.5
            vort = calc_vort(get_data_region(preds.sel(level=850), time_delta, lat, lon, vort_deg))
            vort_threshold = -5e-5
            if np.all(vort >= vort_threshold):
                continue

            # if it passes, track this as a new cyclone
            tc_tracks[next_tc_id] = {'lats': [lat], 'lons': [lon]}
            active_tcs.append(next_tc_id)
            next_tc_id += 1

        # update positions for all existing cyclones (active_tcs)
        for tc_id in active_tcs:

            current_lat = tc_tracks[tc_id]['lats'][-1]
            current_lon = tc_tracks[tc_id]['lons'][-1]

            # estimate next location based on DLSF
            guess_lat, guess_lon = first_guess_w_wind(
                                preds['u_component_of_wind'].sel(time=time_delta),
                                preds['v_component_of_wind'].sel(time=time_delta),
                                current_lat, current_lon)

            # find local minima around the guessed position
            mslp = get_data_region(preds['mean_sea_level_pressure'], time_delta, guess_lat, guess_lon, mslp_deg)
            mslp_mask = (mslp == minimum_filter(mslp, size=5))
            mslp_minima_coords = np.column_stack(np.where(mslp_mask))
            mslp_minima_lats = mslp.lat[mslp_minima_coords[:, 0]]
            mslp_minima_lons = mslp.lon[mslp_minima_coords[:, 1]]

            # find the nearest valid TC position
            mslp_distances = [geopy.distance.distance((guess_lat, guess_lon), (lat, lon)).km
                              for lat, lon in zip(mslp_minima_lats, mslp_minima_lons)]
            sorted_indices = np.argsort(mslp_distances)

            valid_found = False
            for idx in sorted_indices:
                mslp_lat = mslp_minima_lats[idx]
                mslp_lon = mslp_minima_lons[idx]

                # vorticity check
                vort = calc_vort(get_data_region(preds.sel(level=850), time_delta, mslp_lat, mslp_lon, vort_deg))
                if np.all(vort >= vort_threshold):
                    continue  # If it doesn't meet vorticity criteria, skip this location

                # if valid, update the track for this cyclone
                tc_tracks[tc_id]['lats'].append(mslp_lat)
                tc_tracks[tc_id]['lons'].append(mslp_lon)
                valid_found = True
                break

            # if no valid position is found for this TC, consider it dissipated and remove it from active tracking
            if not valid_found:
                active_tcs.remove(tc_id)

    return tc_tracks

def gc_track_end(preds, start_lat, start_lon, start_time, end_time):

    n_steps = int((end_time - start_time) / utils.TIME_STEP)
    time_steps = [start_time + i * utils.TIME_STEP for i in range(1, n_steps)]

    current_lat, current_lon = start_lat, start_lon
    track_lats, track_lons = [], []

    for time in time_steps:

        track_lats.append(current_lat)
        track_lons.append(current_lon)

        time_delta = utils.datetime_to_timedelta(time, start_time)

        # estimate next location based on DLSF
        guess_lat, guess_lon = first_guess_w_wind(
                            preds['u_component_of_wind'].sel(time=time_delta),
                            preds['v_component_of_wind'].sel(time=time_delta),
                            current_lat, current_lon)
        
        # find all local minima of mslp
        mslp_deg = 4 # GC paper: 445km (* 0.5)?
        mslp = get_data_region(preds['mean_sea_level_pressure'], time_delta, guess_lat, guess_lon, mslp_deg)
        mslp_mask = (mslp == minimum_filter(mslp, size=5))
        mslp_minima_coords = np.column_stack(np.where(mslp_mask))
        mslp_minima_lats = mslp.lat[mslp_minima_coords[:, 0]]
        mslp_minima_lons = mslp.lon[mslp_minima_coords[:, 1]]
        
        # loop through locations by distance ascending
        mslp_distances = [geopy.distance.distance((guess_lat, guess_lon), \
                                                  (min_lat, min_lon)).km
                        for min_lat, min_lon in zip(mslp_minima_lats, mslp_minima_lons)]
        mslp_distances = np.argsort(mslp_distances)

        for idx in mslp_distances:

            mslp_lat = mslp_minima_lats[idx]
            mslp_lon = mslp_minima_lons[idx]

            # vorticity check
            vort_deg = 2.5 # GC paper uses 278km radius
            vort = calc_vort(get_data_region(preds.sel(level=850), time_delta, mslp_lat, mslp_lon, vort_deg))
            vort_threshold = -3.5e-5 # note that the GC paper uses -5e-5
            if np.all(vort >= vort_threshold):
                continue

            # if all criteria met this is a valid position for the track
            current_lat = mslp_lat
            current_lon = mslp_lon
            break

    return track_lats, track_lons


def gc_track(preds, start_lat, start_lon, start_time):

    current_lat, current_lon = start_lat, start_lon
    track_lats, track_lons = [], []

    num_time_steps = 1
    max_time_steps = 25
    pred_end_time = None

    while True:

        track_lats.append(current_lat)
        track_lons.append(current_lon)

        time = start_time + (num_time_steps * utils.TIME_STEP)
        time_delta = utils.datetime_to_timedelta(time, start_time)

        if time_delta not in preds['time'].values:
            break

        # estimate next location based on DLSF
        guess_lat, guess_lon = first_guess_w_wind(
                            preds['u_component_of_wind'].sel(time=time_delta),
                            preds['v_component_of_wind'].sel(time=time_delta),
                            current_lat, current_lon)
        
        # find all local minima of mslp
        mslp_deg = 4 # GC paper: 445km (* 0.5)?
        mslp = get_data_region(preds['mean_sea_level_pressure'], time_delta, guess_lat, guess_lon, mslp_deg)
        mslp_mask = (mslp == minimum_filter(mslp, size=5))
        mslp_minima_coords = np.column_stack(np.where(mslp_mask))
        mslp_minima_lats = mslp.lat[mslp_minima_coords[:, 0]]
        mslp_minima_lons = mslp.lon[mslp_minima_coords[:, 1]]
        
        # loop through locations by distance ascending
        mslp_distances = [geopy.distance.distance((guess_lat, guess_lon), \
                                                  (min_lat, min_lon))
                        for min_lat, min_lon in zip(mslp_minima_lats, mslp_minima_lons)]
        mslp_distances = np.argsort(mslp_distances)

        criteria_met = False

        for idx in mslp_distances:

            mslp_lat = mslp_minima_lats[idx]
            mslp_lon = mslp_minima_lons[idx]

            # vorticity check
            vort_deg = 2.5 # GC paper uses 278km radius
            vort = calc_vort(get_data_region(preds.sel(level=850), time_delta, mslp_lat, mslp_lon, vort_deg))
            vort_threshold = -3.5e-5
            if np.all(vort >= vort_threshold):
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
            print("No location met TC criteria, tracking stopped.")
            break

        if num_time_steps >= max_time_steps:
            break

    return track_lats, track_lons, pred_end_time

def calc_vort_old(data):

    u = data['u_component_of_wind'].drop_vars(['level', 'time'])
    v = data['v_component_of_wind'].drop_vars(['level', 'time'])

    lat = data['latitude']
    lon = data['longitude']

    dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
    
    dudy = mpcalc.first_derivative(u, delta=dy, axis=0)
    dvdx = mpcalc.first_derivative(v, delta=dx, axis=1)

    vort = dvdx - dudy

    return xr.DataArray(
        vort.magnitude,
        coords=[lat, lon],
        dims=['lat', 'lon']
    )

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

def get_data_region(data, timedelta, lat, lon, radius):
    return data \
            .sel(time=timedelta, method='nearest') \
            .sel(lat=slice(lat - radius, lat + radius),
                    lon=slice(lon - radius, lon + radius - 0.25)) \
            .squeeze('batch')

def restrict_search_region(data, start_lat, start_lon, radius_deg):
    return data.sel(lat=slice(start_lat - radius_deg, start_lat + radius_deg),
                    lon=slice(start_lon - radius_deg, start_lon + radius_deg))

def find_nearest(latitudes, longitudes, lat, lon):
    lat_idx = (np.abs(latitudes - lat)).argmin()
    lon_idx = (np.abs(longitudes - lon)).argmin()
    return lat_idx, lon_idx

def first_guess_w_wind(u_wind, v_wind, lat, lon):

    # extract the region around the current position
    u_region = restrict_search_region(u_wind, lat, lon, 3).squeeze('batch').drop_vars('time')
    v_region = restrict_search_region(v_wind, lat, lon, 3).squeeze('batch').drop_vars('time')

    # calculate deep layer steering flow (DLSF)
    _, u_dlsf, v_dlsf = calc_dlsf(u_region,
                                     v_region,
                                     u_region['lat'],
                                     u_region['lon'],
                                     u_region['level'])

    ilat, ilon = find_nearest(u_region['lat'], u_region['lon'], lat, lon)
    
    u_current = u_dlsf[ilat, ilon]
    v_current = v_dlsf[ilat, ilon]

    u_deg_per_sec = u_current / (utils.EARTH_RADIUS_M * np.cos(np.deg2rad(lat)))
    v_deg_per_sec = v_current / utils.EARTH_RADIUS_M

    time_step_seconds = 21600  # 6 hours in seconds
    delta_lon = u_deg_per_sec * time_step_seconds
    delta_lat = v_deg_per_sec * time_step_seconds

    pred_lat = lat + np.rad2deg(delta_lat)
    pred_lon = lon + np.rad2deg(delta_lon)

    return pred_lat, pred_lon

# Linh's code snippets: (*cite*)

def calc_dlsf(ulev, vlev, lat, lon, lev):

    u_dlsf, v_dlsf = (np.full([len(lat), len(lon)], np.nan) for _ in range(2))

    for ilat in range (len(lat)):
        for ilon in range (len(lon)):
            
            windmean = concatenate(
                mpcalc.mean_pressure_weighted(
                    lev * units.hPa,
                    ulev.sel(lat=ilat, lon=ilon, method='nearest') * units('knot/second'), \
                    vlev.sel(lat=ilat, lon=ilon, method='nearest') * units('knot/second'),
                    bottom=850 * units.hPa, depth=200 * units.hPa)).magnitude
            
            u_dlsf[ilat, ilon] = windmean[0]
            v_dlsf[ilat, ilon] = windmean[1]

    wspd_dlsf = calc_wspeed(u_dlsf, v_dlsf)

    return wspd_dlsf, u_dlsf, v_dlsf

def calc_wspeed(u, v):
    wspd = np.sqrt(u*u + v*v)
    return wspd