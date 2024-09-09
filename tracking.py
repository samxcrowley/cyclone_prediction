from datetime import datetime, timedelta

import xarray as xr
import numpy as np
import geopy.distance
from scipy.ndimage import minimum_filter, maximum_filter
from metpy.units import units, concatenate
import metpy.calc as mpcalc

import utils

def old_track(start_lat, start_lon, start_time, end_time):

    n_steps = int((end_time - start_time) / utils.TIME_STEP)
    time_steps = [start_time + i * utils.TIME_STEP for i in range(n_steps)]

    current_lat, current_lon = start_lat, start_lon
    track_lats, track_lons = [], []

    for time in time_steps:

        track_lats.append(current_lat)
        track_lons.append(current_lon)

        guess_lat, guess_lon = first_guess_w_wind(
                            utils.load_pl_var(time.year, time.month, time, 'u')['u'],
                            utils.load_pl_var(time.year, time.month, time, 'v')['v'],
                            current_lat, current_lon)

        mslp_deg = 4
        mslp = utils.load_sl_var(time.year, time.month, time, 'msl') \
                .sel(latitude=slice(current_lat + mslp_deg, current_lat - mslp_deg),
                    longitude=slice(current_lon - mslp_deg, current_lon + mslp_deg))
        mslp_mask = (mslp['msl'] == minimum_filter(mslp['msl'], size=5))
        mslp_minima_coords = np.column_stack(np.where(mslp_mask))
        mslp_minima_lats = mslp.latitude[mslp_minima_coords[:, 0]]
        mslp_minima_lons = mslp.longitude[mslp_minima_coords[:, 1]]

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
            vort = utils.load_pl_var(time.year, time.month, time, 'vo') \
                .sel(latitude=slice(current_lat + vort_deg, current_lat - vort_deg),
                    longitude=slice(current_lon - vort_deg, current_lon + vort_deg),
                    level=850)
            vort_threshold = -3.5e-5 # note that the GC paper uses -5e-5
            if np.all(vort['vo'] >= vort_threshold):
                continue

            # if all criteria met this is a valid position for the track
            current_lat = mslp_lat
            current_lon = mslp_lon
            break

    return track_lats, track_lons

def gc_track(preds, start_lat, start_lon, start_time, end_time):

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
    return data.sel(latitude=slice(start_lat + radius_deg, start_lat - radius_deg),
                    longitude=slice(start_lon - radius_deg, start_lon + radius_deg))

def find_nearest(latitudes, longitudes, lat, lon):
    lat_idx = (np.abs(latitudes - lat)).argmin()
    lon_idx = (np.abs(longitudes - lon)).argmin()
    return lat_idx, lon_idx

def first_guess_w_wind(u_wind, v_wind, lat, lon):

    # extract the region around the current position
    u_region = restrict_search_region(u_wind, lat, lon, 3).drop_vars('time')
    v_region = restrict_search_region(v_wind, lat, lon, 3).drop_vars('time')
    # u_region = restrict_search_region(u_wind, lat, lon, 3).squeeze('batch').drop_vars('time')
    # v_region = restrict_search_region(v_wind, lat, lon, 3).squeeze('batch').drop_vars('time')

    # calculate deep layer steering flow (DLSF)
    _, u_dlsf, v_dlsf = calc_dlsf(u_region,
                                     v_region,
                                     u_region['latitude'],
                                     u_region['longitude'],
                                     u_region['level'])

    ilat, ilon = find_nearest(u_region['latitude'], u_region['longitude'], lat, lon)
    
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
                    ulev.sel(latitude=ilat, longitude=ilon, method='nearest') * units('knot/second'), \
                    vlev.sel(latitude=ilat, longitude=ilon, method='nearest') * units('knot/second'),
                    bottom=850 * units.hPa, depth=200 * units.hPa)).magnitude
            
            u_dlsf[ilat, ilon] = windmean[0]
            v_dlsf[ilat, ilon] = windmean[1]

    wspd_dlsf = calc_wspeed(u_dlsf, v_dlsf)

    return wspd_dlsf, u_dlsf, v_dlsf

def calc_wspeed(u, v):
    wspd = np.sqrt(u*u + v*v)
    return wspd



# def lazy_track(preds, start_lat, start_lon, start_time, end_time):

#     n_steps = int((end_time - start_time) / utils.TIME_STEP)
#     time_steps = [start_time + i * utils.TIME_STEP for i in range(1, n_steps)]

#     current_lat, current_lon = start_lat, start_lon
#     track_lats, track_lons = [], []

#     for time in time_steps:

#         track_lats.append(current_lat)
#         track_lons.append(current_lon)

#         timedelta = utils.datetime_to_timedelta(time, start_time)

#         # C1: vorticity
#         vort_deg = 2.5 # GC paper uses 278km radius
#         vort = calc_vort(get_data_region(preds.sel(level=850), timedelta, current_lat, current_lon, vort_deg))
#         vort_threshold = -3.5e-5 # note that the GC paper uses -5e-5
#         vort_threshold_dec = 0.1e-5
#         vort_mask = vort.where(vort < vort_threshold, drop=True)

#         # if no vorticities found > threshold, decrease threshold until found
#         # while vort_mask.size == 0:
#         #     vort_threshold -= vort_threshold_dec
#         #     vort_mask = vort['vo'].where(vort['vo'] >= vort_threshold, drop=True)
        
#         max_vort_lats = vort_mask['lat'].values
#         max_vort_lons = vort_mask['lon'].values

#         vort_distances = [geopy.distance.distance((current_lat, current_lon), \
#                                                   (max_lat, max_lon)).km 
#                     for max_lat, max_lon in zip(max_vort_lats, max_vort_lons)]
#         vort_distance_idx = np.argmin(vort_distances)

#         max_vort_lat = max_vort_lats[vort_distance_idx].item()
#         max_vort_lon = max_vort_lons[vort_distance_idx].item()

#         # C2: mslp
#         mslp_deg = 8
#         mslp = get_data_region(preds['mean_sea_level_pressure'], timedelta, max_vort_lat, max_vort_lon, mslp_deg)

#         # find all local minima of mslp
#         mslp_mask = (mslp == minimum_filter(mslp, size=5))
#         mslp_minima_coords = np.column_stack(np.where(mslp_mask))
#         mslp_minima_lats = mslp.lat[mslp_minima_coords[:, 0]]
#         mslp_minima_lons = mslp.lon[mslp_minima_coords[:, 1]]
        
#         mslp_distances = [geopy.distance.distance((max_vort_lat, max_vort_lon), \
#                                                   (min_lat, min_lon)).km
#                         for min_lat, min_lon in zip(mslp_minima_lats, mslp_minima_lons)]

#         mslp_closest_idx = np.argmin(mslp_distances)

#         storm_center_lat = mslp_minima_lats[mslp_closest_idx].item()
#         storm_center_lon = mslp_minima_lons[mslp_closest_idx].item()

#         # C3: temperature
#         temp_deg = 1
#         temp = preds['temperature'] \
#                 .sel(time=timedelta) \
#                 .sel(lat=slice(storm_center_lat - temp_deg, storm_center_lat + temp_deg),
#                     lon=slice(storm_center_lon - temp_deg, storm_center_lon + temp_deg),
#                     level=slice(200, 500)) \
#                 .squeeze('batch')
#         avg_temp = temp.mean(dim='level')
        
#         # find all local maxima of temp.
#         temp_mask = (avg_temp == maximum_filter(avg_temp, size=5))

#         temp_maxima_coords = np.column_stack(np.where(temp_mask))
#         temp_maxima_lats = temp.lat[temp_maxima_coords[:, 0]]
#         temp_maxima_lons = temp.lon[temp_maxima_coords[:, 1]]

#         temp_distances = [geopy.distance.distance((storm_center_lat, storm_center_lon), \
#                                                   (max_lat, max_lon)).km 
#                     for max_lat, max_lon in zip(temp_maxima_lats, temp_maxima_lons)]

#         temp_closest_idx = np.argmin(temp_distances)

#         warm_core_lat = temp_maxima_lats[temp_closest_idx].item()
#         warm_core_lon = temp_maxima_lons[temp_closest_idx].item()

#         print(time)
#         print("Current", current_lat, current_lon)
#         print("Vort", max_vort_lat, max_vort_lon)
#         print("Storm center", storm_center_lat, storm_center_lon)
#         print("Warm core", warm_core_lat, warm_core_lon)
#         print()

#         # choose the warm-core TC closest to the first guess and within 350 km
#         pred_lat, pred_lon = first_guess_w_wind(
#                             preds['u_component_of_wind'].sel(time=timedelta),
#                             preds['v_component_of_wind'].sel(time=timedelta),
#                             current_lat, current_lon)

#         distance_to_first_guess = np.sqrt((warm_core_lat - pred_lat) ** 2 + (warm_core_lon - pred_lon) ** 2)
#         if distance_to_first_guess <= 350:
#             current_lat, current_lon = warm_core_lat, warm_core_lon
#         else: # TO-DO: TC can disappear for some time
#             print(f"First guess at time {time} was further than 350km away")
#             break

#     return track_lats, track_lons