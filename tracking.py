from datetime import datetime, timedelta

import xarray as xr
import numpy as np
import geopy.distance
from scipy.ndimage import minimum_filter, maximum_filter
from metpy.units import units, concatenate
import metpy.calc as mpcalc

import utils

def calc_vort(data):

    u = data['u_component_of_wind'].squeeze('batch').drop_vars(['level', 'time'])
    v = data['v_component_of_wind'].squeeze('batch').drop_vars(['level', 'time'])

    dudy, dudx = np.gradient(u, axis=(0, 1))
    dvdy, dvdx = np.gradient(v, axis=(0, 1))

    lat = data['lat']
    lon = data['lon']

    dx = np.deg2rad(lon) * utils.EARTH_RADIUS_M * np.cos(np.deg2rad(lat))
    dy = np.deg2rad(lat) * utils.EARTH_RADIUS_M

    vort = (dvdx / dx) - (dudy / dy)

    return xr.DataArray(
        vort,
        coords=[lat, lon],
        dims=['lat', 'lon']
    )

def lazy_track(preds, start_lat, start_lon, start_time, end_time):

    n_steps = int((end_time - start_time) / utils.TIME_STEP)
    time_steps = [start_time + i * utils.TIME_STEP for i in range(1, n_steps)]

    current_lat, current_lon = start_lat, start_lon
    track_lats, track_lons = [], []

    for time in time_steps:

        track_lats.append(current_lat)
        track_lons.append(current_lon)

        dtime = utils.datetime_to_timedelta(time, start_time)

        # C1: vorticity
        vort_deg = 1
        vort = calc_vort(preds \
                .sel(time=dtime, method='nearest') \
                .sel(lat=slice(current_lat - vort_deg, current_lat + vort_deg),
                    lon=slice(current_lon - vort_deg, current_lon + vort_deg - 0.25),
                    level=850)) \
                .squeeze('batch')

        vort_threshold = 3.5e-5
        vort_threshold_dec = 0.1e-5
        vort_mask = vort.where(vort >= vort_threshold, drop=True)

        # if no vorticities found > threshold, decrease threshold until found
        while vort_mask.size == 0:
            vort_threshold -= vort_threshold_dec
            vort_mask = vort['vo'].where(vort['vo'] >= vort_threshold, drop=True)
        
        max_vort_lats = vort_mask['latitude'].values
        max_vort_lons = vort_mask['longitude'].values

        vort_distances = [geopy.distance.distance((current_lat, current_lon), \
                                                  (max_lat, max_lon)).km 
                    for max_lat, max_lon in zip(max_vort_lats, max_vort_lons)]
        vort_distance_idx = np.argmin(vort_distances)

        max_vort_lat = max_vort_lats[vort_distance_idx].item()
        max_vort_lon = max_vort_lons[vort_distance_idx].item()

        # C2: mslp
        mslp_deg = 4
        mslp = preds['mean_sea_level_pressure'] \
                .sel(time=dtime, method='nearest') \
                .sel(lat=slice(max_vort_lat - mslp_deg, max_vort_lat + mslp_deg),
                    lon=slice(max_vort_lon - mslp_deg, max_vort_lon + mslp_deg)) \
                .squeeze('batch')

        # find all local minima of mslp
        mslp_mask = (mslp == minimum_filter(mslp, size=5))
        mslp_minima_coords = np.column_stack(np.where(mslp_mask))
        mslp_minima_lats = mslp.lat[mslp_minima_coords[:, 0]]
        mslp_minima_lons = mslp.lon[mslp_minima_coords[:, 1]]
        
        mslp_distances = [geopy.distance.distance((max_vort_lat, max_vort_lon), \
                                                  (min_lat, min_lon)).km
                        for min_lat, min_lon in zip(mslp_minima_lats, mslp_minima_lons)]

        mslp_closest_idx = np.argmin(mslp_distances)

        storm_center_lat = mslp_minima_lats[mslp_closest_idx].item()
        storm_center_lon = mslp_minima_lons[mslp_closest_idx].item()

        # C3: temperature
        temp_deg = 1
        temp = preds['temperature'] \
                .sel(time=dtime) \
                .sel(lat=slice(storm_center_lat - temp_deg, storm_center_lat + temp_deg),
                    lon=slice(storm_center_lon - temp_deg, storm_center_lon + temp_deg),
                    level=slice(200, 500)) \
                .squeeze('batch')
        avg_temp = temp.mean(dim='level')
        
        # find all local maxima of temp.
        temp_mask = (avg_temp == maximum_filter(avg_temp, size=5))

        temp_maxima_coords = np.column_stack(np.where(temp_mask))
        temp_maxima_lats = temp.lat[temp_maxima_coords[:, 0]]
        temp_maxima_lons = temp.lon[temp_maxima_coords[:, 1]]

        temp_distances = [geopy.distance.distance((storm_center_lat, storm_center_lon), \
                                                  (max_lat, max_lon)).km 
                    for max_lat, max_lon in zip(temp_maxima_lats, temp_maxima_lons)]

        temp_closest_idx = np.argmin(temp_distances)

        warm_core_lat = temp_maxima_lats[temp_closest_idx].item()
        warm_core_lon = temp_maxima_lons[temp_closest_idx].item()

        print(time)
        print("Current", current_lat, current_lon)
        print("Vort", max_vort_lat, max_vort_lon)
        print("Storm center", storm_center_lat, storm_center_lon)
        print("Warm core", warm_core_lat, warm_core_lon)
        print()

        # choose the warm-core TC closest to the first guess and within 350 km
        pred_lat, pred_lon = first_guess_w_wind(
                            preds['u_component_of_wind'].sel(time=dtime),
                            preds['v_component_of_wind'].sel(time=dtime),
                            current_lat, current_lon)

        distance_to_first_guess = np.sqrt((warm_core_lat - pred_lat) ** 2 + (warm_core_lon - pred_lon) ** 2)
        if distance_to_first_guess <= 350:
            current_lat, current_lon = warm_core_lat, warm_core_lon
        else: # TO-DO: TC can disappear for some time
            print(f"First guess at time {time} was further than 350km away")
            break

    return track_lats, track_lons

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
    wspd, u_dlsf, v_dlsf = calc_dlsf(u_region,
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