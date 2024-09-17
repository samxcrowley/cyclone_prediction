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
    time_steps = [start_time + i * utils.TIME_STEP for i in range(1, n_steps)]

    current_lat, current_lon = start_lat, start_lon
    track_lats, track_lons = [], []

    tc_pred_end_time = None

    for time in time_steps[:5]:

        track_lats.append(current_lat)
        track_lons.append(current_lon)

        guess_lat, guess_lon = first_guess_w_wind(
                            utils.load_pl_var(time.year, time.month, time, 'u')['u'],
                            utils.load_pl_var(time.year, time.month, time, 'v')['v'],
                            current_lat, current_lon)

        mslp_deg = 4
        mslp = utils.load_sl_var(time.year, time.month, time, 'msl') \
                .sel(latitude=slice(guess_lat + mslp_deg, guess_lat - mslp_deg),
                    longitude=slice(guess_lon - mslp_deg, guess_lon + mslp_deg))

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
            # vort_deg = 2.5 # GC paper uses 278km radius
            # vort = utils.load_pl_var(time.year, time.month, time, 'vo') \
            #     .sel(latitude=slice(current_lat + vort_deg, current_lat - vort_deg),
            #         longitude=slice(current_lon - vort_deg, current_lon + vort_deg),
            #         level=850)
            # vort_threshold = -3.5e-5 # note that the GC paper uses -5e-5
            # if np.all(vort['vo'] >= vort_threshold):
            #     continue

            # if all criteria met this is a valid position for the track
            current_lat = mslp_lat
            current_lon = mslp_lon
            
            break

        tc_pred_end_time = time

    return track_lats, track_lons, tc_pred_end_time

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