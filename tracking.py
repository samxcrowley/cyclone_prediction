from datetime import datetime, timedelta

import xarray as xr
import numpy as np
import geopy.distance
from scipy.ndimage import minimum_filter, maximum_filter
from metpy.units import units, concatenate
import metpy.calc as mpcalc

import utils

def lazy_track(start_lat, start_lon, start_time, end_time):

    n_steps = int((end_time - start_time) / utils.TIME_STEP)
    time_steps = [start_time + i * utils.TIME_STEP for i in range(n_steps)]

    current_lat, current_lon = start_lat, start_lon
    track_lats, track_lons = [], []

    for time in time_steps:

        track_lats.append(current_lat)
        track_lons.append(current_lon)

        # C1: vorticity
        vort_deg = 0.5
        vort = utils.load_pl_var(time.year, time.month, time, 'vo') \
                .sel(latitude=slice(current_lat + vort_deg, current_lat - vort_deg),
                    longitude=slice(current_lon - vort_deg, current_lon + vort_deg),
                    level=850)

        max_vorticity = vort.max()
        max_vorticity_coords = vort.where(vort == max_vorticity, drop=True)
        max_vort_loc = list(zip(max_vorticity_coords['latitude'].values, max_vorticity_coords['longitude'].values))
        
        max_vort_lat, max_vort_lon = max_vort_loc[0]
        max_vort_lat = float(max_vort_lat)
        max_vort_lon = float(max_vort_lon)

        # C2: mslp
        mslp_deg = 4
        mslp = utils.load_sl_var(time.year, time.month, time, 'msl') \
                .sel(latitude=slice(max_vort_lat + mslp_deg, max_vort_lat - mslp_deg),
                    longitude=slice(max_vort_lon - mslp_deg, max_vort_lon + mslp_deg))
        min_mslp_loc = mslp['msl'].argmin(dim=['latitude', 'longitude'])
        min_mslp_lat = mslp['latitude'][min_mslp_loc['latitude']]
        min_mslp_lon = mslp['longitude'][min_mslp_loc['longitude']]

        storm_center_lat = min_mslp_lat
        storm_center_lon = min_mslp_lon

        # C3: temperature
        temp_deg = 2
        temp_region = utils.load_pl_var(time.year, time.month, time, 't') \
                .sel(latitude=slice(storm_center_lat + temp_deg, storm_center_lat - temp_deg),
                    longitude=slice(storm_center_lon - temp_deg, storm_center_lon + temp_deg),
                    level=slice(200, 500))
        avg_temp = temp_region.mean(dim='level')['t']
        
        # find all local maxima of temp.
        temp_mask = (avg_temp == maximum_filter(avg_temp, size=5))

        temp_maxima_coords = np.column_stack(np.where(temp_mask))
        temp_maxima_lats = temp_region.latitude[temp_maxima_coords[:, 0]]
        temp_maxima_lons = temp_region.longitude[temp_maxima_coords[:, 1]]

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
                            utils.load_pl_var(time.year, time.month, time, 'u')['u'],
                            utils.load_pl_var(time.year, time.month, time, 'v')['v'],
                            current_lat, current_lon)

        distance_to_first_guess = np.sqrt((warm_core_lat - pred_lat) ** 2 + (warm_core_lon - pred_lon) ** 2)
        if distance_to_first_guess <= 350:
            current_lat, current_lon = warm_core_lat, warm_core_lon
        else: # TO-DO: TC can disappear for some time
            print(f"First guess at time {time} was further than 350km away")
            break

    return track_lats, track_lons

def restrict_search_region(data, start_lat, start_lon, radius_deg):
    return data.sel(latitude=slice(start_lat + radius_deg, start_lat - radius_deg),
                    longitude=slice(start_lon - radius_deg, start_lon + radius_deg))

def find_nearest(latitudes, longitudes, lat, lon):
    lat_idx = (np.abs(latitudes - lat)).argmin()
    lon_idx = (np.abs(longitudes - lon)).argmin()
    return lat_idx, lon_idx

def first_guess_w_wind(u_wind, v_wind, lat, lon):

    # extract the region around the current position
    u_region = restrict_search_region(u_wind, lat, lon, 3)
    v_region = restrict_search_region(v_wind, lat, lon, 3)

    # calculate deep layer steering flow (DLSF)
    wspd, u_dlsf, v_dlsf = calc_dlsf(u_region,
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

# def track_from_start_and_end(data, start_lat, start_lon, start_time, end_time, ref_time):

#     n_steps = int((end_time - start_time) / utils.TIME_STEP)
#     time_steps = [start_time + i * utils.TIME_STEP for i in range(0, n_steps)]
    
#     ###
#     # for debugging:
#     # time_steps = time_steps[0:10]
#     # print(time_steps)
#     ###

#     track_lons = []
#     track_lats = []

#     current_lon, current_lat = start_lon, start_lat

#     track_lons.append(current_lon)
#     track_lats.append(current_lat)

#     criterion_3b = False
#     crtierion_4b = False
    
#     for time in time_steps:

#         print(time)

#         time = utils.datetime_to_timedelta(time, ref_time)

#         print("Current", current_lat, current_lon)

#         # criterion 1: local maximum vorticity > 3.5e-5 s^-1 at 850 hPa
#         vort_region_deg = 3

#         # vort_region = data['relative_vorticity'] \
#         #     .sel(lat=slice(current_lat - vort_region_deg, current_lat + vort_region_deg),
#         #          lon=slice(current_lon - vort_region_deg, current_lon + vort_region_deg))

#         vort_region = data \
#             .sel(lat=slice(current_lat - vort_region_deg, current_lat + vort_region_deg),
#                  lon=slice(current_lon - vort_region_deg, current_lon + vort_region_deg)) \
#             .sel(time=time, method='nearest') \
#             .sel(level=850)
#         vorticity = calculate_vorticity(vort_region)

#         vorticity_threshold = 3.5e-5 #3.5e-5
#         vorticity_mask = vort_region.where(vorticity > vorticity_threshold, drop=True)

#         # skip if no significant vorticity found
#         if vorticity_mask.isnull().all():
#             print(f"No significant vorticity found at timestep {time}")
#             continue

#         max_vort_idx = vorticity_mask.argmax(dim=['lat', 'lon'])

#         max_vort_lat = vorticity_mask['lat'].isel(lat=max_vort_idx['lat']).item()
#         max_vort_lon = vorticity_mask['lon'].isel(lon=max_vort_idx['lon']).item()

#         # criterion 2: identify storm center as closest local MSLP within 8° of vorticity maximum
#         mslp_region_deg = 8
#         mslp_region = data.sel(time=time, method='nearest') \
#             .sel(lat=slice(max_vort_lat - mslp_region_deg, max_vort_lat + mslp_region_deg), \
#                  lon=slice(max_vort_lon - mslp_region_deg, max_vort_lon + mslp_region_deg))
#         mslp = mslp_region['mean_sea_level_pressure']

#         # find all local minima of mslp
#         mslp_mask = (mslp == minimum_filter(mslp, size=3))

#         mslp_minima_coords = np.column_stack(np.where(mslp_mask))
#         mslp_minima_lats = mslp_region.lat[mslp_minima_coords[:, 1]]
#         mslp_minima_lons = mslp_region.lon[mslp_minima_coords[:, 2]]

#         mslp_distances = [geopy.distance.distance((max_vort_lat, max_vort_lon), (min_lat, min_lon)).km 
#                     for min_lat, min_lon in zip(mslp_minima_lats, mslp_minima_lons)]

#         mslp_closest_idx = np.argmin(mslp_distances)

#         storm_center_lat = mslp_minima_lats[mslp_closest_idx].item()
#         storm_center_lon = mslp_minima_lons[mslp_closest_idx].item()

#         # criterion 3: identify warm-core center based on 200-500 hPa average temperature within 2° of storm center
#         temp_region_deg = 2
#         temp_region = data.sel(time=time, method='nearest') \
#             .sel(lat=slice(storm_center_lat - temp_region_deg, storm_center_lat + temp_region_deg), \
#                  lon=slice(storm_center_lon - temp_region_deg, storm_center_lon + temp_region_deg), \
#                  level=slice(200, 500))
#         avg_temp = temp_region['temperature'].mean(dim='level')
        
#         # find all local maxima of temp.
#         temp_mask = (avg_temp == maximum_filter(avg_temp, size=3))
        
#         temp_maxima_coords = np.column_stack(np.where(temp_mask))

#         temp_maxima_lats = temp_region.lat[temp_maxima_coords[:, 1]]
#         temp_maxima_lons = temp_region.lon[temp_maxima_coords[:, 2]]

#         temp_distances = [geopy.distance.distance((storm_center_lat, storm_center_lon), \
#                                                   (max_lat, max_lon)).km 
#                     for max_lat, max_lon in zip(temp_maxima_lats, temp_maxima_lons)]

#         temp_closest_idx = np.argmin(temp_distances)

#         warm_core_lat = temp_maxima_lats[temp_closest_idx].item()
#         warm_core_lon = temp_maxima_lons[temp_closest_idx].item()

#         # criterion 4: Identify closest local maximum of 200-1000 hPa thickness within 2° of storm center
#         # thickness = region['geopotential'].sel(level=200).sel(time=time) - data['geopotential'].sel(level=1000).sel(time=time)
#         # thickness_anomaly = thickness - thickness.mean()
#         # thickness_loc = thickness_anomaly.sel(lat=slice(storm_center_lat - 2, storm_center_lat + 2), \
#         #                                       lon=slice(storm_center_lon - 2, storm_center_lon + 2)).argmax(dim=['lat', 'lon'])
#         # thickness_lat = thickness_loc['lat'].values[0]
#         # thickness_lon = thickness_loc['lon'].values[0]

#         # Check if thickness anomaly is at least 5 meters
#         # if thickness_anomaly.sel(lat=thickness_lat, lon=thickness_lon) < 5:
#         #     continue  # Skip if thickness anomaly is insufficient

#         # use the first guess from the steering wind and find the TC within 350 km of this guess
#         pred_lat, pred_lon = first_guess_w_wind(data, current_lon, current_lat, time)

#         print("Pred coords", pred_lat, pred_lon)
#         print("Storm center coords", storm_center_lat, storm_center_lon)
#         print("Warm core coords", warm_core_lat, warm_core_lon)

#         warm_core_lat = storm_center_lat
#         warm_core_lon = storm_center_lon

#         # choose the warm-core TC closest to the first guess and within 350 km
#         distance_to_first_guess = np.sqrt((warm_core_lat - pred_lat) ** 2 + (warm_core_lon - pred_lon) ** 2)
#         if distance_to_first_guess <= 350:
#             current_lat, current_lon = warm_core_lat, warm_core_lon

#         else: # TO-DO: TC can disappear for some time
#             print(f"First guess at time {time} was further than 350km away")
#             break

#         track_lats.append(current_lat)
#         track_lons.append(current_lon)

#         print()

#     # if not criterion_3b or not crtierion_4b:
#     #     return [], []
    
#     return track_lats, track_lons