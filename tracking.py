from datetime import datetime, timedelta

import numpy as np
from scipy.spatial import distance
from metpy.units import units, concatenate
import metpy.calc as mpcalc

import utils, plotting

def track_from_start_and_end(data, start_lon, start_lat, start_time, end_time, ref_time):

    start_delta = utils.datetime_to_timedelta(start_time, ref_time)
    end_delta = utils.datetime_to_timedelta(end_time, ref_time)

    n_steps = int((end_time - start_time) / utils.TIME_STEP)
    time_steps = [utils.datetime_to_timedelta(ref_time + i * utils.TIME_STEP, ref_time) \
                   for i in range(1, n_steps)]

    track_lons = []
    track_lats = []

    current_lon, current_lat = start_lon, start_lat
    
    for time in time_steps:
        
        region = restrict_search_region(data, current_lon, current_lat, 350)

        # criterion 1: local maximum vorticity > 3.5e-5 s^-1 at 850 hPa
        vorticity = calculate_vorticity(region, time, 850)
        vorticity_threshold = 3.5e-5
        vorticity_mask = vorticity.where(vorticity > vorticity_threshold, drop=True)

        # skip if no significant vorticity found
        if vorticity_mask.isnull().all():
            continue

        max_vort_loc = vorticity_mask.sel(batch=0)
        max_vort_lat, max_vort_lon = max_vort_loc['lat'].values[0], max_vort_loc['lon'].values[0]

        # criterion 2: identify storm center as closest local MSLP within 8° of vorticity maximum
        mslp = data['mean_sea_level_pressure'].sel(time=time).sel(lat=slice(max_vort_lat-8, max_vort_lat+8), lon=slice(max_vort_lon-8, max_vort_lon+8))
        storm_center_idx = mslp.argmin(dim=['lat', 'lon'])
        storm_center_lat = mslp['lat'].isel(lat=storm_center_idx['lat']).values[0]
        storm_center_lon = mslp['lon'].isel(lon=storm_center_idx['lon']).values[0]

        # criterion 3: identify warm-core center based on 200-500 hPa average temperature within 2° of storm center
        avg_temp = data['temperature'].sel(level=slice(200, 500)).mean(dim='level').sel(time=time)
        warm_core_region = avg_temp.sel(lat=slice(storm_center_lat-2, storm_center_lat+2), lon=slice(storm_center_lon-2, storm_center_lon+2))
        warm_core_idx = warm_core_region.argmax(dim=['lat', 'lon'])
        warm_core_lat = warm_core_region['lat'].isel(lat=warm_core_idx['lat']).values[0]
        warm_core_lon = warm_core_region['lon'].isel(lon=warm_core_idx['lon']).values[0]

        # temp_anomaly = avg_temp - avg_temp.mean()

        # warm_core_region = avg_temp.sel(lat=slice(storm_center_lat-2, storm_center_lat+2), lon=slice(storm_center_lon-2, storm_center_lon+2))
        # warm_core_idx = warm_core_region.argmax(dim=['lat', 'lon'])
        # warm_core_lat = warm_core_region['lat'].isel(lat=warm_core_idx['lat']).values[0]
        # warm_core_lon = warm_core_region['lon'].isel(lon=warm_core_idx['lon']).values[0]

        ## Criterion 3b: Check if temperature anomaly is at least 0.5°C
        # if temp_anomaly.sel(lat=warm_core_lat, lon=warm_core_lon) < 0.5:
        #     print("temp_anomaly triggered")
        #     continue  # Skip if not a warm-core cyclone

        # Criterion 4: Identify closest local maximum of 200-1000 hPa thickness within 2° of storm center
        # thickness = region['geopotential'].sel(level=200).sel(time=time) - data['geopotential'].sel(level=1000).sel(time=time)
        # thickness_anomaly = thickness - thickness.mean()
        # thickness_loc = thickness_anomaly.sel(lat=slice(storm_center_lat-2, storm_center_lat+2), lon=slice(storm_center_lon-2, storm_center_lon+2)).argmax(dim=['lat', 'lon'])
        # thickness_lat, thickness_lon = thickness_loc['lat'].values[0], thickness_loc['lon'].values[0]

        # # Check if thickness anomaly is at least 5 meters
        # if thickness_anomaly.sel(lat=thickness_lat, lon=thickness_lon) < 5:
        #     continue  # Skip if thickness anomaly is insufficient

        # use the first guess from the steering wind and find the TC within 350 km of this guess
        pred_lat, pred_lon = first_guess_w_wind(data, current_lon, current_lat, time)

        # choose the warm-core TC closest to the first guess and within 350 km
        distance_to_first_guess = np.sqrt((warm_core_lat - pred_lat) ** 2 + (warm_core_lon - pred_lon) ** 2)
        if distance_to_first_guess <= 350:
            current_lat, current_lon = warm_core_lat, warm_core_lon

        track_lats.append(current_lat)
        track_lons.append(current_lon)

    return track_lons, track_lats

def restrict_search_region(data, start_lon, start_lat, radius):

    lon_radius_deg = radius / (utils.EARTH_RADIUS * np.cos(np.radians(start_lat))) * (180 / np.pi)
    lat_radius_deg = radius / utils.EARTH_RADIUS * (180 / np.pi)

    return data.sel(lon=slice(start_lon - lon_radius_deg, start_lon + lon_radius_deg), lat=slice(start_lat - lat_radius_deg, start_lat + lat_radius_deg))

# first guess with steering wind between 200 and 800 hPa
def first_guess_w_wind(data, lon, lat, time):

    steering_wind_u = data['u_component_of_wind'].sel(level=slice(200, 850)).mean(dim='level')
    steering_wind_v = data['v_component_of_wind'].sel(level=slice(200, 850)).mean(dim='level')

    current_u_wind = steering_wind_u.sel(lat=lat, lon=lon, time=time, method='nearest')
    current_v_wind = steering_wind_v.sel(lat=lat, lon=lon, time=time, method='nearest')

    deg_per_sec_lat = current_v_wind / (111320)  # 111.32 km per degree of latitude
    deg_per_sec_lon = current_u_wind / (111320 * np.cos(np.radians(lat)))

    # calculate displacement over 6 hours (in seconds)
    delta_lat = deg_per_sec_lat * 21600
    delta_lon = deg_per_sec_lon * 21600

    return lat + delta_lat, lon + delta_lon

def calculate_vorticity(data, time, level):
    
    u = data['u_component_of_wind'].sel(level=level, time=time)
    v = data['v_component_of_wind'].sel(level=level, time=time)

    lat = u['lat']
    lon = u['lon']

    R = utils.EARTH_RADIUS * 1000  # Earth's radius in meters

    # Calculate the grid spacing in the lat/lon directions
    dy = np.deg2rad(lat.diff('lat')) * R
    dx = np.deg2rad(lon.diff('lon')) * R * np.cos(np.deg2rad(lat))

    # Calculate the partial derivatives
    dvdx = v.diff('lon') / dx
    dudy = u.diff('lat') / dy

    # Ensure vorticity has the correct shape
    vorticity = dvdx - dudy

    # Adjust the coordinate sizes to match the vorticity array
    vorticity = vorticity.assign_coords(lat=lat[1:], lon=lon[1:])

    return vorticity


# Linh's code snippets: (cite*)
def calc_wspeed(u, v):
    wspd = np.sqrt(u * u + v * v)
    return wspd

def calc_dlsf(ulev, vlev, lat, lon, lev, deep):

    u_dlsf, v_dlsf = (np.full([len(lat), len(lon)], np.nan) for k in range(2))

    if deep == '850-200mb':

        for ilat in range (len(lat)):
            for ilon in range (len(lon)): 

                windmean = concatenate(mpcalc.mean_pressure_weighted(lev * units.hPa, ulev[:, ilat, ilon] * units('knot/second'), \
                                    vlev[:, ilat, ilon] * units('knot/second'), bottom=850 * units.hPa, depth=200 * units.hPa)).magnitude
                u_dlsf[ilat,ilon] = windmean[0]
                v_dlsf[ilat,ilon] = windmean[1]

    wspd_dlsf = calc_wspeed(u_dlsf,v_dlsf)

    return wspd_dlsf, u_dlsf, v_dlsf