from datetime import datetime, timedelta

import numpy as np
from scipy.spatial import distance

import utils, plotting

def track_from_start_and_end(data, start_lon, start_lat, start_time, end_time, ref_time):

    start_delta = utils.datetime_to_timedelta(start_time, ref_time)
    end_delta = utils.datetime_to_timedelta(end_time, ref_time)

    n_steps = int((end_time - start_time) / utils.TIME_STEP)
    time_steps = [utils.datetime_to_timedelta(ref_time + i * utils.TIME_STEP, ref_time) \
                   for i in range(1, n_steps)]

    track_lons = []
    track_lats = []

    region = restrict_search_region(data, start_lon, start_lat, 350)

    # first_times = region['mean_sea_level_pressure'].time.isel(time=slice(0, 10))
    # print("First few time values (using isel):")
    # print(first_times.values)

    for time in time_steps:

        # predict next location with first guess method based on steering wind
        pred_lat, pred_lon = first_guess_w_wind(data, start_lon, start_lat, time)

        min_mslp_loc = region['mean_sea_level_pressure'].sel(time=time).argmin(dim=['lat', 'lon'])

        # vorticity at 850 hPa level
        vorticity = calculate_vorticity(region, time, 850)
        
        vorticity_threshold = 3.5e-5  # s^-1
        vorticity_mask = vorticity > vorticity_threshold


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

    R = utils.EARTH_RADIUS * 1000  # earth's radius in meters

    # calculate the grid spacing in the lat/lon directions
    dy = np.deg2rad(lat.diff('lat')) * R
    dx = np.deg2rad(lon.diff('lon')) * R * np.cos(np.deg2rad(lat))

    # calculate the partial derivatives
    dvdx = (v.diff('lon') / dx).mean('lon')
    dudy = (u.diff('lat') / dy).mean('lat')

    vorticity = dvdx - dudy

    # assign the coordinate values to the vorticity array
    vorticity = vorticity.assign_coords(lat=lat[:-1], lon=lon[:-1])

    return vorticity