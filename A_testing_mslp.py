from datetime import datetime, timedelta

import xarray as xr
import numpy as np
import pandas as pd
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import metpy.plots as mp
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import plotting
import utils
import tracking
import density_map_testing

######################################

ibtracs = xr.open_dataset("/scratch/ll44/sc6160/data/IBTrACS/IBTrACS.last3years.v04r01.nc")
tc_id = "2022076S10126".encode('utf-8')
tc_data = ibtracs.where(ibtracs["sid"] == tc_id, drop=True)
tc_lats = tc_data['lat']
tc_lons = tc_data['lon']
tc_start_lat = tc_lats.values[0][0]
tc_start_lon = tc_lons.values[0][0]

year = 2022
month = 3
tc_name = "CHARLOTTE"

start_time = datetime(2022, 3, 19, 6)
end_time = datetime(2022, 3, 25, 12)
n_steps = (end_time - start_time) / utils.TIME_STEP
time_steps = [start_time + n * utils.TIME_STEP for n in range(int(n_steps))]

######################################


# track mslp
current_lat, current_lon = tc_start_lat, tc_start_lon
track_lats, track_lons = [], []

for time in time_steps[:10]:

    track_lats.append(current_lat)
    track_lons.append(current_lon)

    mslp_data = utils.load_sl_var(year, month, time, 'msl')
    
    radius = 1

    min_mslp_region = mslp_data['msl'] \
        .sel(latitude=slice(current_lat + radius, current_lat - radius),
             longitude=slice(current_lon - radius, current_lon + radius))
    
    min_mslp_loc = min_mslp_region.argmin(dim=['latitude', 'longitude'])
    
    min_mslp_lat = min_mslp_region['latitude'][min_mslp_loc['latitude']].values
    min_mslp_lon = min_mslp_region['longitude'][min_mslp_loc['longitude']].values

    current_lat = min_mslp_lat
    current_lon = min_mslp_lon

    # plotting.plot_pred_track_with_fields(mslp_data, tc_id,
    #                                      tc_lats, tc_lons,
    #                                      current_lat, current_lon, time)

plotting.plot_tc_track_with_pred(tc_id, tc_name, tc_lats, tc_lons, track_lats, track_lons)