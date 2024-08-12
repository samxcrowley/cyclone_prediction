import os, math
from datetime import datetime, timedelta

import numpy as np
import xarray, matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic

import utils

# track format:
# - {start_time: *, end_time: *, lats: [], lons: []}
def plot_density_map(tracks):

    # earliest = datetime(2100, 1, 1)
    # latest = datetime(1800, 1, 1)

    # for track in tracks:
    #     if track['start_time'] < earliest:
    #         earliest = track['start_time']
    #     if track['end_time'] > latest:
    #         latest = track['end_time']

    # times = [earliest + i * utils.TIME_STEP for i in range(int((latest - earliest) / utils.TIME_STEP) + 1)]

    grid_res = 0.25
    ri = 300

    lats = np.arange(utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1], grid_res)
    lons = np.arange(utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1], grid_res)

    density = np.zeros((len(lats), len(lons)))

    for track in tracks:
        
        tc_lats = track['lats']
        tc_lons = track['lons']

        for lat, lon in zip(tc_lats, tc_lons):

            if np.isnan(lat) or np.isnan(lon):
                continue

            for i, grid_lat in enumerate(lats):
                for j, grid_lon in enumerate(lons):

                    r = geodesic((lat, lon), (grid_lat, grid_lon)).km
                    density[i, j] += impact_factor(r, ri)



    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    pcm = ax.pcolormesh(lons, lats, density, shading='auto', cmap='jet')

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.set_extent(utils.AUS_LON_BOUNDS + utils.AUS_LAT_BOUNDS, crs=ccrs.PlateCarree())

    plt.colorbar(pcm, ax=ax, orientation='vertical', label='Density')
    plt.title('Tropical Cyclone Density Map')
    plt.savefig("/scratch/ll44/sc6160/out/plots/density.png", dpi=300, bbox_inches='tight')

def impact_factor(r, ri):
    if r > ri:
        return 0
    return 1 - (r / ri)