import sys
import xarray as xr
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.colors as mcolors
import metpy.plots as mp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic

import utils

tc_file = sys.argv[1]
tc_name, tc_id, start_time, end_time, tc_dir = utils.load_tc_data(tc_file)

ds = None
try:
    ds = xr.open_dataset(f"/scratch/ll44/sc6160/out/tracks/{tc_name}_{tc_id}_tracks.nc")
except:
    print(f"No data for {tc_name} yet.")
    sys.exit(0)

def plot_track(ax, lons, lats, color, title):
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.set_extent(utils.AUS_LON_BOUNDS + utils.AUS_LAT_BOUNDS, crs=ccrs.PlateCarree())
    ax.plot(lons, lats, marker='o', color=color, markersize=7, linestyle='-', linewidth=2, transform=ccrs.PlateCarree())
    ax.set_title(title, fontsize=14)

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 8), subplot_kw={'projection': ccrs.PlateCarree()})
plt.subplots_adjust(wspace=0.05, left=0.05, right=0.95, top=0.90, bottom=0.05)

ax_trac, ax_obs, ax_pred = axes

plot_track(ax_trac,
           ds['ibtracs_lons'].values,
           ds['ibtracs_lats'].values,
           'green',
           'IBTrACS')
plot_track(ax_obs,
           ds['obs_lons'].values,
           ds['obs_lats'].values,
           'blue',
           'Track w/ ERA5 data')
plot_track(ax_pred,
           ds['gc_lons'].values,
           ds['gc_lats'].values,
           'red',
           'Track w/ GC predicted data')

plt.savefig(f"/scratch/ll44/sc6160/out/plots/tracks/{tc_name}_{tc_id}_tracks.png")