import sys
import platform
import os
import glob
import shutil
import cftime
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic
import utils
from subprocess import Popen
from getpass import getpass

tc_file = sys.argv[1]
tc_name, tc_id, start_time, end_time, tc_dir = utils.load_tc_data(tc_file)

ds = xr.open_dataset(f"/scratch/ll44/sc6160/out/rainfall/{tc_name}_rainfall.nc")

precip_density_imerg = ds['obs']
precip_density_graphcast = ds['pred']

latitudes = np.arange(utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1], 0.25)
longitudes = np.arange(utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1], 0.25)

fig = plt.figure(figsize=(20, 6))
gs = fig.add_gridspec(1, 3, hspace=0, wspace=0.1)
proj = ccrs.PlateCarree()
axes = [fig.add_subplot(gs[i], projection=proj) for i in range(3)]
ax1, ax2, ax3 = axes

for ax in axes:
    ax.set_extent([utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1], 
                   utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1]], 
                  crs=proj)

max_value = max(np.max(precip_density_imerg), np.max(precip_density_graphcast))
if max_value == 0:
    max_value = 1  # fallback if both datasets are empty

# IMERG
im1 = ax1.pcolormesh(longitudes, latitudes, precip_density_imerg, 
                     shading='auto', cmap='Blues', vmin=0, vmax=max_value,
                     transform=proj)
ax1.coastlines(linewidth=0.5)
ax1.set_title('Observed', pad=5)

# GraphCast
im2 = ax2.pcolormesh(longitudes, latitudes, precip_density_graphcast, 
                     shading='auto', cmap='Blues', vmin=0, vmax=max_value,
                     transform=proj)
ax2.coastlines(linewidth=0.5)
ax2.set_title('Predicted', pad=5)

# difference
precip_density_difference = precip_density_imerg - precip_density_graphcast
max_diff = max(abs(precip_density_difference.min()), abs(precip_density_difference.max()))
if max_diff == 0:
    max_diff = 1  # fallback if difference is zero everywhere

im3 = ax3.pcolormesh(longitudes, latitudes, precip_density_difference, 
                     shading='auto', cmap='RdBu_r', 
                     vmin=-max_diff, vmax=max_diff,
                     transform=proj)
ax3.coastlines(linewidth=0.5)
ax3.set_title('Difference', pad=5)

for ax, im in zip(axes, [im1, im2, im3]):
    cax = fig.add_axes([ax.get_position().x0, 
                        ax.get_position().y0 - 0.06,
                        ax.get_position().width, 
                        0.02])
    plt.colorbar(im, cax=cax, orientation='horizontal')

plt.savefig(f"/scratch/ll44/sc6160/out/plots/rainfall/{tc_name}_rainfall.png", bbox_inches='tight', pad_inches=0.1)