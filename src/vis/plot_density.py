import xarray as xr
import numpy as np
import pandas as pd
import xarray, matplotlib
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.units import units
from matplotlib import animation
import matplotlib.colors as mcolors
from metpy.plots.declarative import (BarbPlot, ContourPlot,
FilledContourPlot, MapPanel, PanelContainer, PlotObs)
from metpy.plots.declarative import PlotGeometry
import cartopy.mpl.gridliner as gridliner
import metpy.plots as mp
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import src.utils.utils as utils

ds = xr.open_dataset("/scratch/ll44/sc6160/out/density/density.nc")

obs_density = ds['obs']
pred_density = ds['pred']

lats = np.arange(utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1], 0.25)
lons = np.arange(utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1], 0.25)

# Get common min/max for obs and pred density
vmin = min(obs_density.min(), pred_density.min())
vmax = max(obs_density.max(), pred_density.max())

def plot_density(ax, density, title, cmap='jet', norm=None):
    ax.set_extent(utils.AUS_LON_BOUNDS + utils.AUS_LAT_BOUNDS, crs=ccrs.PlateCarree())
    pcm = ax.pcolormesh(lons, lats, density, shading='auto', cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    ax.set_title(title)
    return pcm

# create the figure and subplots with gridspec for layout control
fig = plt.figure(figsize=(18, 6))
gs = fig.add_gridspec(1, 6, width_ratios=[1, 0.02, 1, 0.02, 1, 0.02], wspace=0.05)

# plot obs_density
ax1 = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
pcm1 = plot_density(ax1, obs_density, 'Observed Density', cmap='jet', norm=mcolors.Normalize(vmin=vmin, vmax=vmax))

# colorbar for obs_density
cbar1 = fig.colorbar(pcm1, ax=ax1, orientation='vertical')
cbar1.set_label('Density')

# plot pred_density
ax2 = fig.add_subplot(gs[2], projection=ccrs.PlateCarree())
pcm2 = plot_density(ax2, pred_density, 'Predicted Density', cmap='jet', norm=mcolors.Normalize(vmin=vmin, vmax=vmax))

# colorbar for pred_density
cbar2 = fig.colorbar(pcm2, ax=ax2, orientation='vertical')
cbar2.set_label('Density')

# plot the difference (obs - pred) with a custom diverging colormap
ax3 = fig.add_subplot(gs[4], projection=ccrs.PlateCarree())
diff_density = obs_density - pred_density

# set symmetric color limits for difference plot
diff_max = np.abs(diff_density).max()
norm_diff = mcolors.TwoSlopeNorm(vmin=-diff_max, vcenter=0, vmax=diff_max)
pcm3 = plot_density(ax3, diff_density, 'Difference (Observed - Predicted)', cmap='RdBu_r', norm=norm_diff)

# colorbar for difference plot
cbar3 = fig.colorbar(pcm3, ax=ax3, orientation='vertical')
cbar3.set_label('Difference')

cbar1.ax.set_position([ax1.get_position().x1 + 0.005, ax1.get_position().y0, 0.01, ax1.get_position().height])
cbar2.ax.set_position([ax2.get_position().x1 + 0.005, ax2.get_position().y0, 0.01, ax2.get_position().height])
cbar3.ax.set_position([ax3.get_position().x1 + 0.005, ax3.get_position().y0, 0.01, ax3.get_position().height])

plt.savefig("/scratch/ll44/sc6160/out/plots/density/density_all.png", dpi=300, bbox_inches='tight')