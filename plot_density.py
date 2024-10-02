import plotting
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

import utils

ds = xr.open_dataset("/scratch/ll44/sc6160/out/density/density.nc")

obs_density = ds['obs']
pred_density = ds['pred']

lats = np.arange(utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1], 0.25)
lons = np.arange(utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1], 0.25)

def plot(density, filename):

    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    pcm = ax.pcolormesh(lons, lats, density, shading='auto', cmap='jet')

    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.set_extent(utils.AUS_LON_BOUNDS + utils.AUS_LAT_BOUNDS, crs=ccrs.PlateCarree())

    plt.colorbar(pcm, ax=ax, orientation='vertical', label='Density')
    plt.title("")
    plt.savefig(f"/scratch/ll44/sc6160/out/plots/density/{filename}.png", dpi=300, bbox_inches='tight')

plot(obs_density, "obs_density")
plot(pred_density, "pred_density")
plot(obs_density - pred_density, "diff_density")