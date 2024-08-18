import os, math, datetime

import numpy as np
import xarray, matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import metpy.plots as mp
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import utils

from typing import Optional

def scale(data: xarray.Dataset, center: Optional[float] = None, robust: bool = False) -> tuple[xarray.Dataset, matplotlib.colors.Normalize, str]:
    
    vmin = np.nanpercentile(data, (2 if robust else 0))
    vmax = np.nanpercentile(data, (98 if robust else 100))

    if center is not None:
        diff = max(vmax - center, center - vmin)
        vmin = center - diff
        vmax = center + diff
    
    return (data, matplotlib.colors.Normalize(vmin, vmax), ("RdBu_r" if center is not None else "viridis"))

def plot_data(
    data: dict[str, xarray.Dataset],
    fig_title: str,
    plot_size: float = 5,
    robust: bool = False,
    cols: int = 4,
    output_dir: str = "/scratch/ll44/sc6160/out/plots",
    output_prefix: str = ""):

    os.makedirs(output_dir, exist_ok=True)

    first_data = next(iter(data.values()))[0]
    max_steps = first_data.sizes.get("time", 1)
    assert all(max_steps == d.sizes.get("time", 1) for d, _, _ in data.values())

    cols = min(cols, len(data))
    rows = math.ceil(len(data) / cols)
    figure = plt.figure(figsize=(plot_size * 2 * cols,  plot_size * rows))
    figure.suptitle(fig_title, fontsize=16)
    figure.subplots_adjust(wspace=0, hspace=0)
    figure.tight_layout()

    images = []
    for i, (title, (plot_data, norm, cmap)) in enumerate(data.items()):
        ax = figure.add_subplot(rows, cols, i+1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        im = ax.imshow(
            plot_data.isel(time=0, missing_dims="ignore"), norm=norm,
            origin="lower", cmap=cmap)
        plt.colorbar(
            mappable=im,
            ax=ax,
            orientation="vertical",
            pad=0.02,
            aspect=16,
            shrink=0.75,
            cmap=cmap,
            extend=("both" if robust else "neither"))
        images.append(im)

    def update(frame):
        
        if "time" in first_data.dims:
            td = datetime.timedelta(microseconds=first_data["time"][frame].item() / 1000)
            figure.suptitle(f"{fig_title}, {td}", fontsize=16)
        else:
            figure.suptitle(fig_title, fontsize=16)
        # for im, (plot_data, norm, cmap) in zip(images, data.values()):
        #     im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))
        for idx, (plot_data, _, _) in enumerate(data.values()):
           
           im = images[idx]
           im.set_data(plot_data.isel(time=frame, missing_dims="ignore"))

           # save frame
           plt.savefig(os.path.join(output_dir, f"{output_prefix}frame_{frame:02d}.png"))

    anim = animation.FuncAnimation(fig=figure, func=update, frames=max_steps, interval=250)
    anim.save(os.path.join(output_dir, f"{output_prefix}animation.gif"), writer="pillow", fps=2)

    plt.close(figure.number)

def plot_metrics(preds, evals, metrics, lat_bounds, lon_bounds):

    for key in metrics:
        data_dict = utils.prepare_data_dict(preds, evals, key, lat_bounds, lon_bounds)
        plot_data(data_dict, key, plot_size=5, robust=True, cols=3, output_prefix=f"metrics/{metrics[key]}")

def plot_tc_track(tc_id, tc_name, tc_lons, tc_lats, aus_bounds=True):

    # create a new figure with a specific size and projection
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # set the bounds to Australia
    if aus_bounds:
        ax.set_extent(utils.AUS_LON_BOUNDS + utils.AUS_LAT_BOUNDS, crs=ccrs.PlateCarree())

    # plot the cyclone track
    ax.plot(tc_lats, tc_lons, marker='o', color='red', markersize=5, linestyle='-', linewidth=2, transform=ccrs.PlateCarree())

    # tc_id = tc_id.decode("utf-8")
    plt.title(f"Cyclone Track for {tc_id}")
    plt.savefig(f"/scratch/ll44/sc6160/out/plots/{tc_name}_track_plot.png", dpi=300, bbox_inches='tight')

def plot_tc_track_with_pred(tc_id, tc_name, tc_lats, tc_lons, tc_pred_lats, tc_pred_lons, aus_bounds=True):

    # create a new figure with a specific size and projection
    plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # add map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAKES, alpha=0.5)
    ax.add_feature(cfeature.RIVERS)

    # set the bounds to Australia
    if aus_bounds:
        ax.set_extent(utils.AUS_LON_BOUNDS + utils.AUS_LAT_BOUNDS, crs=ccrs.PlateCarree())

    ax.plot(tc_lons, tc_lats, marker='o', color='blue', markersize=5, linestyle='-', linewidth=2, transform=ccrs.PlateCarree())
    ax.plot(tc_pred_lons, tc_pred_lats, marker='o', color='red', markersize=5, linestyle='-', linewidth=2, transform=ccrs.PlateCarree())

    plt.title(f"Cyclone Track for {tc_id}")
    plt.savefig(f"/scratch/ll44/sc6160/out/plots/{tc_name}_track_plot.png", dpi=300, bbox_inches='tight')


def plot_pred_track_with_fields(data, tc_id, tc_lat, tc_lon, track_lat, track_lon, time):

    cntr2 = mp.ContourPlot()
    cntr2.data = data
    cntr2.field = 'msl'
    # cntr2.time = time
    cntr2.contours = np.arange(900, 1100, 2).tolist()
    cntr2.linecolor = 'black'
    cntr2.linestyle = 'solid'
    cntr2.label_fontsize = 15
    cntr2.clabels = True
    cntr2.plot_units = 'hPa' # Change from Pa to hPa
    cntr2.mpl_args = {'alpha': 0.7}

    panel = mp.MapPanel()
    panel.area = [105, 160, -50, 0] # Boundary of Australia
    panel.layers = ['states', 'coastline', 'borders']
    panel.title = f'UTC {time}'
    panel.title_fontsize = 30
    panel.plots = [cntr2]

    pc = mp.PanelContainer()
    pc.size = (15, 15)
    pc.panels = [panel]

    # panel.ax.plot(tc_lon, tc_lat, marker='X', color='red', markersize=15, transform=ccrs.PlateCarree())

    panel.ax.plot(track_lon, track_lat, marker='X', color='blue', markersize=15, transform=ccrs.PlateCarree())

    pc.save(f"/scratch/ll44/sc6160/out/plots/metpy_tracking/{tc_id}_{time.strftime('%Y%m%d%H')}.png")