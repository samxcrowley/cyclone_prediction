import os, math, datetime

import numpy as np
import xarray, matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.colors as mcolors
import metpy.plots as mp
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geopy.distance import geodesic

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
    plt.savefig(f"/scratch/ll44/sc6160/out/plots/tracks/{tc_name}_track_plot.png", dpi=300, bbox_inches='tight')

def plot_tc_track_with_pred(
        tc_id, tc_name,
        tc_lats, tc_lons, tc_start_time, tc_end_time,
        tc_pred_lats, tc_pred_lons, tc_pred_start_time, tc_pred_end_time,
        aus_bounds=True):

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

    # annotate the start and end timestamps for actual track
    tc_start_lon = tc_lons[0]
    tc_start_lat = tc_lats[0]
    tc_end_lon = tc_lons[-1]
    tc_end_lat = tc_lats[-1]

    if np.isfinite(tc_start_lon) and np.isfinite(tc_start_lat):
        ax.text(tc_start_lon, tc_start_lat, f'{tc_start_time}', color='blue', fontsize=10, transform=ccrs.PlateCarree())
    if np.isfinite(tc_end_lon) and np.isfinite(tc_end_lat):
        ax.text(tc_end_lon, tc_end_lat, f'{tc_end_time}', color='blue', fontsize=10, transform=ccrs.PlateCarree())

    offset = 2

    # annotate the start and end timestamps for predicted track
    ax.text(tc_pred_lons[0], tc_pred_lats[0] + offset, f'{tc_pred_start_time}', color='black', fontsize=10, transform=ccrs.PlateCarree())
    ax.text(tc_pred_lons[-1], tc_pred_lats[-1] + offset, f'{tc_pred_end_time}', color='black', fontsize=10, transform=ccrs.PlateCarree())

    plt.title(f"Cyclone Track for {tc_id}")
    plt.savefig(f"/scratch/ll44/sc6160/out/plots/{tc_name}_track_plot.png", dpi=300, bbox_inches='tight')

def plot_density_map(tracks, title, filename):

    grid_res = 0.1
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

                    mid_lat = grid_lat + (grid_res / 2)
                    mid_lon = grid_lon + (grid_res / 2)

                    r = geodesic((lat, lon), (mid_lat, mid_lon)).km
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
    plt.title(title)
    plt.savefig(f"/scratch/ll44/sc6160/out/plots/{filename}.png", dpi=300, bbox_inches='tight')


# plots the fields of MSLP of a dataset
# TC tracks must line up with the times of the dataset
def plot_mslp_field(obs_data, pred_data, obs_track_lats, obs_track_lons, pred_track_lats, pred_track_lons):

    obs_mslp = obs_data['mean_sea_level_pressure']
    gc_mslp = pred_data['mean_sea_level_pressure']
    projection = ccrs.PlateCarree()
    timesteps = len(obs_mslp['time'])
    
    # timesteps = 1

    mslp_min = 950
    mslp_max = 1050

    for t in range(timesteps):

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        
        ax_obs, ax_pred = axes

        # OBS plot
        ax_obs.add_feature(cfeature.COASTLINE)
        ax_obs.add_feature(cfeature.BORDERS)
        ax_obs.set_extent([utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1],
                       utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1]], crs=ccrs.PlateCarree())

        obs_mslp_timestep = obs_mslp.isel(time=t).drop_vars('time').squeeze('batch')

        cs_obs = ax_obs.contour(obs_mslp_timestep['lon'].values,
                        obs_mslp_timestep['lat'].values, 
                        obs_mslp_timestep.values / 100.0, # Pa -> hPa
                        levels=np.arange(mslp_min, mslp_max, 2),
                        colors='black',
                        linewidths=0.5,
                        transform=projection)
        ax_obs.clabel(cs_obs, inline=True, fontsize=10, fmt='%1.0f hPa')
        ax_obs.legend()
        ax_obs.set_title(f'Observed MSLP at Timestep {t}')

        # GC plot
        ax_pred.add_feature(cfeature.COASTLINE)
        ax_pred.add_feature(cfeature.BORDERS)
        ax_pred.set_extent([utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1],
                       utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1]], crs=ccrs.PlateCarree())

        gc_mslp_timestep = gc_mslp.isel(time=t).drop_vars('time').squeeze('batch')

        cs_pred = ax_pred.contour(gc_mslp_timestep['lon'].values,
                        gc_mslp_timestep['lat'].values, 
                        gc_mslp_timestep.values / 100.0, # Pa -> hPa
                        levels=np.arange(mslp_min, mslp_max, 2),
                        colors='black',
                        linewidths=0.5,
                        transform=projection)
        ax_pred.clabel(cs_pred, inline=True, fontsize=10, fmt='%1.0f hPa')
        ax_pred.legend()
        ax_pred.set_title(f'Predicted MSLP at Timestep {t}')

        # plot tracks
        ax_pred.plot(pred_track_lons, pred_track_lats, color='red', alpha=0.5, label='Full Predicted Track')
        ax_obs.plot(obs_track_lons, obs_track_lats, color='blue', alpha=0.5, label='Full Observed Track')

        # plot this timestep's track location
        if t < len(pred_track_lats) or t < len(obs_track_lats):
            ax_pred.plot(pred_track_lons[t], pred_track_lats[t], marker='o', color='red', markersize=8, label='Predicted Track')
            ax_obs.plot(obs_track_lons[t], obs_track_lats[t], marker='o', color='blue', markersize=8, label='Observed Track')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"/scratch/ll44/sc6160/out/plots/fields/olga/gc/{t}.png")

def impact_factor(r, ri):
    if r > ri:
        return 0
    return 1 - (r / ri)