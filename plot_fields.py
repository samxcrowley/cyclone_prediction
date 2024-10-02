import os, sys
import numpy as np
import xarray as xr
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

tc_file = sys.argv[1]
tc_name, tc_id, start_time, end_time, tc_dir = utils.load_tc_data(tc_file)

tracks_ds = xr.open_dataset(f"/scratch/ll44/sc6160/out/tracks/{tc_name}_{tc_id}_tracks.nc")
obs = xr.open_dataset(f"/scratch/{tc_dir}/sc6160/data/obs/{tc_name}_{tc_id}_obs_data.nc")
preds = xr.open_dataset(f"/scratch/{tc_dir}/sc6160/out/pred/{tc_name}_{tc_id}_pred_data.nc")

obs_lats = tracks_ds['obs_lats']
obs_lons = tracks_ds['obs_lons']
pred_lats = tracks_ds['gc_lats']
pred_lons = tracks_ds['gc_lons']

def plot_fields(obs_data, pred_data, obs_track_lats, obs_track_lons, pred_track_lats, pred_track_lons, tc_name):

    obs_mslp = obs_data['mean_sea_level_pressure']
    gc_mslp = pred_data['mean_sea_level_pressure']
    projection = ccrs.PlateCarree()
    timesteps = len(obs_mslp['time'])

    obs_data['wind_speed'] = mpcalc.wind_speed(obs_data['10m_u_component_of_wind'],
                                               obs_data['10m_v_component_of_wind']).astype('float32')
    pred_data['wind_speed'] = mpcalc.wind_speed(pred_data['10m_u_component_of_wind'] * units('m/s'),
                                               pred_data['10m_v_component_of_wind'] * units('m/s')).astype('float32')
    
    timesteps = 5

    mslp_min = 900
    mslp_max = 1100

    for t in range(timesteps):

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(30, 12), subplot_kw={'projection': ccrs.PlateCarree()})
        
        ax_obs, ax_pred = axes

        # OBS plot
        ax_obs.add_feature(cfeature.COASTLINE)
        ax_obs.add_feature(cfeature.BORDERS)
        ax_obs.set_extent([utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1],
                       utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1]], crs=ccrs.PlateCarree())

        obs_mslp_timestep = obs_mslp.isel(time=t).drop_vars('time').squeeze('batch')
        obs_u10 = obs_data['10m_u_component_of_wind'].isel(time=t).drop_vars('time').squeeze('batch') \
            .isel(lon=slice(None, None, 10), lat=slice(None, None, 10))
        obs_v10 = obs_data['10m_v_component_of_wind'].isel(time=t).drop_vars('time').squeeze('batch') \
            .isel(lon=slice(None, None, 10), lat=slice(None, None, 10))
        obs_wind_speed = obs_data['wind_speed'].isel(time=t).drop_vars('time').squeeze('batch')

        pred_windspeed_contour = ax_obs.contourf(obs_wind_speed['lon'].values,
                                            obs_wind_speed['lat'].values,
                                            obs_wind_speed.values,
                                            levels=np.arange(0, 50, 2),
                                            cmap='BuPu',
                                            transform=projection)
        plt.colorbar(pred_windspeed_contour, ax=ax_obs, orientation='horizontal', pad=0.05, label='Wind Speed (knots)')

        cs_obs = ax_obs.contour(obs_mslp_timestep['lon'].values,
                        obs_mslp_timestep['lat'].values, 
                        obs_mslp_timestep.values / 100.0, # Pa -> hPa
                        levels=np.arange(mslp_min, mslp_max, 1),
                        colors='black',
                        linewidths=1,
                        transform=projection)
        ax_obs.clabel(cs_obs, inline=True, fontsize=10, fmt='%1.0f hPa')

        # Plot wind barbs for observed data
        ax_obs.barbs(obs_u10['lon'].values, obs_u10['lat'].values, 
                obs_u10.values, obs_v10.values, 
                length=5, color='black', transform=projection)

        ax_obs.legend()
        ax_obs.set_title(f'Observed Fields at Timestep {t}')

        # GC plot
        ax_pred.add_feature(cfeature.COASTLINE)
        ax_pred.add_feature(cfeature.BORDERS)
        ax_pred.set_extent([utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1],
                       utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1]], crs=ccrs.PlateCarree())

        gc_mslp_timestep = gc_mslp.isel(time=t).drop_vars('time').squeeze('batch')
        gc_u10 = pred_data['10m_u_component_of_wind'].isel(time=t).drop_vars('time').squeeze('batch') \
                    .isel(lon=slice(None, None, 10), lat=slice(None, None, 10))
        gc_v10 = pred_data['10m_v_component_of_wind'].isel(time=t).drop_vars('time').squeeze('batch') \
                    .isel(lon=slice(None, None, 10), lat=slice(None, None, 10))
        gc_wind_speed = pred_data['wind_speed'].isel(time=t).drop_vars('time').squeeze('batch')

        pred_windspeed_contour = ax_pred.contourf(gc_wind_speed['lon'].values,
                                            gc_wind_speed['lat'].values,
                                            gc_wind_speed.values,
                                            levels=np.arange(0, 50, 2),
                                            cmap='BuPu',
                                            transform=projection)
        plt.colorbar(pred_windspeed_contour, ax=ax_pred, orientation='horizontal', pad=0.05, label='Wind Speed (knots)')

        cs_pred = ax_pred.contour(gc_mslp_timestep['lon'].values,
                        gc_mslp_timestep['lat'].values, 
                        gc_mslp_timestep.values / 100.0, # Pa -> hPa
                        levels=np.arange(mslp_min, mslp_max, 1),
                        colors='black',
                        linewidths=1,
                        transform=projection)
        ax_pred.clabel(cs_pred, inline=True, fontsize=10, fmt='%1.0f hPa')

        ax_pred.barbs(gc_u10['lon'].values, gc_u10['lat'].values, 
                  gc_u10.values, gc_v10.values, 
                  length=5, color='black', transform=projection)

        ax_pred.legend()
        ax_pred.set_title(f'Predicted Fields at Timestep {t}')

        # plot tracks
        ax_pred.plot(pred_track_lons, pred_track_lats, color='red', alpha=0.5, label='Full Predicted Track')
        ax_obs.plot(obs_track_lons, obs_track_lats, color='blue', alpha=0.5, label='Full Observed Track')

        # plot this timestep's track location
        if t < len(pred_track_lats) or t < len(obs_track_lats):
            ax_pred.plot(pred_track_lons[t], pred_track_lats[t], marker='o', color='red', markersize=8, label='Predicted Track')
            ax_obs.plot(obs_track_lons[t], obs_track_lats[t], marker='o', color='blue', markersize=8, label='Observed Track')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"/scratch/ll44/sc6160/out/plots/fields/{tc_name}/{t}.png")

plot_fields(obs, preds, obs_lats, obs_lons, pred_lats, pred_lons, tc_name)