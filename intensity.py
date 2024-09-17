import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import utils

tc_name, tc_id, start_time, end_time = utils.load_tc_data()

ibtracs = xr.open_dataset("/scratch/ll44/sc6160/data/IBTrACS/IBTrACS.ALL.v04r01.nc")
obs = xr.open_dataset(f"/scratch/ll44/sc6160/data/obs/{tc_name}_{tc_id}_obs_data.nc")
preds = xr.open_dataset(f"/scratch/ll44/sc6160/out/{tc_name}_{tc_id}_pred_data.nc")

tracks_ds = xr.open_dataset(f"/scratch/ll44/sc6160/out/tracks/{tc_name}_{tc_id}_tracks.nc")

ibt_lats = tracks_ds['ibtracs_lats']
ibt_lons = tracks_ds['ibtracs_lons']
obs_lats = tracks_ds['obs_lats']
obs_lons = tracks_ds['obs_lons']
pred_lats = tracks_ds['gc_lats']
pred_lons = tracks_ds['gc_lons']

def get_data_region(data, timedelta, lat, lon, radius):
    return data \
            .sel(time=timedelta, method='nearest') \
            .sel(lat=slice(lat - radius, lat + radius),
                    lon=slice(lon - radius, lon + radius)) \
            .squeeze('batch')

def ibtracs_max_wind():

    tc_data = ibtracs.where(ibtracs['sid'] == tc_id, drop=True)

    max_winds = tc_data['wmo_wind'][0].values
    max_winds = max_winds[~np.isnan(max_winds)]

    return max_winds / 2

def max_wind(data, track_lats, track_lons):

    times = data['time'].values

    max_winds = []

    idx = 0
    for time in times:

        if idx >= len(track_lats) or idx >= len(track_lons):
            continue

        lat = track_lats[idx]
        lon = track_lons[idx]

        radius = 0.5
        data_step = get_data_region(data, time, lat, lon, radius)
        
        u10 = data_step['10m_u_component_of_wind']
        v10 = data_step['10m_v_component_of_wind']

        wind_speed = np.sqrt(u10**2 + v10**2)
        max_wind_speed = wind_speed.max().values

        max_winds.append(float(max_wind_speed))

        idx += 1

    return max_winds

ibt_wind_v = ibtracs_max_wind()
ibt_wind = max_wind(obs, ibt_lats, ibt_lons)
obs_wind = max_wind(obs, obs_lats, obs_lons)
pred_wind = max_wind(preds, pred_lats, pred_lons)

max_len = max(len(ibt_wind), len(obs_wind), len(pred_wind))

ibt_wind_v_padded = np.pad(ibt_wind_v, (0, max_len - len(ibt_wind_v)), constant_values=np.nan)
ibt_wind_padded = np.pad(ibt_wind, (0, max_len - len(ibt_wind)), constant_values=np.nan)
obs_wind_padded = np.pad(obs_wind, (0, max_len - len(obs_wind)), constant_values=np.nan)
pred_wind_padded = np.pad(pred_wind, (0, max_len - len(pred_wind)), constant_values=np.nan)

x = range(max_len)

plt.plot(x, ibt_wind_v_padded, marker='o', label='IBTrACS values')
plt.plot(x, ibt_wind_padded, marker='o', label='IBTrACS')
plt.plot(x, obs_wind_padded, marker='s', label='OBS')
plt.plot(x, pred_wind_padded, marker='^', label='GC')
plt.xlabel("n timesteps")
plt.ylabel("Maximum wind speed (m/s)")
plt.legend()
plt.savefig(f"/scratch/ll44/sc6160/out/plots/intensity/{tc_name}_{tc_id}_intensity.png")