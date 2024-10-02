import sys
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import utils

tc_file = sys.argv[1]
tc_name, tc_id, start_time, end_time, tc_dir = utils.load_tc_data(tc_file)

ibtracs = xr.open_dataset("/scratch/ll44/sc6160/data/IBTrACS/IBTrACS.ALL.v04r01.nc")
obs = None
try:
    obs = xr.open_dataset(f"/scratch/ll44/sc6160/data/obs/{tc_name}_{tc_id}_obs_data.nc")
except:
    print(f"No data for {tc_name} yet.")
    sys.exit(0)
preds = xr.open_dataset(f"/scratch/ll44/sc6160/out/pred/{tc_name}_{tc_id}_pred_data.nc")

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

    tc_data = ibtracs.where(ibtracs['sid'] == tc_id.encode('utf-8'), drop=True)

    max_winds = tc_data['bom_wind'][0].values
    max_winds = max_winds[::2]
    max_winds = max_winds[~np.isnan(max_winds)]

    return max_winds / 1.944 # knots -> m/s

def max_wind(data, track_lats, track_lons):

    times = data['time'].values

    max_winds = []

    idx = 0
    for time in times:

        if idx >= len(track_lats) or idx >= len(track_lons):
            continue

        lat = track_lats[idx]
        lon = track_lons[idx]

        radius = 2
        data_step = get_data_region(data, time, lat, lon, radius)
        
        u10 = data_step['10m_u_component_of_wind']
        v10 = data_step['10m_v_component_of_wind']

        wind_speed = np.sqrt(u10**2 + v10**2)
        max_wind_speed = wind_speed.max().values

        max_winds.append(float(max_wind_speed))

        idx += 1

    return max_winds

def mslp(data, track_lats, track_lons):

    times = data['time'].values

    mslps = []

    idx = 0
    for time in times:

        if idx >= len(track_lats) or idx >= len(track_lons):
            continue

        lat = track_lats[idx]
        lon = track_lons[idx]

        radius = 2
        data_step = get_data_region(data, time, lat, lon, radius)
        
        mslp = data_step['mean_sea_level_pressure']
        min_mslp = mslp.min().values / 100.0 # -> Pa to hPa

        mslps.append(float(min_mslp))

        idx += 1

    return mslps

ibt_wind_v = ibtracs_max_wind()
ibt_wind = max_wind(obs, ibt_lats, ibt_lons)
obs_wind = max_wind(obs, obs_lats, obs_lons)
pred_wind = max_wind(preds, pred_lats, pred_lons)

max_len = max(len(ibt_wind_v), len(ibt_wind), len(obs_wind), len(pred_wind))

ibt_wind_v_padded = np.pad(ibt_wind_v, (0, max_len - len(ibt_wind_v)), constant_values=np.nan)
ibt_wind_padded = np.pad(ibt_wind, (0, max_len - len(ibt_wind)), constant_values=np.nan)
obs_wind_padded = np.pad(obs_wind, (0, max_len - len(obs_wind)), constant_values=np.nan)
pred_wind_padded = np.pad(pred_wind, (0, max_len - len(pred_wind)), constant_values=np.nan)

x = range(max_len)

plt.plot(x, ibt_wind_v_padded, marker='o', label='IBTrACS BOM winds', color=utils.IBT_V_COLOUR)
plt.plot(x, ibt_wind_padded, marker='o', label='ERA5 winds (IBTrACS track)', color=utils.IBT_COLOUR)
plt.plot(x, obs_wind_padded, marker='o', label='ERA5 winds (STracker)', color=utils.OBS_COLOUR)
plt.plot(x, pred_wind_padded, marker='o', label='GC winds (STracker)', color=utils.PRED_COLOUR)
plt.xlabel("n timesteps")
plt.ylabel("Maximum wind speed (m/s)")
plt.legend()
plt.savefig(f"/scratch/ll44/sc6160/out/plots/intensity/{tc_name}_{tc_id}_intensity_wind.png")

wind_ds = xr.Dataset(
    data_vars={
        'ibt_v': (['index1'], ibt_wind_v),
        'ibt': (['index2'], ibt_wind),
        'obs': (['index3'], obs_wind),
        'pred': (['index4'], pred_wind)
    }
)
wind_ds.to_netcdf(f"/scratch/ll44/sc6160/out/intensity/{tc_name}_{tc_id}_intensity_wind.nc")

ibt_mslp = mslp(obs, ibt_lats, ibt_lons)
obs_mslp = mslp(obs, obs_lats, obs_lons)
pred_mslp = mslp(preds, pred_lats, pred_lons)

max_len = max(len(ibt_mslp), len(obs_mslp), len(pred_mslp))

ibt_mslp_padded = np.pad(ibt_mslp, (0, max_len - len(ibt_mslp)), constant_values=np.nan)
obs_mslp_padded = np.pad(obs_mslp, (0, max_len - len(obs_mslp)), constant_values=np.nan)
pred_mslp_padded = np.pad(pred_mslp, (0, max_len - len(pred_mslp)), constant_values=np.nan)

x = range(max_len)

plt.clf()
plt.plot(x, ibt_mslp_padded, marker='o', label='IBTrACS track on OBS', color=utils.IBT_COLOUR)
plt.plot(x, obs_mslp_padded, marker='o', label='Algo. track on OBS', color=utils.OBS_COLOUR)
plt.plot(x, pred_mslp_padded, marker='o', label='Algo. track on GC', color=utils.PRED_COLOUR)
plt.xlabel("n timesteps")
plt.ylabel("Minimum mean sea level pressure (hPa)")
plt.legend()
plt.savefig(f"/scratch/ll44/sc6160/out/plots/intensity/{tc_name}_{tc_id}_intensity_mslp.png")

mslp_ds = xr.Dataset(
    data_vars={
        'ibt': (['index2'], ibt_mslp),
        'obs': (['index3'], obs_mslp),
        'pred': (['index4'], pred_mslp)
    }
)
mslp_ds.to_netcdf(f"/scratch/ll44/sc6160/out/intensity/{tc_name}_{tc_id}_intensity_mslp.nc")