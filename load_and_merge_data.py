# code taken from Jonas' final report
# remember to cite !

import os

import xarray as xr
from glob import glob

year = 2022
month = 1

# surface-level data
sfc_2t = xr.open_dataset((glob(f"/g/data/rt52/era5/single-levels/reanalysis/2t/{year}/2t_era5_oper_sfc_{year}{month:02d}01-*.nc"))[0])
sfc_10u = xr.open_dataset((glob(f"/g/data/rt52/era5/single-levels/reanalysis/10u/{year}/10u_era5_oper_sfc_{year}{month:02d}01-*.nc"))[0])
sfc_10v = xr.open_dataset((glob(f"/g/data/rt52/era5/single-levels/reanalysis/10v/{year}/10v_era5_oper_sfc_{year}{month:02d}01-*.nc"))[0])
# sfc_lsm = xr.open_dataset((glob(f"/g/data/rt52/era5/single-levels/reanalysis/lsm/{year}/lsm_era5_oper_sfc_{year}{month:02d}01-*.nc"))[0])
sfc_msl = xr.open_dataset((glob(f"/g/data/rt52/era5/single-levels/reanalysis/msl/{year}/msl_era5_oper_sfc_{year}{month:02d}01-*.nc"))[0])
# sfc_tisr = xr.open_dataset((glob(f"/g/data/rt52/era5/single-levels/reanalysis/tisr/{year}/tisr_era5_oper_sfc_{year}{month:02d}01-*.nc"))[0])
sfc_tp = xr.open_dataset((glob(f"/g/data/rt52/era5/single-levels/reanalysis/tp/{year}/tp_era5_oper_sfc_{year}{month:02d}01-*.nc"))[0])
sfc_z = xr.open_dataset((glob(f"/g/data/rt52/era5/single-levels/reanalysis/z/{year}/z_era5_oper_sfc_{year}{month:02d}01-*.nc"))[0])

# pressure-level data
# pl_q = xr.open_dataset((glob(f"/g/data/rt52/era5/pressure-levels/reanalysis/q/{year}/q_era5_oper_pl_{year}{month:02d}01-*.nc"))[0])
pl_t = xr.open_dataset((glob(f"/g/data/rt52/era5/pressure-levels/reanalysis/t/{year}/t_era5_oper_pl_{year}{month:02d}01-*.nc"))[0])
pl_u = xr.open_dataset((glob(f"/g/data/rt52/era5/pressure-levels/reanalysis/u/{year}/u_era5_oper_pl_{year}{month:02d}01-*.nc"))[0])
pl_v = xr.open_dataset((glob(f"/g/data/rt52/era5/pressure-levels/reanalysis/v/{year}/v_era5_oper_pl_{year}{month:02d}01-*.nc"))[0])
# pl_w = xr.open_dataset((glob(f"/g/data/rt52/era5/pressure-levels/reanalysis/w/{year}/w_era5_oper_pl_{year}{month:02d}01-*.nc"))[0])
pl_z = xr.open_dataset((glob(f"/g/data/rt52/era5/pressure-levels/reanalysis/z/{year}/z_era5_oper_pl_{year}{month:02d}01-*.nc"))[0])

sfc_2t = sfc_2t.rename({'t2m':'2m_temperature'})
sfc_10u = sfc_10u.rename({'u10':'10m_u_component_of_wind'})
sfc_10v = sfc_10v.rename({'v10':'10m_v_component_of_wind'})
# sfc_lsm = sfc_lsm.rename({'lsm':'land_sea_mask'})
sfc_msl = sfc_msl.rename({'msl':'mean_sea_level_pressure'})
# sfc_tisr = sfc_tisr.rename({'tisr':'toa_incident_solar_radiation'})
sfc_tp = sfc_tp.rename({'tp':'total_precipitation_6hr'})
sfc_z = sfc_z.rename({'z':'geopotential_at_surface'})
# pl_q = pl_q.rename({'q':'specific_humidity'})
pl_t = pl_t.rename({'t':'temperature'})
pl_u = pl_u.rename({'u':'u_component_of_wind'})
pl_v = pl_v.rename({'v':'v_component_of_wind'})
# pl_w = pl_w.rename({'w':'vertical_velocity'})
pl_z = pl_z.rename({'z':'geopotential'})

# Remove "time" from "sfc_lsm" and "sfc_z" to achieve 2D-data with (lat, lon) dimension only
sfc_z = sfc_z.isel(time=0).drop_vars("time")
# sfc_lsm = sfc_lsm.isel(time=0).drop_vars("time")

# Unsqueeze existing variables to add a new 'batch' dimension
sfc_2t['2m_temperature'] = sfc_2t['2m_temperature'].expand_dims('batch', axis=0)
sfc_10u['10m_u_component_of_wind'] = sfc_10u['10m_u_component_of_wind'].expand_dims('batch', axis=0)
sfc_10v['10m_v_component_of_wind'] = sfc_10v['10m_v_component_of_wind'].expand_dims('batch', axis=0)
sfc_msl['mean_sea_level_pressure'] = sfc_msl['mean_sea_level_pressure'].expand_dims('batch', axis=0)
# sfc_tisr['toa_incident_solar_radiation'] = sfc_tisr['toa_incident_solar_radiation'].expand_dims('batch', axis=0)
sfc_tp['total_precipitation_6hr'] = sfc_tp['total_precipitation_6hr'].expand_dims('batch', axis=0)
# pl_q['specific_humidity'] = pl_q['specific_humidity'].expand_dims('batch', axis=0)
pl_t['temperature'] = pl_t['temperature'].expand_dims('batch', axis=0)
pl_u['u_component_of_wind'] = pl_u['u_component_of_wind'].expand_dims('batch', axis=0)
pl_v['v_component_of_wind'] = pl_v['v_component_of_wind'].expand_dims('batch', axis=0)
# pl_w['vertical_velocity'] = pl_w['vertical_velocity'].expand_dims('batch', axis=0)
pl_z['geopotential'] = pl_z['geopotential'].expand_dims('batch', axis=0)

sfc_2t = sfc_2t['2m_temperature'].astype('float32')
sfc_10u = sfc_10u['10m_u_component_of_wind'].astype('float32')
sfc_10v = sfc_10v['10m_v_component_of_wind'].astype('float32')
# sfc_lsm = sfc_lsm['land_sea_mask'].astype('float32')
sfc_msl = sfc_msl['mean_sea_level_pressure'].astype('float32')
# sfc_tisr = sfc_tisr['toa_incident_solar_radiation'].astype('float32')
sfc_tp = sfc_tp['total_precipitation_6hr'].astype('float32')
sfc_z = sfc_z['geopotential_at_surface'].astype('float32')
# pl_q = pl_q['specific_humidity'].astype('float32')
pl_t = pl_t['temperature'].astype('float32')
pl_u = pl_u['u_component_of_wind'].astype('float32')
pl_v = pl_v['v_component_of_wind'].astype('float32')
# pl_w = pl_w['vertical_velocity'].astype('float32')
pl_z = pl_z['geopotential'].astype('float32')

merged = xr.merge([sfc_2t, sfc_10u, sfc_10v, sfc_msl,
sfc_tp, sfc_z, pl_t, pl_u, pl_v, pl_z])

# resample to every 6 hours and select the nearest data point
merged = merged.resample(time='6h').nearest()

merged = merged.rename({'longitude': 'lon'})
merged = merged.rename({'latitude': 'lat'})

# reverse the order of latitude from [90.0,-90.0] to [-90.0,90.0]
merged = merged.reindex(lat=list(reversed(merged.lat)))

# Adjust the longitude from [-180,179.75] to [0,359.75]
# If original 'lon' is negative, add 360 to its value, else remains the same
merged['_longitude_adjusted'] = xr.where(merged['lon'] < 0, merged['lon'] + 360, merged['lon'])

merged = (merged.swap_dims({'lon': '_longitude_adjusted'}).sel(**{'_longitude_adjusted': sorted(merged._longitude_adjusted)}).drop_vars('lon'))
merged = merged.rename({'_longitude_adjusted': 'lon'})

merged["datetime"] = merged["time"]

merged["time"] = merged["datetime"][:] - merged["datetime"][0]

merged["datetime"] = merged["datetime"].expand_dims('batch', axis=0)

merged = merged.set_coords("datetime")

merged.attrs = {}

merged.to_netcdf('/scratch/ll44/sc6160/data/2022-01/merged_resampled_6h_less.nc')