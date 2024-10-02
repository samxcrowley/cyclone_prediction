import os, sys
from datetime import datetime

import xarray as xr
from glob import glob

import utils

tc_file = sys.argv[1]
tc_name, tc_id, start_time, end_time, tc_dir = utils.load_tc_data(tc_file)
year = start_time.year
month = start_time.month

folder_path = f"/scratch/{tc_dir}/sc6160/data/obs/"
file_name = f"{tc_name}_{tc_id}_obs_data.nc"
os.makedirs(folder_path, exist_ok=True)

# surface level variables
sl_vars = {
    '2t': '2m_temperature',
    '10u': '10m_u_component_of_wind',
    '10v': '10m_v_component_of_wind',
    'lsm': 'land_sea_mask',
    'tisr': 'toa_incident_solar_radiation',
    'msl': 'mean_sea_level_pressure',
    'tp': 'total_precipitation_6hr'
}

# pressure level variables
pl_vars = {
    't': 'temperature',
    'u': 'u_component_of_wind',
    'v': 'v_component_of_wind',
    'vo': 'relative_vorticity',
    'q': 'specific_humidity',
    'w': 'vertical_velocity',
    'z': 'geopotential'
}

paths = []

# get surface level filepaths
for var in sl_vars.keys():
    path = glob(f"/g/data/rt52/era5/single-levels/reanalysis/{var}/{year}/{var}_era5_oper_sfc_{year}{month:02d}01-*.nc")[0]
    paths.append(path)

# get pressure level filepaths
for var in pl_vars.keys():
    path = glob(f"/g/data/rt52/era5/pressure-levels/reanalysis/{var}/{year}/{var}_era5_oper_pl_{year}{month:02d}01-*.nc")[0]
    paths.append(path)

paths.append(glob(f"/g/data/rt52/era5/single-levels/reanalysis/z/{year}/z_era5_oper_sfc_{year}{month:02d}01-*.nc")[0])

datasets = [
    xr.open_mfdataset(
        path, 
        combine='by_coords',
        # chunks={'time': 1, 'lat': 90, 'lon': 180},
        preprocess=lambda ds: ds.reindex(latitude=list(reversed(ds['latitude']))) \
                                .sel(time=slice(start_time, end_time)) \
                                .resample(time='6h').nearest())
    for path in paths
]

z_surface = xr.open_dataset(glob(f"/g/data/rt52/era5/single-levels/reanalysis/z/{year}/z_era5_oper_sfc_{year}{month:02d}01-*.nc")[0])
z_surface = z_surface.reindex(latitude=list(reversed(z_surface['latitude'])))
z_surface = z_surface \
                    .sel(time=slice(start_time, end_time)) \
                    .resample(time='6h').nearest() \
                    .rename({'z': 'geopotential_at_surface'}) \

datasets.append(z_surface)

combined_dataset = xr.merge(datasets, compat='override')

# rename variables
combined_dataset = combined_dataset.rename({'latitude': 'lat'})
combined_dataset = combined_dataset.rename({'longitude': 'lon'})

for k, v in sl_vars.items():
    if k == '2t':
        combined_dataset = combined_dataset.rename({'t2m': '2m_temperature'})
    elif k == '10u':
        combined_dataset = combined_dataset.rename({'u10': '10m_u_component_of_wind'})
    elif k == '10v':
        combined_dataset = combined_dataset.rename({'v10': '10m_v_component_of_wind'})
    else:
        combined_dataset = combined_dataset.rename({k: v})
        
for k, v in pl_vars.items():
    combined_dataset = combined_dataset.rename({k: v})

# gpu flags

# add new datetime coord
combined_dataset['datetime'] = combined_dataset['time']
combined_dataset = combined_dataset.set_coords('datetime')

# change time coord to timedelta format
combined_dataset['time'] = combined_dataset['datetime'][:] - combined_dataset['datetime'][0]

combined_dataset['geopotential_at_surface'] = combined_dataset['geopotential_at_surface'].isel(time=0).drop_vars('time')
combined_dataset['land_sea_mask'] = combined_dataset['land_sea_mask'].isel(time=0).drop_vars('time')

# add batch dimension to those that need it
batch_ls = ['datetime',
            '2m_temperature',
            'mean_sea_level_pressure',
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'total_precipitation_6hr',
            'toa_incident_solar_radiation',
            'temperature',
            'geopotential',
            'u_component_of_wind',
            'v_component_of_wind',
            'relative_vorticity',
            'vertical_velocity',
            'specific_humidity'
]
for v in batch_ls:
    combined_dataset[v] = combined_dataset[v].expand_dims('batch', axis=0)

# convert variables to float32 type
for v in sl_vars.values():
    combined_dataset[v] = combined_dataset[v].astype('float32')
for v in pl_vars.values():
    combined_dataset[v] = combined_dataset[v].astype('float32')
combined_dataset['geopotential_at_surface'] = \
    combined_dataset['geopotential_at_surface'].astype('float32')

# reorder longitude from [-180, 180] to [0, 360] (from Jonas' code)
combined_dataset['_longitude_adjusted'] = xr.where(combined_dataset['lon'] < 0,
                combined_dataset['lon'] + 360, combined_dataset['lon'])
combined_dataset = combined_dataset \
                .swap_dims({'lon': '_longitude_adjusted'}) \
                .sel(**{'_longitude_adjusted': sorted(combined_dataset._longitude_adjusted)}) \
                .drop_vars('lon')
combined_dataset = combined_dataset.rename({'_longitude_adjusted': 'lon'})

# reorder coords
coords_order = ('lon', 'lat', 'level', 'time', 'datetime')

# remove attribute data
combined_dataset.attrs = {}

# compression
compression_settings = {
    'zlib': True,
    'complevel': 5 # recommended comp. level from http://climate-cms.wikis.unsw.edu.au/NetCDF_Compression_Tools
}
encoding = {
    var: compression_settings for var in combined_dataset.data_vars
}

# save dataset
combined_dataset.to_netcdf(os.path.join(folder_path, file_name))
# combined_dataset.to_netcdf(os.path.join(folder_path, f"comp_{file_name}"), encoding=encoding)

print(f"Succesfully saved dataset {os.path.join(folder_path, file_name)}")