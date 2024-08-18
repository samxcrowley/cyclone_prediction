import xarray as xr
from glob import glob

import utils

year = 2022
month = 3

# surface level variables
sl_vars = {'msl': 'mean_sea_level_pressure', 'tp': 'total_precipitation_6hr'}
# pressure level variables
pl_vars = {'t': 'temperature', 'u': 'u_component_of_wind', 'v': 'v_component_of_wind', 'vo': 'relative_vorticity'}

paths = []

for var in sl_vars.keys():
    path = glob(f"/g/data/rt52/era5/single-levels/reanalysis/{var}/{year}/{var}_era5_oper_sfc_{year}{month:02d}01-*.nc")[0]
    paths.append(path)

for var in pl_vars.keys():
    path = glob(f"/g/data/rt52/era5/pressure-levels/reanalysis/{var}/{year}/{var}_era5_oper_pl_{year}{month:02d}01-*.nc")[0]
    paths.append(path)

# datasets = [xr.open_mfdataset(path, combine='by_coords') for path in paths]

datasets = [
    xr.open_mfdataset(
        path, 
        combine='by_coords',
        preprocess=lambda ds: ds.reindex(latitude=list(reversed(ds['latitude'])))\
                                .sel(latitude=slice(utils.AUS_LAT_BOUNDS[0], utils.AUS_LAT_BOUNDS[1]), \
                                     longitude=slice(utils.AUS_LON_BOUNDS[0], utils.AUS_LON_BOUNDS[1])) \
                                .resample(time='6h').nearest())
    for path in paths
]

# for i, ds in enumerate(datasets):
#     print(f"Dataset {i} time range: {ds.time.min().values} to {ds.time.max().values}")
#     print(f"Dataset {i} latitude range: {ds.latitude.min().values} to {ds.latitude.max().values}")
#     print(f"Dataset {i} longitude range: {ds.longitude.min().values} to {ds.longitude.max().values}")

combined_dataset = xr.merge(datasets)

# rename variables
combined_dataset = combined_dataset.rename({'latitude': 'lat'})
combined_dataset = combined_dataset.rename({'longitude': 'lon'})
for k, v in sl_vars.items():
    combined_dataset = combined_dataset.rename({k: v})
for k, v in pl_vars.items():
    combined_dataset = combined_dataset.rename({k: v})

print(combined_dataset)

combined_dataset.to_netcdf(f"/scratch/ll44/sc6160/data/{year}-{month:02d}/source-era5_data-{year}-{month}_res-0.25_levels-37_ausonly_time-6h.nc")