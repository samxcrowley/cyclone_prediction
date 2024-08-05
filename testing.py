import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import numpy as np

import plotting
import data_utils

# pred_file_path = "out/preds.nc"
# eval_file_path = "out/evals.nc"

# preds = xr.open_dataset(pred_file_path)
# evals = xr.open_dataset(eval_file_path)

ibtracs = xr.open_dataset("data/IBTrACS/IBTrACS.last3years.v04r01.nc")

target_name = "ANIKA"
target_season = 2022

# locate TC id
matching_ids = []

for i in range(ibtracs.sizes["storm"]):

    name = ibtracs["name"].isel(storm=i).item().decode("utf-8")
    season = ibtracs["season"].isel(storm=i).item()
    sid = ibtracs["sid"].isel(storm=i).item()

    if name == target_name and season == target_season:
        matching_ids.append(sid)

if len(matching_ids) == 0:
    raise ValueError(f"No cyclone found with name {target_name} and year {target_season}.")
elif len(matching_ids) > 1:
    raise ValueError(f"Multiple cyclones found with name {target_name} and year {target_season}: {matching_ids}")

tc_id = matching_ids[0]
tc_data = ibtracs.where(ibtracs["sid"] == tc_id, drop=True)
tc_lats = tc_data['lat']
tc_lats_np = tc_lats.values
tc_lons = tc_data['lon']
tc_lons_np = tc_lons.values

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

# set the extent to focus on Australia
# ax.set_extent([110, 160, -45, -10], crs=ccrs.PlateCarree())

# plot the cyclone track
ax.plot(tc_lons, tc_lats, marker='o', color='red', markersize=5, linestyle='-', linewidth=2, transform=ccrs.PlateCarree())

plt.title(f'Cyclone Track for {tc_id.decode("utf-8")}')
plt.savefig(f"out/plots/{target_name}_track_plot.png", dpi=300, bbox_inches='tight')