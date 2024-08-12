import xarray as xr

# TIFFANY: 2022008S13148
# VERNON: 2022054S13100
# ANIKA: 2022055S13129
# KARIM: 2022127S07088
# DOVI: 2022038S19164

ibtracs = xr.open_dataset("/scratch/ll44/sc6160/data/IBTrACS/IBTrACS.last3years.v04r01.nc")

tc_id = "2022008S13148".encode("utf-8")
tc_name = "DOVI"
tc_season = 2022

# locate TC id
matching_ids = []

for i in range(ibtracs.sizes["storm"]):

    name = ibtracs["name"].isel(storm=i).item().decode("utf-8")
    season = ibtracs["season"].isel(storm=i).item()
    sid = ibtracs["sid"].isel(storm=i).item()

    if name == tc_name and season == tc_season:
        matching_ids.append(sid)

print(matching_ids[0])