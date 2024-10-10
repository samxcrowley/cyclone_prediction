import xarray as xr
import sys
import os

names = ["Penny", "Gretel",
         "Damien", "Blake", "Marian", "Lucas", "Anika",
         "Paul", "Lincoln", "Kirrily"]

ibtracs = xr.open_dataset("/scratch/ll44/sc6160/data/IBTrACS/IBTrACS.ALL.v04r01.nc")

for name in names:

    if not os.path.exists(f"/scratch/ll44/sc6160/data/tc_data/{name}.json"):
        with open(f"/scratch/ll44/sc6160/data/tc_data/{name}.json", 'x'): pass

    matching_ids = []

    for i in range(ibtracs.sizes["storm"]):

        tc_name = ibtracs["name"].isel(storm=i).item().decode("utf-8")
        sid = ibtracs["sid"].isel(storm=i).item()

        if name.upper() == tc_name:
            matching_ids.append(sid)

    print(name, matching_ids)
    print()
    print()