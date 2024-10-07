import xarray as xr
import sys

# TIFFANY: 2022008S13148
# VERNON: 2022054S13100
# ANIKA: 2022055S13129
# KARIM: 2022127S07088
# DOVI: 2022038S19164
# KIRRILY: 2024017S15151
# CHARLOTTE: 2022076S10126
# SETH: 2021358S09130
# IMOGEN: 2021001S14136

ibtracs = xr.open_dataset("/scratch/ll44/sc6160/data/IBTrACS/IBTrACS.ALL.v04r01.nc")

tc_name = sys.argv[1]

matching_ids = []

for i in range(ibtracs.sizes["storm"]):

    name = ibtracs["name"].isel(storm=i).item().decode("utf-8")
    # season = ibtracs["season"].isel(storm=i).item()
    sid = ibtracs["sid"].isel(storm=i).item()

    if name == tc_name:
        matching_ids.append(sid)

print(matching_ids)