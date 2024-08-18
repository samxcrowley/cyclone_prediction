from datetime import datetime, timedelta

import xarray as xr
import numpy as np
import pandas as pd

import plotting
import utils
import tracking

ibtracs = xr.open_dataset("/scratch/ll44/sc6160/data/IBTrACS/IBTrACS.last3years.v04r01.nc")

# pred_file_path = "/scratch/ll44/sc6160/data/2022-03/source-era5_data-2022-3_res-0.25_levels-37_ausonly_time-6h.nc"
pred_file_path = "/scratch/ll44/sc6160/data/ERA5/source-era5_date-2022-01-01_res-1.0_levels-37_steps-40.nc"

preds = xr.open_dataset(pred_file_path)

# tc_name = "OLGA"
# tc_id = "2024097S13119".encode('utf-8')
# tc_start_time = datetime(2024, 4, 4, 18, 0, 0)
# tc_end_time = datetime(2022, 4, 12, 0, 0, 0)

tcs = {'charlotte':
            {'id': '2022076S10126',
            'start': datetime(2022, 3, 19, 6),
            'end': datetime(2022, 3, 25, 18)},
        'tiffany':
            {'id': '2022008S13148',
             'start': datetime(2022, 1, 8, 0, 0, 0),
             'end': datetime(2022, 1, 17, 6, 0, 0)},
        'ilsa':
            {'id': '2023096S08133',
             'start': datetime(2023, 4, 6, 0, 0, 0),
             'end': datetime(2023, 4, 15, 12, 0, 0)},
        'olga':
            {'id': '2024097S13119',
             'start': datetime(2024, 4, 4, 18, 0, 0),
             'end': datetime(2022, 4, 12, 0, 0, 0)}
    }

for tc, data in tcs.items():

    tc_id = data['id']
    tc_start_time = data['start_time']
    tc_end_time = data['end_time']

    tc_data = ibtracs.where(ibtracs["sid"] == tc_id, drop=True)

    tc_lats = tc_data['lat']
    tc_lats_np = tc_lats.values[0]
    tc_lons = tc_data['lon']
    tc_lons_np = tc_lons.values[0]

    tc_start_lat = tc_lats_np[0]
    tc_start_lon = tc_lons_np[0]

    track_lats, track_lons = tracking.lazy_track(tc_start_lat, tc_start_lon, tc_start_time, tc_end_time)

    plotting.plot_tc_track_with_pred(tc_id, tc_name, tc_lats, tc_lons, track_lats, track_lons)