from datetime import datetime, timedelta
import sys

import xarray as xr
import numpy as np
import pandas as pd

import plotting
import utils
import tracking, old_tracking

ibtracs = xr.open_dataset("/scratch/ll44/sc6160/data/IBTrACS/IBTrACS.ALL.v04r01.nc")

obs = xr.open_dataset("/scratch/ll44/sc6160/data/2023-04/source-era5_data-2023-4_res-0.25_levels-37_tc-ilsa.nc") \
        # .isel(time=slice(1, -1))
preds = xr.open_dataset("/scratch/ll44/sc6160/out/preds_ilsa.nc")

tcs = {
        # 'charlotte':
        #     {'id': '2022076S10126',
        #     'start': datetime(2022, 3, 19, 6),
        #     'end': datetime(2022, 3, 25, 18)},
        # 'tiffany':
        #     {'id': '2022008S13148',
        #      'start': datetime(2022, 1, 8, 0, 0, 0),
        #      'end': datetime(2022, 1, 17, 6, 0, 0)},
        'ilsa':
            {'id': '2023096S08133',
             'start': datetime(2023, 4, 15, 12, 0, 0),
             'end': datetime(2023, 4, 17, 0, 0, 0)},
        # 'olga':
        #     {'id': '2024096S11120',
        #      'start': datetime(2024, 4, 4, 0, 0, 0),
        #      'end': datetime(2024, 4, 12, 0, 0, 0)},
        # 'imogen':
        #     {'id': '2021001S14136',
        #     'start': datetime(2021, 1, 1),
        #     'end':datetime(2021, 1, 6)}
}

for tc_name, data in tcs.items():

    tc_id = data['id'].encode('utf-8')
    tc_start_time = data['start']
    tc_end_time = data['end']

    tc_data = ibtracs.where(ibtracs['sid'] == tc_id, drop=True)

    tc_lats = tc_data['lat'].values[0]
    tc_lats = tc_lats[~np.isnan(tc_lats)]
    tc_start_lat = tc_lats[0]

    tc_lons = tc_data['lon'].values[0]
    tc_lons = tc_lons[~np.isnan(tc_lons)]
    tc_start_lon = tc_lons[0]

    track_lats, track_lons, tc_pred_end_time = tracking.gc_track(obs, tc_start_lat, tc_start_lon, tc_start_time)
    plotting.plot_tc_track_with_pred(tc_id, f"{tc_name}_gc",
                                     tc_lats, tc_lons, tc_start_time, tc_end_time,
                                     track_lats, track_lons, tc_start_time, tc_pred_end_time)
    # plotting.plot_mslp_field(obs, preds, tc_lats, tc_lons, track_lats, track_lons)