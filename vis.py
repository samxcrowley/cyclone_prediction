from datetime import datetime, timedelta
import sys

import xarray as xr
import numpy as np
import pandas as pd

import plotting

start_date = np.datetime64('2018-01-01')

ibtracs = xr.open_dataset('/scratch/ll44/sc6160/data/IBTrACS/IBTrACS.ALL.v04r01.nc')
ibtracs = ibtracs.where(ibtracs['time'] >= start_date, drop=True)

tracks = []

sid_array = ibtracs['sid'].values
sids = []
for s in sid_array:
    id = s[0]
    sids.append(id)

for tc_id in sids:
    
    storm_data = ibtracs.where(ibtracs['sid'] == tc_id, drop=True)
    
    if len(storm_data['lat']) == 0 or len(storm_data['lon']) == 0:
        continue

    lats = storm_data['lat'].values[0]
    lons = storm_data['lon'].values[0]
    
    track = {'lats': lats.tolist(), 'lons': lons.tolist()}
    
    tracks.append(track)

plotting.plot_density_map(tracks, 'Density map of TC tracks since 2018', '2018_onwards_density_map')