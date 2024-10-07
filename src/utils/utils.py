import os
from typing import Optional
import json
import dataclasses
from datetime import datetime, timedelta
import xarray
import numpy as np
from glob import glob

# IMERG bounds: 100,-50,175,0

AUS_LON_BOUNDS = [100, 175]
AUS_LAT_BOUNDS = [-50, 0]
EARTH_RADIUS_KM = 6371.0
EARTH_RADIUS_M = 6371000
TIME_STEP = timedelta(hours=6)

IBT_V_COLOUR = 'purple'
IBT_COLOUR = 'green'
OBS_COLOUR = 'blue'
PRED_COLOUR = 'red'

def ms_to_knots(ms):
    return ms * 1.94384
def knots_to_ms(knots):
    return knots / 1.94384

def impact_factor(r, ri):
    if r > ri:
        return 0
    return 1 - (r / ri)

def parse_file_parts(file_name):
    parts = {}
    for part in file_name.split("/")[-1].split("_"):
        print(part)
        part_split = part.split("-", 1)
        print(part_split)
        parts[part_split[0]] = part_split[1]
    return parts

def get_data_region(data, timedelta, lat, lon, radius):
    return data \
            .sel(time=timedelta, method='nearest') \
            .sel(lat=slice(lat - radius, lat + radius),
                    lon=slice(lon - radius, lon + radius - 0.25)) \
            .squeeze('batch')

def datetime_to_timedelta(datetime, ref_time):

    datetime = np.datetime64(datetime, 'ns')
    ref_time = np.datetime64(ref_time, 'ns')
    
    timedelta_value = datetime - ref_time
    
    return timedelta_value

def timedelta_to_datetime(timedelta_obj, ref_time):

    if isinstance(timedelta_obj, np.timedelta64):
        timedelta_obj = timedelta(seconds=timedelta_obj / np.timedelta64(1, 's'))

    result_datetime = ref_time + timedelta_obj
    
    return result_datetime

def get_month_range(start, end):

    months = []
    current = start

    while current <= end:
        months.append((current.year, current.month))
        if current.month == 12:
            current = datetime(current.year + 1, 1, 1)
        else:
            current = datetime(current.year, current.month + 1, 1)

    return months

def is_on_land(mask_data, lat, lon):
    return mask_data.sel(lat=lat, lon=lon, method='nearest')['land_sea_mask'] == 1

def load_tc_data(tc_file):

    tc_data = {}

    try:
        with open(tc_file, 'r') as f:
            tc_data = json.load(f)
    except:
        print(f"No file {tc_file}")
        return None

    tc_data['start_time'] = datetime.fromisoformat(tc_data['start_time'])
    tc_data['end_time'] = datetime.fromisoformat(tc_data['end_time'])

    return tc_data['tc_name'], tc_data['tc_id'], tc_data['start_time'], tc_data['end_time'], tc_data['dir']

def get_all_tc_names():

    all_files = os.listdir("/scratch/ll44/sc6160/data/tc_data")
    tc_names = [f[:-5] for f in all_files if f.endswith('.json')]
    return tc_names