import dataclasses
from datetime import datetime, timedelta
import xarray
import numpy as np

from graphcast import data_utils as g_data_utils

from typing import Optional

AUS_LON_BOUNDS = [110, 160]
AUS_LAT_BOUNDS = [-45, -10]
EARTH_RADIUS = 6371.0
TIME_STEP = timedelta(hours=6)

def parse_file_parts(file_name):
    parts = {}
    for part in file_name.split("/")[-1].split("_"):
        print(part)
        part_split = part.split("-", 1)
        print(part_split)
        parts[part_split[0]] = part_split[1]
    return parts

# selects data matching variable, level, max_steps
def select(data: xarray.Dataset, variable: str, level: Optional[int] = None, max_steps: Optional[int] = None) -> xarray.Dataset:
  
    data = data[variable]
    
    if "batch" in data.dims:
        data = data.isel(batch=0)

    if max_steps is not None and "time" in data.sizes and max_steps < data.sizes["time"]:
        data = data.isel(time=range(0, max_steps))

    if level is not None and "level" in data.coords:
        data = data.sel(level=level)

    return data

# subsets data to bounds of specified latitudes and longitudes
def subset_to_region(data, lat_bounds, lon_bounds):
    return data.sel(lat=slice(lat_bounds[0], lat_bounds[1]), lon=slice(lon_bounds[0], lon_bounds[1]))

def prepare_data_dict(predictions, eval_targets, variable, lat_bounds, lon_bounds):
    
    pred_subset = subset_to_region(predictions, lat_bounds, lon_bounds)
    eval_subset = subset_to_region(eval_targets, lat_bounds, lon_bounds)
    diff_subset = eval_subset - pred_subset

    # set colormap based on variable being plotted
    if variable == 'mean_sea_level_pressure':
        cmap = 'coolwarm'
    elif variable == '2m_temperature':
        cmap = 'inferno' # or 'inferno'
    else:
        cmap = 'viridis' # default colormap

    data_dict = {
        "Prediction": (pred_subset[variable].squeeze(), None, cmap),
        "Target": (eval_subset[variable].squeeze(), None, cmap),
        "Difference": (diff_subset[variable].squeeze(), None, cmap)
    }

    return data_dict

def extract_inputs_targets_forcings(example_batch, task_config):

    train_steps = example_batch.sizes["time"] - 2
    eval_steps = example_batch.sizes["time"] - 2

    train_inputs, train_targets, train_forcings = g_data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("6h", f"{train_steps*6}h"), **dataclasses.asdict(task_config))
    eval_inputs, eval_targets, eval_forcings = g_data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("6h", f"{eval_steps*6}h"), **dataclasses.asdict(task_config))
    
    return train_inputs, train_targets, train_forcings, eval_inputs, eval_targets, eval_forcings

def datetime_to_timedelta(datetime, ref_time):

    datetime = np.datetime64(datetime, 'ns')
    ref_time = np.datetime64(ref_time, 'ns')
    
    timedelta_value = datetime - ref_time
    
    return timedelta_value