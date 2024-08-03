from plotting import plot_data
from data_utils import prepare_data_dict
import xarray as xr
import numpy as np

pred_file_path = "out/predictions_example.nc"
eval_file_path = "out/eval_example.nc"

pred_data = xr.open_dataset(pred_file_path)
eval_data = xr.open_dataset(eval_file_path)

eval_data = eval_data.transpose('time', 'batch', 'lat', 'lon', 'level')

# subset and prepare data for plotting
lat_bounds = [-45, -10]
lon_bounds = [110, 155]
# mslp_data_dict = prepare_data_dict(predictions, eval_targets, 'mean_sea_level_pressure', lat_bounds, lon_bounds)
temp_data_dict = prepare_data_dict(pred_data, eval_data, '2m_temperature', lat_bounds, lon_bounds)
# prec_data_dict = prepare_data_dict(pred_data, eval_data, 'total_precipitation_6hr', lat_bounds, lon_bounds)
# shum_data_dict = prepare_data_dict(pred_data, eval_data, 'specific_humidity', lat_bounds, lon_bounds)
# wind_data_dict = prepare_data_dict(pred_data, eval_data, 'u_component_of_wind', lat_bounds, lon_bounds)

# plot data
# plot_data(mslp_data_dict, "Mean Sea Level Pressure (Australia Region)", plot_size=5, robust=True, cols=3, output_prefix="mslp_")
plot_data(temp_data_dict, "2m Temperature (Australia Region)", plot_size=5, robust=True, cols=3, output_prefix="temp_")
# plot_data(prec_data_dict, "2m Temperature (Australia Region)", plot_size=5, robust=True, cols=3, output_prefix="prec_")
# plot_data(shum_data_dict, "2m Temperature (Australia Region)", plot_size=5, robust=True, cols=3, output_prefix="shum_")
# plot_data(wind_data_dict, "2m Temperature (Australia Region)", plot_size=5, robust=True, cols=3, output_prefix="wind_")