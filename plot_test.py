import plotting, data_utils
import xarray as xr
import numpy as np

pred_file_path = "out/predictions_example.nc"
eval_file_path = "out/eval_example.nc"

pred_data = xr.open_dataset(pred_file_path)
eval_data = xr.open_dataset(eval_file_path)

eval_data = eval_data.transpose('time', 'batch', 'lat', 'lon', 'level')

plot_size = 5
plot_max_steps = 4
plot_pred_variable = "mean_sea_level_pressure"
plot_pred_level = 500

data = {
    "Targets": plotting.scale(data_utils.select(eval_data, plot_pred_variable, plot_pred_level, plot_max_steps), robust=True),
    "Predictions": plotting.scale(data_utils.select(pred_data, plot_pred_variable, plot_pred_level, plot_max_steps), robust=True),
    "Diff": plotting.scale((data_utils.select(eval_data, plot_pred_variable, plot_pred_level, plot_max_steps) -
                        data_utils.select(pred_data, plot_pred_variable, plot_pred_level, plot_max_steps)),
                    robust=True, center=0),
}

fig_title = plot_pred_variable
if "level" in pred_data[plot_pred_variable].coords:
    fig_title += f" at {plot_pred_level} hPa"

plotting.plot_data(data, fig_title, plot_size, robust=True)

lat_bounds = [-45, -10]
lon_bounds = [110, 155]

mslp_data_dict = data_utils.prepare_data_dict(pred_data, eval_data, 'mean_sea_level_pressure', lat_bounds, lon_bounds)
temp_data_dict = data_utils.prepare_data_dict(pred_data, eval_data, '2m_temperature', lat_bounds, lon_bounds)

plotting.plot_data(mslp_data_dict, "Mean Sea Level Pressure (Australia Region)", plot_size=5, robust=True, cols=3, output_prefix="mslp_")
plotting.plot_data(temp_data_dict, "2m Temperature (Australia Region)", plot_size=5, robust=True, cols=3, output_prefix="temp_")