import model
from data_utils import select, prepare_data_dict, extract_inputs_targets_forcings
from prediction import run_predictions
from plotting import scale, plot_data

import xarray
import jax
import data_utils
import numpy as np
from graphcast import graphcast
from graphcast import data_utils as g_data_utils

import model

def main():
    
    # load model
    model_path = "/scratch/ll44/sc6160/model/params_GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
    # model_path = "/scratch/ll44/sc6160/model/params_GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
    params, model_config, task_config = model.load_model_from_cache(model_path)
    state = {}

    # load weather (test) data
    dataset_path = "/scratch/ll44/sc6160/test_data/source:era5_date:2022-01-01_res:1.0_levels:13_steps:04.nc"
    # dataset_path = "/scratch/ll44/sc6160/test_data/dataset_source-era5_date-2022-01-01_res-0.25_levels-37_steps-04.nc"

    def data_valid_for_model(
        file_name: str,
        model_config: graphcast.ModelConfig,
        task_config: graphcast.TaskConfig):

        file_parts = data_utils.parse_file_parts(file_name.removesuffix(".nc"))

        return (
            model_config.resolution in (0, float(file_parts["res"])) and
            len(task_config.pressure_levels) == int(file_parts["levels"]) and
            (
                ("total_precipitation_6hr" in task_config.input_variables and
                    file_parts["source"] in ("era5", "fake")) or
                ("total_precipitation_6hr" not in task_config.input_variables and
                    file_parts["source"] in ("hres", "fake"))
            )
        )
    
    if not data_valid_for_model(dataset_path, model_config, task_config):
        raise ValueError("Invalid dataset file, rerun the cell above and choose a valid dataset file.")
    
    example_batch = xarray.open_dataset(dataset_path, engine='netcdf4')

    # extract train and eval data
    train_inputs, train_targets, train_forcings, \
        eval_inputs, eval_targets, eval_forcings = \
            extract_inputs_targets_forcings(example_batch, task_config)
    
    init_jitted = jax.jit(model.with_configs(model.run_forward.init, model_config, task_config))

    if params is None:
        params, state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=train_inputs,
            targets_template=train_targets,
            forcings=train_forcings)
        
    # loss_fn_jitted = \
    #     model.drop_state(model.with_params(jax.jit(model.with_configs(model.loss_fn.apply, model_config, task_config)), params, state))
    # grads_fn_jitted = \
    #     model.with_params(jax.jit(model.with_configs(model.grads_fn, model_config, task_config)), params, state)
    run_forward_jitted = \
        model.drop_state(model.with_params(jax.jit(model.with_configs(model.run_forward.apply, model_config, task_config)), params, state))

    assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
        "Model resolution doesn't match the data resolution. You likely want to"
        "re-filter the dataset list, and download the correct data.")
    
    # run predictions
    predictions = run_predictions(run_forward_jitted, eval_inputs, eval_targets, eval_forcings)
    print()
    print("########## PREDICTIONS")
    print(predictions["2m_temperature"])
    print()
    print()
    print()

    # save data
    predictions.to_netcdf("/scratch/ll44/sc6160/out/predictions_example.nc")
    eval_targets.to_netcdf("/scratch/ll44/sc6160/out/eval_example.nc")

    # subset and prepare data for plotting
    lat_bounds = [-45, -10]
    lon_bounds = [110, 155]
    # mslp_data_dict = prepare_data_dict(predictions, eval_targets, 'mean_sea_level_pressure', lat_bounds, lon_bounds)
    temp_data_dict = prepare_data_dict(predictions, eval_targets, '2m_temperature', lat_bounds, lon_bounds)
    # prec_data_dict = prepare_data_dict(predictions, eval_targets, 'total_precipitation_6hr', lat_bounds, lon_bounds)
    # shum_data_dict = prepare_data_dict(predictions, eval_targets, 'specific_humidity', lat_bounds, lon_bounds)
    # wind_data_dict = prepare_data_dict(predictions, eval_targets, 'u_component_of_wind', lat_bounds, lon_bounds)
    
    # plot data
    # plot_data(mslp_data_dict, "Mean Sea Level Pressure (Australia Region)", plot_size=5, robust=True, cols=3, output_prefix="mslp_")
    plot_data(temp_data_dict, "2m Temperature (Australia Region)", plot_size=5, robust=True, cols=3, output_prefix="temp_")
    # plot_data(prec_data_dict, "2m Temperature (Australia Region)", plot_size=5, robust=True, cols=3, output_prefix="prec_")
    # plot_data(shum_data_dict, "2m Temperature (Australia Region)", plot_size=5, robust=True, cols=3, output_prefix="shum_")
    # plot_data(wind_data_dict, "2m Temperature (Australia Region)", plot_size=5, robust=True, cols=3, output_prefix="wind_")


if __name__ == "__main__":
    main()