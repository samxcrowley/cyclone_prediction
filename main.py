import model
import utils
import prediction
import plotting

import xarray
import jax
import utils
import numpy as np
from graphcast import graphcast
from graphcast import data_utils as g_data_utils

import model

def main():

    # testing model and data
    model_path = "/scratch/ll44/sc6160/model/params_GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
    dataset_path = "/scratch/ll44/sc6160/data/ERA5/source-era5_date-2022-01-01_res-1.0_levels-13_steps-40.nc"

    # full model and data
    # model_path = "/scratch/ll44/sc6160/model/params_GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
    # dataset_path = "/scratch/ll44/sc6160/data/ERA5/source-era5_date-2022-01-01_res-1.0_levels-37_steps-40.nc"

    # load model
    params, model_config, task_config = model.load_model_from_cache(model_path)
    state = {}

    # print(xarray.open_dataset(dataset_path))
    
    if not model.data_valid_for_model(dataset_path, model_config, task_config):
        raise ValueError(f"Invalid dataset file {dataset_path}.")
    
    example_batch = xarray.open_dataset(dataset_path)

    # extract train and eval data
    train_inputs, train_targets, train_forcings, \
        eval_inputs, eval_targets, eval_forcings = \
            utils.extract_inputs_targets_forcings(example_batch, task_config)
    
    init_jitted = jax.jit(model.with_configs(model.run_forward.init, model_config, task_config))

    if params is None:
        params, state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=train_inputs,
            targets_template=train_targets,
            forcings=train_forcings)
    
    run_forward_jitted = \
        model.drop_state(model.with_params(jax.jit(model.with_configs(model.run_forward.apply, model_config, task_config)), params, state))

    assert model_config.resolution in (0, 360. / eval_inputs.sizes["lon"]), (
        "Model resolution doesn't match the data resolution. You likely want to"
        "re-filter the dataset list, and download the correct data.")
    
    # run predictions
    preds = prediction.run_predictions(run_forward_jitted, eval_inputs, eval_targets, eval_forcings)

    # save data
    preds.to_netcdf("/scratch/ll44/sc6160/out/preds.nc")
    eval_targets.to_netcdf("/scratch/ll44/sc6160/out/evals.nc")

    metrics = {'2m_temperature': "temp_",
           'mean_sea_level_pressure': "mslp_",
           'total_precipitation_6hr': "prec_"
        #    'u_component_of_wind': "u_wind_"
           }
    
    plotting.plot_metrics(preds, eval_targets, metrics, utils.AUS_LAT_BOUNDS, utils.AUS_LON_BOUNDS)

    # loss_fn_jitted = \
    #     model.drop_state(model.with_params(jax.jit(model.with_configs(model.loss_fn.apply, model_config, task_config)), params, state))
    # grads_fn_jitted = \
    #     model.with_params(jax.jit(model.with_configs(model.grads_fn, model_config, task_config)), params, state)


if __name__ == "__main__":
    main()