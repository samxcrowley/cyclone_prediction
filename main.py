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

    model_path = "/scratch/ll44/sc6160/model/params_GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
    # model_path = "/scratch/ll44/sc6160/model/params_GraphCast_small - ERA5 1979-2015 - resolution 1.0 - pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
    
    dataset_path = "/scratch/ll44/sc6160/data/2022-01/source-era5_res-0.25_levels-37_merged_resampled_6h_full.nc"

    # load model
    params, model_config, task_config = model.load_model_from_cache(model_path)
    state = {}
    
    # if not model.data_valid_for_model(dataset_path, model_config, task_config):
    #     raise ValueError(f"Invalid dataset file {dataset_path}.")
    
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
    preds.to_netcdf(f"/scratch/ll44/sc6160/out/preds_jan_22.nc")
    eval_targets.to_netcdf(f"/scratch/ll44/sc6160/out/evals_jan_22.nc")

    metrics = {'2m_temperature': "temp_",
        'mean_sea_level_pressure': "mslp_",
        'total_precipitation_6hr': "prec_"
    }
    
    plotting.plot_metrics(preds, eval_targets, metrics, utils.AUS_LAT_BOUNDS, utils.AUS_LON_BOUNDS)


if __name__ == "__main__":
    main()