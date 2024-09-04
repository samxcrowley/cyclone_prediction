import model
import utils
import prediction
import plotting
import dataclasses

import xarray
import jax
import utils
import numpy as np
from graphcast import graphcast, data_utils

import model

def main():

    model_path = "/scratch/ll44/sc6160/model/params_GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
    
    year = 2023
    month = 4
    dataset_path = "/scratch/ll44/sc6160/data/2023-04/source-era5_data-2023-4_res-0.25_levels-37_tc-ilsa.nc"

    # load model
    params, model_config, task_config = model.load_model_from_cache(model_path)
    state = {}
    
    example_batch = xarray.open_dataset(dataset_path)
    example_batch = example_batch.drop_vars('relative_vorticity')

    train_steps = example_batch.sizes['time'] - 2
    eval_steps = example_batch.sizes['time'] - 2

    train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("6h", f"{train_steps * 6}h"), **dataclasses.asdict(task_config))
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch, target_lead_times=slice("6h", f"{eval_steps * 6}h"), **dataclasses.asdict(task_config))
    
    init_jitted = jax.jit(model.with_configs(model.run_forward.init, model_config, task_config))

    if params is None:
        params, state = init_jitted(
            rng=jax.random.PRNGKey(0),
            inputs=train_inputs,
            targets_template=train_targets,
            forcings=train_forcings)
    
    run_forward_jitted = \
        model.drop_state(model.with_params(jax.jit(model.with_configs(model.run_forward.apply, model_config, task_config)), params, state))
    
    # run predictions
    preds = prediction.run_predictions(run_forward_jitted, eval_inputs, eval_targets, eval_forcings)

    # save data
    preds.to_netcdf(f"/scratch/ll44/sc6160/out/preds_ilsa.nc")
    eval_targets.to_netcdf(f"/scratch/ll44/sc6160/out/evals_ilsa.nc")

    # plot metrics
    metrics = {
        '2m_temperature': "ilsa_temp_",
        'mean_sea_level_pressure': "ilsa_mslp_",
        'total_precipitation_6hr': "ilsa_prec_"
    }
    plotting.plot_metrics(preds, eval_targets, metrics, utils.AUS_LAT_BOUNDS, utils.AUS_LON_BOUNDS)

if __name__ == "__main__":
    main()