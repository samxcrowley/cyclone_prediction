import os
import model
import utils
import plotting
import dataclasses

import xarray
import jax
import utils
import numpy as np
from graphcast import data_utils, rollout

import model

def main():

    model_path = "/scratch/ll44/sc6160/model/params_GraphCast - ERA5 1979-2017 - resolution 0.25 - pressure levels 37 - mesh 2to6 - precipitation input and output.npz"
    
    tc_name, tc_id, _, _ = utils.load_tc_data()
    dataset_path = f"/scratch/ll44/sc6160/data/obs/{tc_name}_{tc_id}_obs_data.nc"

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
    preds = run_predictions(run_forward_jitted, eval_inputs, eval_targets, eval_forcings)

    # save data
    folder_path = f"/scratch/ll44/sc6160/out/pred/"
    os.makedirs(folder_path, exist_ok=True)
    preds.to_netcdf(os.path.join(folder_path, f"{tc_name}_{tc_id}_pred_data.nc"))
    eval_targets.to_netcdf(os.path.join(folder_path, f"{tc_name}_{tc_id}_eval_data.nc"))

def run_predictions(run_forward_jitted, eval_inputs, eval_targets, eval_forcings):
    
    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings)
    
    return predictions

if __name__ == "__main__":
    main()