import functools
import haiku as hk
import xarray
import jax

from graphcast import autoregressive, casting, checkpoint, \
    graphcast, normalization, xarray_tree, xarray_jax

import data_utils

def load_model_from_cache(model_path):

    with open(model_path, "rb") as f:

        ckpt = checkpoint.load(f, graphcast.CheckPoint)
        params = ckpt.params
        model_config = ckpt.model_config
        task_config = ckpt.task_config

        print("Model description:\n", ckpt.description, "\n")
        print("Model license:\n", ckpt.license, "\n")
    
    print(model_config)

    return params, model_config, task_config

# constructs and wraps the GraphCast predictor
def construct_wrapped_graphcast(model_config: graphcast.ModelConfig, task_config: graphcast.TaskConfig):
    
    # load normalisation data
    diffs_stddev_path = xarray.load_dataset("/scratch/ll44/sc6160/model/diffs_stddev_by_level.nc").compute()
    mean_by_level_path = xarray.load_dataset("/scratch/ll44/sc6160/model/mean_by_level.nc").compute()
    stddev_by_level_path = xarray.load_dataset("/scratch/ll44/sc6160/model/stddev_by_level.nc").compute()

    # deeper one-step predictor.
    predictor = graphcast.GraphCast(model_config, task_config)

    # modify inputs/outputs to `graphcast.GraphCast` to handle conversion to from/to float32 to/from BFloat16.
    predictor = casting.Bfloat16Cast(predictor)

    # modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from BFloat16 happens after applying normalization to the inputs/targets.
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_path,
        mean_by_level=mean_by_level_path,
        stddev_by_level=stddev_by_level_path)

    # wraps everything so the one-step model can produce trajectories.
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)

    return predictor

@hk.transform_with_state
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)

@hk.transform_with_state
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  loss, diagnostics = predictor.loss(inputs, targets, forcings)
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.
def with_configs(fn, model_config, task_config):
  return functools.partial(fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn, params, state):
  return functools.partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

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