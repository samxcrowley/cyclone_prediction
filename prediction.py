import numpy as np
import jax
from graphcast import rollout

def run_predictions(run_forward_jitted, eval_inputs, eval_targets, eval_forcings):
    
    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings)
    
    return predictions