import jax
from jax import grad
from jax import random
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal, ones, zeros

def Dense(out_dim, weight_init=glorot_normal(), bias_init=normal()):
  """Layer constructor function for a dense (fully-connected) layer."""

  def init_fun(rng, input_shape):
    """Takes in a 2D input shape such as (3, 1). The out_dim is just a single digit such as 3 however."""
    k1, k2 = random.split(rng)
    weights, bias = weight_init(k1, (out_dim, input_shape)), bias_init(k2, (out_dim, 1))
    return out_dim, (weights, bias)
    
  def apply_fun(params, inputs, **kwargs):
    weights, bias = params

    # jax.debug.print("weights: {weights}", weights=weights.shape)
    # jax.debug.print("inputs: {inputs}", inputs=inputs.shape)
    return jnp.dot(weights, inputs) + bias
    
  return init_fun, apply_fun