from jax import random
import jax.numpy as jnp
from jax.nn.initializers import glorot_normal, normal
import logging

def Dense(out_dim, weight_init=glorot_normal(), bias_init=normal()):
  """Layer constructor function for a dense (fully-connected) layer."""

  def init_fun(rng, input_shape):
    """Takes in a 2D input shape such as (3, 1). The out_dim is just a single digit such as 3 however."""
    k1, k2 = random.split(rng)
    weights, bias = weight_init(k1, (out_dim, input_shape)), bias_init(k2, (out_dim, 1))
    return out_dim, (weights, bias)
    
  def apply_fun(params, inputs, **kwargs):
    if inputs.ndim <= 1:
      raise ValueError("Input must be at least 2 dimensional. This helps eliminate any confusion with mixing vector and matrix multiplication.")
    
    weights, bias = params
    logging.info('W{} @ I{} + B{}'.format(weights.shape, inputs.shape, bias.shape))
    return jnp.matmul(weights, inputs) + bias
    
  return init_fun, apply_fun