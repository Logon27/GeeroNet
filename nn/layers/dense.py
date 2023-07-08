from jax import random
from jax.nn.initializers import glorot_normal, normal
import jax.numpy as jnp
import logging

def Dense(out_dim, weight_init=glorot_normal(), bias_init=normal()):
  """Layer constructor function for a dense (fully-connected) layer."""

  def init_fun(rng, input_shape):
    """Takes in a input shape such as 784. The out_dim is just a single digit such as 3."""
    k1, k2 = random.split(rng)
    weights, bias = weight_init(k1, (input_shape, out_dim)), bias_init(k2, (1, out_dim))
    logging.info('W_shape: {}, B_shape: {}'.format(weights.shape, bias.shape))
    return out_dim, (weights, bias)
    
  def apply_fun(params, inputs, **kwargs):
    if inputs.ndim <= 1:
      raise ValueError("Input must be at least 2 dimensional. This helps eliminate any confusion with mixing vector and matrix multiplication.")
    
    weights, bias = params
    logging.info('{:<15} @ {:<15} + {:<15}'.format("I"+str(inputs.shape), "W"+str(weights.shape), "B"+str(bias.shape)))
    return jnp.matmul(inputs, weights) + bias
    
  return init_fun, apply_fun