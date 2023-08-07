from jax import random
import jax.numpy as jnp

# Using a modified implementation
# https://github.com/google/jax/issues/16218
def Dropout(rate, mode='train'):
  """Layer construction function for a dropout layer with given rate."""
  def init_fun(rng, input_shape):
    return input_shape, ()
  def apply_fun(params, inputs, **kwargs):    
    mode = kwargs.get('mode', None)
    if mode == 'train':
      rng = kwargs.get('rng', None)
      if rng is None:
        msg = ("Dropout layer requires apply_fun to be called with a PRNG key "
              "argument. That is, instead of `apply_fun(params, inputs)`, call "
              "it like `apply_fun(params, inputs, rng)` where `rng` is a "
              "jax.random.PRNGKey value.")
        raise ValueError(msg)
    
      keep = random.bernoulli(rng, rate, inputs.shape)
      return jnp.where(keep, inputs / rate, 0)
    else:
      return inputs
  return init_fun, apply_fun