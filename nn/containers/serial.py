from jax import random
import jax.numpy as jnp

def serial(*layers):
  """Combinator for composing layers in serial.

  Args:
    *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

  Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the serial
    composition of the given sequence of layers.
  """
  nlayers = len(layers)
  init_funs, apply_funs = zip(*layers)
  def init_fun(rng, input_shape):
    params = []
    for init_fun in init_funs:
      rng, layer_rng = random.split(rng)
      input_shape, param = init_fun(layer_rng, input_shape)
      params.append(param)
    return input_shape, params
  def apply_fun(params, inputs, **kwargs):
    rng = kwargs.pop('rng', None)
    rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
    for fun, param, rng in zip(apply_funs, params, rngs):
      inputs = fun(param, inputs, rng=rng, **kwargs)
    return inputs
  return init_fun, apply_fun