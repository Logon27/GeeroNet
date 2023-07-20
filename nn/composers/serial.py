from jax import random
import logging
from ..print_utils.print_params import *
import jax

def serial(*layers):
  """Combinator for composing layers in serial.

  Args:
    *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

  Returns:
    A new layer, meaning an (init_fun, apply_fun) pair, representing the serial
    composition of the given sequence of layers.
  """
  num_layers = len(layers)
  init_funs, apply_funs = zip(*layers)
  def init_fun(rng, input_shape):
    params = []
    for init_fun in init_funs:
      rng, layer_rng = random.split(rng)
      input_shape, param = init_fun(layer_rng, input_shape)
      params.append(param)

    if logging.getLevelName(logging.root.level) == "INFO":
      print_params(params)

    # input_shape at this point represents the final layer's output dimension
    return input_shape, params
  def apply_fun(params, inputs, **kwargs):
    if logging.getLevelName(logging.root.level) == "INFO2":
      jax.debug.print("=== Start Forward Pass Execution ===")

    rng = kwargs.pop('rng', None)
    rngs = random.split(rng, num_layers) if rng is not None else (None,) * num_layers
    for fun, param, rng in zip(apply_funs, params, rngs):
      inputs = fun(param, inputs, rng=rng, **kwargs)
    
    if logging.getLevelName(logging.root.level) == "INFO2":
        jax.debug.print("=== End Forward Pass Execution ===")
        jax.debug.breakpoint()
    
    return inputs
  return init_fun, apply_fun