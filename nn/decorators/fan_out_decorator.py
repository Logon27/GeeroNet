from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
import functools
import logging
import jax
import jax.numpy as jnp


def debug_decorator(fanout_debug):
    """
    Decorator to print debug information of the forward pass for INFO2 log level.
    """
    @functools.wraps(fanout_debug)
    def FanOut(*args, **kwargs):
        if logging.getLevelName(logging.root.level) == "INFO2":
            init_fun_debug, apply_fun_debug = fanout_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng: PRNGKey, input_shape: ArrayLike):
                output_shape, () = init_fun_debug(rng, input_shape)
                debug_msg = "FanOut(Input Shape: {}, Output Shape: {})".format(input_shape, output_shape)
                debug_msg = debug_msg.replace("-1", "*")
                jax.debug.print(debug_msg)
                return output_shape, ()
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params: Params, inputs: ArrayLike, **kwargs):
                result = apply_fun_debug(params, inputs, **kwargs)
                jax.debug.print("FanOut({}) = Output Shape: {}".format(
                    inputs.shape, jnp.array(result).shape
                ))
                return result

            return init_fun, apply_fun
        else:
            return fanout_debug(*args, **kwargs)
        
    return FanOut