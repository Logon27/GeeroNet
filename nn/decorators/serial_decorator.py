from typing import List
from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
import functools
import logging

def debug_decorator(serial_debug):
    """
    Decorator to print debug information of the forward pass for INFO2 log level.
    """
    @functools.wraps(serial_debug)
    def serial(*args, **kwargs):
        if logging.getLevelName(logging.root.level) == "INFO2":
            init_fun_debug, apply_fun_debug = serial_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng: PRNGKey, input_shape):
                output_shape, params = init_fun_debug(rng, input_shape)
                return output_shape, params
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params: List[Params], inputs: ArrayLike, **kwargs):
                result = apply_fun_debug(params, inputs, **kwargs)
                return result

            return init_fun, apply_fun
        else:
            return serial_debug(*args, **kwargs)
        
    return serial