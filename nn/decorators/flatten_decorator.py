from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
import logging
import jax
import functools

def debug_decorator(flatten_debug):
    """
    Decorator to print debug information of the forward pass for INFO2 log level.
    """
    @functools.wraps(flatten_debug)
    def Flatten(*args, **kwargs):
        if logging.getLevelName(logging.root.level) == "INFO2":
            init_fun_debug, apply_fun_debug = flatten_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng: PRNGKey, input_shape: ArrayLike):
                output_shape, () = init_fun_debug(rng, input_shape)
                jax.debug.print("Flatten(Input Shape: {}, Output Shape: {})",
                    input_shape, output_shape
                )
                return output_shape, ()
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params: Params, inputs: ArrayLike, **kwargs):
                result = apply_fun_debug(params, inputs, **kwargs)
                jax.debug.print("Flatten{} = Output Shape: {}",
                    inputs.shape, result.shape
                )
                return result

            return init_fun, apply_fun
        else:
            return flatten_debug(*args, **kwargs)
        
    return Flatten