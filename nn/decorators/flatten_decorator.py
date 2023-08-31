from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
import functools
import logging
import jax


def debug_decorator(flatten_debug):
    """
    Decorator to wrap the Flatten layer.
    """
    @functools.wraps(flatten_debug)
    def Flatten(*args, **kwargs):
        if logging.getLevelName(logging.root.level) == "INFO2":
            init_fun_debug, apply_fun_debug = flatten_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng: PRNGKey, input_shape: ArrayLike):
                output_shape, () = init_fun_debug(rng, input_shape)
                debug_msg = "Flatten(Input Shape: {}, Output Shape: {})".format(input_shape, output_shape)
                debug_msg = debug_msg.replace("-1", "*")
                jax.debug.print(debug_msg)
                return output_shape, ()
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params: Params, inputs: ArrayLike, **kwargs):
                result = apply_fun_debug(params, inputs, **kwargs)
                jax.debug.print("Flatten{} = Output Shape: {}".format(
                    inputs.shape, result.shape
                ))
                return result

            return init_fun, apply_fun
        else:
            return flatten_debug(*args, **kwargs)
        
    return Flatten