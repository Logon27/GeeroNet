from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
from nn.helpers.envvar import getenv
import functools

def debug_decorator(flatten_debug):
    """
    Decorator to print debug information of the forward pass for INFO2 log level.
    """
    @functools.wraps(flatten_debug)
    def Flatten(*args, **kwargs):
        if getenv("MODEL_DEBUG", True):
            init_fun_debug, apply_fun_debug = flatten_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng: PRNGKey, input_shape: ArrayLike):
                output_shape, () = init_fun_debug(rng, input_shape)
                debug_msg = "Flatten(Input Shape: {}, Output Shape: {})".format(input_shape, output_shape)
                debug_msg = debug_msg.replace("-1", "*")
                print(debug_msg)
                return output_shape, ()
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params: Params, inputs: ArrayLike, **kwargs):
                result = apply_fun_debug(params, inputs, **kwargs)
                print("Flatten{} = Output Shape: {}".format(
                    inputs.shape, result.shape
                ))
                return result

            return init_fun, apply_fun
        else:
            return flatten_debug(*args, **kwargs)
        
    return Flatten