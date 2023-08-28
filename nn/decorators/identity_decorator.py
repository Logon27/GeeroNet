from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
from nn.helpers.envvar import getenv
import functools

def debug_decorator(identity_debug):
    """
    Decorator to print debug information of the forward pass for INFO2 log level.
    """
    @functools.wraps(identity_debug)
    def Identity(*args, **kwargs):
        if getenv("MODEL_DEBUG", 1):
            init_fun_debug, apply_fun_debug = identity_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng: PRNGKey, input_shape: ArrayLike):
                output_shape, () = init_fun_debug(rng, input_shape)
                debug_msg = "Identity(Input Shape: {}, Output Shape: {})".format(input_shape, output_shape)
                debug_msg = debug_msg.replace("-1", "*")
                print(debug_msg)
                return output_shape, ()
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params: Params, inputs: ArrayLike, **kwargs):
                result = apply_fun_debug(params, inputs, **kwargs)
                print("Identity{} = Output Shape: {}".format(
                    inputs.shape, result.shape
                ))
                return result

            return init_fun, apply_fun
        else:
            return identity_debug(*args, **kwargs)
        
    return Identity