from typing import List
from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
from nn.helpers.envvar import getenv
import functools


def debug_decorator(parallel_debug):
    """
    Decorator to print debug information of the forward pass for INFO2 log level.
    """
    @functools.wraps(parallel_debug)
    def parallel(*args, **kwargs):
        if getenv("MODEL_DEBUG", True):
            init_fun_debug, apply_fun_debug = parallel_debug(*args, **kwargs)

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
            return parallel_debug(*args, **kwargs)
        
    return parallel