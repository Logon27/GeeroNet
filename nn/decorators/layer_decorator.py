from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
from typing import List
import functools
import logging
import jax


def debug_decorator(tuple_of_functions, print_separator):
    """
    Generic decorator to wrap the init and apply functions. This is used to print separators for parallel layers.
    """
    if logging.getLevelName(logging.root.level) == "INFO2":
        tuple_init_fun = tuple_of_functions[0]
        tuple_apply_fun = tuple_of_functions[1]

        @functools.wraps(tuple_init_fun)
        def init_fun(rng: PRNGKey, input_shape):
            output_shape, params = tuple_init_fun(rng, input_shape)
            if print_separator:
                jax.debug.print("-" * 100)
            return output_shape, params
        
        @functools.wraps(tuple_apply_fun)
        def apply_fun(params: List[Params], inputs: ArrayLike, **kwargs):
            result = tuple_apply_fun(params, inputs, **kwargs)
            if print_separator:
                jax.debug.print("-" * 100)
            return result
    
        return (init_fun, apply_fun)
    else:
        return tuple_of_functions        
