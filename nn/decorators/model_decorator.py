from typing import List
from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
import time
import jax
import functools
import logging

def model_decorator(tuple_of_functions):
    """
    Decorator that wraps the outer function of a model in order to track execution time and provide text based separators
    """
    if logging.getLevelName(logging.root.level) == "INFO2":
        tuple_init_fun = tuple_of_functions[0]
        tuple_apply_fun = tuple_of_functions[1]

        @functools.wraps(tuple_init_fun)
        def init_fun(rng: PRNGKey, input_shape):
            jax.debug.print("=== Start Init Fun Execution ===")
            start_time_forward = time.time()

            output_shape, params, states = tuple_init_fun(rng, input_shape)

            end_time_forward = time.time()
            time_elapsed_ms = (end_time_forward - start_time_forward)
            jax.debug.print("=== End Init Fun Execution ===\n")
            jax.debug.print("Initialization Took: {:.2f} seconds".format(time_elapsed_ms))
            jax.debug.breakpoint(num_frames=1)

            return output_shape, params, states
        
        @functools.wraps(tuple_apply_fun)
        def apply_fun(params: List[Params], states, inputs: ArrayLike, **kwargs):
            jax.debug.print("=== Start Forward Pass Execution ===")
            start_time_forward = time.time()

            result, states = tuple_apply_fun(params, states, inputs, **kwargs)

            end_time_forward = time.time()
            time_elapsed_ms = (end_time_forward - start_time_forward) * 1000
            jax.debug.print("=== End Forward Pass Execution ===\n")
            jax.debug.print("Forward Pass Took: {:.2f} ms".format(time_elapsed_ms))
            jax.debug.breakpoint(num_frames=1)
            return result, states
    
        return (init_fun, apply_fun)
    else:
        return tuple_of_functions        
