from typing import List
from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
import time
import jax
import functools
import logging

def model_decorator(tuple_of_functions):
    if logging.getLevelName(logging.root.level) == "INFO2":
        tuple_init_fun = tuple_of_functions[0]
        tuple_apply_fun = tuple_of_functions[1]

        @functools.wraps(tuple_init_fun)
        def init_fun(rng: PRNGKey, input_shape):
            print("=== Start Init Fun Execution ===")
            start_time_forward = time.time()

            output_shape, params = tuple_init_fun(rng, input_shape)

            end_time_forward = time.time()
            time_elapsed_ms = (end_time_forward - start_time_forward)
            print("Initialization Took: {:.2f} seconds".format(time_elapsed_ms))
            print("=== End Init Fun Execution ===\n")
            jax.debug.breakpoint()

            return output_shape, params
        
        @functools.wraps(tuple_apply_fun)
        def apply_fun(params: List[Params], inputs: ArrayLike, **kwargs):
            # Before Function Execution
            print("=== Start Forward Pass Execution ===")
            start_time_forward = time.time()

            result = tuple_apply_fun(params, inputs, **kwargs)

            # After Function Execution
            end_time_forward = time.time()
            time_elapsed_ms = (end_time_forward - start_time_forward) * 1000
            print("Forward Pass Took: {:.2f} ms".format(time_elapsed_ms))
            print("=== End Forward Pass Execution ===\n")
            jax.debug.breakpoint()
            # returning the value to the original frame
            return result
    
        return (init_fun, apply_fun)
    else:
        return tuple_of_functions        
