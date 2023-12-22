from typing import List
from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
import functools
import logging
import jax
from nn.decorators.layer_decorator import debug_decorator as layer_debug_decorator


def debug_decorator(parallel_debug):
    """
    Decorator to wrap the individual parallel layers with separators and wrap the entire parallel layer.
    """
    @functools.wraps(parallel_debug)
    def parallel(*args, **kwargs):
        if logging.getLevelName(logging.root.level) == "INFO2":
            # Wrap each individual layer defined in the parallel layer list. For the last layer, do not print a layer separator.
            # These functions have to be wrapped before the parallel layer execution.
            wrapped_args = []
            for index, arg in enumerate(args):
                if index < len(args) - 1:
                    wrapped_args.append(layer_debug_decorator(arg, True))
                else:
                    wrapped_args.append(layer_debug_decorator(arg, False))
            args = wrapped_args

            init_fun_debug, apply_fun_debug = parallel_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng: PRNGKey, input_shape):
                jax.debug.print("=" * 100)
                output_shape, params, state = init_fun_debug(rng, input_shape)
                jax.debug.print("=" * 100)
                return output_shape, params, state
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params: List[Params], state, inputs: ArrayLike, **kwargs):
                jax.debug.print("=" * 100)
                result, state = apply_fun_debug(params, state, inputs, **kwargs)
                jax.debug.print("=" * 100)
                return result, state

            return init_fun, apply_fun
        else:
            return parallel_debug(*args, **kwargs)
        
    return parallel