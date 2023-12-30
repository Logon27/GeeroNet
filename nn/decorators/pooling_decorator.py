import functools
import logging
import jax
import sys
from typing import Callable, TypeVar
if sys.version_info < (3, 10):
    from typing_extensions import ParamSpec
else:
    from typing import ParamSpec


P = ParamSpec("P")
R = TypeVar("R")

def debug_decorator(pooling_debug: Callable[P, R], func_name) -> Callable[P, R]:
    """
    Decorator to wrap the Pooling layers.
    """
    @functools.wraps(pooling_debug)
    def Pooling(*args: P.args, **kwargs: P.kwargs) -> R:
        if logging.getLevelName(logging.root.level) == "INFO2":
            init_fun_debug, apply_fun_debug = pooling_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng, input_shape):
                output_shape, (), state = init_fun_debug(rng, input_shape)
                debug_msg = "{}(Input Shape: {}, Output Shape: {})".format(func_name, input_shape, output_shape)
                debug_msg = debug_msg.replace("-1", "*")
                jax.debug.print(debug_msg)
                return output_shape, (), state
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params, state, inputs, **kwargs):
                result, state = apply_fun_debug(params, state, inputs, **kwargs)
                jax.debug.print("{}{} = Output Shape: {}".format(
                    func_name, inputs.shape, result.shape
                ))
                return result, state

            return init_fun, apply_fun
        else:
            return pooling_debug(*args, **kwargs)
        
    return Pooling