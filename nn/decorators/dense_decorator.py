from typing import Tuple
from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
import logging
import jax
import functools

def debug_decorator(dense_debug):
    """
    Decorator to print debug information of the forward pass for INFO2 log level.
    """
    @functools.wraps(dense_debug)
    def Dense(*args, **kwargs):
        if logging.getLevelName(logging.root.level) == "INFO2":
            init_fun_debug, apply_fun_debug = dense_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng: PRNGKey, input_shape: Tuple):
                output_shape, (weights, bias) = init_fun_debug(rng, input_shape)
                debug_msg = "Dense(Input Shape: {}, Output Shape: {} => Weight Shape: {}, Bias Shape: {}".format(input_shape, output_shape, weights.shape, bias.shape)
                debug_msg = debug_msg.replace("-1", "*")
                jax.debug.print(debug_msg)
                return output_shape, (weights, bias)
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params: Params, inputs: ArrayLike, **kwargs):
                weights, bias = params
                result = apply_fun_debug(params, inputs, **kwargs)
                jax.debug.print("I({}, {}) @ W({}, {}) + B({}, {}) = Output Shape: {}",
                    inputs.shape[0], inputs.shape[1], weights.shape[0], weights.shape[1], bias.shape[0], bias.shape[1], result.shape
                )
                return result

            return init_fun, apply_fun
        else:
            return dense_debug(*args, **kwargs)
        
    return Dense