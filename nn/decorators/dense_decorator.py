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
            output_shape = args[0]

            @functools.wraps(init_fun_debug)
            def init_fun(rng: PRNGKey, input_shape: ArrayLike):
                output_shape, (weights, bias) = init_fun_debug(rng, input_shape)
                jax.debug.print("Dense(Input Shape: {}, Output Shape: {})",
                    input_shape, output_shape
                )
                return output_shape, (weights, bias)
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params: Params, inputs: ArrayLike, **kwargs):
                weights, bias = params
                jax.debug.print("I({}, {}) @ W({}, {}) + B({}, {}) = output_shape: {}",
                    inputs.shape[0], inputs.shape[1], weights.shape[0], weights.shape[1], bias.shape[0], bias.shape[1], output_shape
                )
                return apply_fun_debug(params, inputs, **kwargs)

            return init_fun, apply_fun
        else:
            return dense_debug(*args, **kwargs)
        
    return Dense