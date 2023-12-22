from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
import functools
import logging
import jax
import jax.numpy as jnp


def debug_decorator(faninsum_debug):
    """
    Decorator to wrap the FanInSum layer.
    """
    @functools.wraps(faninsum_debug)
    def FanInSum(*args, **kwargs):
        if logging.getLevelName(logging.root.level) == "INFO2":
            init_fun_debug, apply_fun_debug = faninsum_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng: PRNGKey, input_shape: ArrayLike):
                output_shape, (), state = init_fun_debug(rng, input_shape)
                debug_msg = "FanInSum(Input Shape: {}, Output Shape: {})".format(input_shape, output_shape)
                debug_msg = debug_msg.replace("-1", "*")
                jax.debug.print(debug_msg)
                return output_shape, (), state
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params: Params, state, inputs: ArrayLike, **kwargs):
                result, state = apply_fun_debug(params, state, inputs, **kwargs)
                
                jax.debug.print("FanInSum({}) = Output Shape: {}".format(
                    jnp.array(inputs).shape, result.shape
                ))
                return result, state

            return init_fun, apply_fun
        else:
            return faninsum_debug(*args, **kwargs)
        
    return FanInSum