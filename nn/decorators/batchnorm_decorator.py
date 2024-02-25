from typing import Tuple
from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
import functools
import logging
import jax


def debug_decorator(batchnorm_debug):
    """
    Decorator to wrap the Batchnorm layer.
    """
    @functools.wraps(batchnorm_debug)
    def Batchnorm(*args, **kwargs):
        if logging.getLevelName(logging.root.level) == "INFO2":
            init_fun_debug, apply_fun_debug = batchnorm_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng: PRNGKey, input_shape: Tuple):
                output_shape, (beta, gamma), (moving_mean, moving_var) = init_fun_debug(rng, input_shape)
                debug_msg = "Batchnorm(Input Shape: {}, Output Shape: {}) => Beta Shape: ({}), Gamma Shape: ({})".format(input_shape, output_shape, beta.shape[0], gamma.shape[0])
                debug_msg = debug_msg.replace("-1", "*")
                jax.debug.print(debug_msg)
                return input_shape, (beta, gamma), (moving_mean, moving_var)
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params: Params, state, inputs: ArrayLike, **kwargs):
                beta, gamma = params
                result, state = apply_fun_debug(params, state, inputs, **kwargs)
                jax.debug.print("Batchnorm{}, Beta({}), Gamma({}) = Output Shape: {}".format(
                    inputs.shape, beta.shape[0], gamma.shape[0], result.shape
                ))
                return result, state

            return init_fun, apply_fun
        else:
            return batchnorm_debug(*args, **kwargs)
        
    return Batchnorm