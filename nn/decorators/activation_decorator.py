from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
import functools
import logging
import jax


def debug_decorator(activation_debug):
    """
    Decorator used to wrap all activation functions.
    """
    @functools.wraps(activation_debug)
    def Activation(*args, **kwargs):
        if logging.getLevelName(logging.root.level) == "INFO2":
            init_fun_debug, apply_fun_debug = activation_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng: PRNGKey, input_shape: ArrayLike):
                output_shape, (), state = init_fun_debug(rng, input_shape)
                jax.debug.print(args[0].__name__.capitalize() + "()")
                return output_shape, (), state
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params: Params, state, inputs: ArrayLike, **kwargs):
                result, state = apply_fun_debug(params, state, inputs, **kwargs)
                jax.debug.print(args[0].__name__.capitalize() + "()")
                return result, state

            return init_fun, apply_fun
        else:
            return activation_debug(*args, **kwargs)
        
    return Activation