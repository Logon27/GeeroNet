from jax.typing import ArrayLike
from jax.random import PRNGKey
from nn.typing import Params
import functools
import logging
import jax


def debug_decorator(dropout_debug):
    """
    Decorator to wrap the Identity layer.
    """
    @functools.wraps(dropout_debug)
    def Dropout(*args, **kwargs):
        if logging.getLevelName(logging.root.level) == "INFO2":
            init_fun_debug, apply_fun_debug = dropout_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng: PRNGKey, input_shape: ArrayLike):
                output_shape, () = init_fun_debug(rng, input_shape)
                jax.debug.print("Dropout(Drop Rate: {:.2%})".format(args[0]))
                return output_shape, ()
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params: Params, inputs: ArrayLike, **kwargs):
                result = apply_fun_debug(params, inputs, **kwargs)
                jax.debug.print("Dropout(Drop Rate: {:.2%})".format(args[0]))
                return result

            return init_fun, apply_fun
        else:
            return dropout_debug(*args, **kwargs)
        
    return Dropout