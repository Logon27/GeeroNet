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

# When running against the fashion set some shape values return 0 for batch size. I believe this is due to max pooling.
def debug_decorator(convolution_debug: Callable[P, R]) -> Callable[P, R]:
    """
    Decorator to wrap the Convolutional layer.
    """
    @functools.wraps(convolution_debug)
    def GeneralConv(*args: P.args, **kwargs: P.kwargs) -> R:
        if logging.getLevelName(logging.root.level) == "INFO2":
            init_fun_debug, apply_fun_debug = convolution_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng, input_shape):
                output_shape, (weights, bias) = init_fun_debug(rng, input_shape)
                debug_msg = "Conv(Input Shape: {}, Output Shape: {}) => Weight Shape: {}, Bias Shape: {}".format(input_shape, output_shape, weights.shape, bias.shape)
                debug_msg = debug_msg.replace("-1", "*")
                jax.debug.print(debug_msg)
                return output_shape, (weights, bias)
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params, inputs, **kwargs):
                weights, bias = params
                result = apply_fun_debug(params, inputs, **kwargs)
                jax.debug.print("I{} * W{} + B{} = Output Shape: {}".format(inputs.shape, weights.shape, bias.shape, result.shape))
                return result

            return init_fun, apply_fun
        else:
            return convolution_debug(*args, **kwargs)
        
    return GeneralConv