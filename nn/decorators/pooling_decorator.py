import functools
import logging
import jax


def debug_decorator(pooling_debug, func_name):
    """
    Decorator to wrap the Convolutional layer.
    """
    @functools.wraps(pooling_debug)
    def Pooling(*args, **kwargs):
        if logging.getLevelName(logging.root.level) == "INFO2":
            init_fun_debug, apply_fun_debug = pooling_debug(*args, **kwargs)

            @functools.wraps(init_fun_debug)
            def init_fun(rng, input_shape):
                output_shape, () = init_fun_debug(rng, input_shape)
                debug_msg = "{}(Input Shape: {}, Output Shape: {})".format(func_name, input_shape, output_shape)
                debug_msg = debug_msg.replace("-1", "*")
                jax.debug.print(debug_msg)
                return output_shape, ()
            
            @functools.wraps(apply_fun_debug)
            def apply_fun(params, inputs, **kwargs):
                result = apply_fun_debug(params, inputs, **kwargs)
                jax.debug.print("{}{} = Output Shape: {}".format(
                    func_name, inputs.shape, result.shape
                ))
                return result

            return init_fun, apply_fun
        else:
            return pooling_debug(*args, **kwargs)
        
    return Pooling