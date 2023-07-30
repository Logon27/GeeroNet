from jax import random
import jax
import logging
import time
import functools
from nn.debug_utils import *


def time_fun(func):
    """
    Decorator to print the execution time of the forward pass for INFO2 loglevel.

    Printing of the execution time will only work when JIT is disabled.
    By default the INFO2 loglevel disables JIT automatically.
    """

    @functools.wraps(func)
    def apply_fun(*args, **kwargs):
        if logging.getLevelName(logging.root.level) == "INFO2":
            # Before Function Execution
            jax.debug.print("\n=== Start Forward Pass Execution ===")
            start_time_forward = time.time()

            result = func(*args, **kwargs)

            # After Function Execution
            end_time_forward = time.time()
            time_elapsed_ms = (end_time_forward - start_time_forward) * 1000
            jax.debug.print("Forward Pass Took: {:.2f} ms", time_elapsed_ms)
            jax.debug.print("=== End Forward Pass Execution ===\n")
            jax.debug.breakpoint()
        else:
            return func(*args, **kwargs)

        # returning the value to the original frame
        return result

    return apply_fun


def serial(*layers):
    """Combinator for composing layers in serial.

    Args:
      *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

    Returns:
      A new layer, meaning an (init_fun, apply_fun) pair, representing the serial
      composition of the given sequence of layers.
    """
    num_layers = len(layers)
    init_funs, apply_funs = zip(*layers)

    def init_fun(rng, input_shape):
        """
        Args:
            rng: A PRNGKey used to initialize random values.
            input_shape: The input shape of serial combinator.

        Returns:
            An ``output_shape, params`` tuple. Where output_shape is the output shape of the combined layers.
            And params is a list of jax arrays representing parameters for each of the layers that were combined.
        """
        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)

        if logging.getLevelName(logging.root.level) == "INFO":
            print_params(params)

        # input_shape at this point represents the final layer's output shape
        output_shape = input_shape
        return output_shape, params

    @time_fun
    def apply_fun(params, inputs, **kwargs):
        """
        Args:
            params (List[jax.Array]): The list of parameters for the serial layer.
            inputs: (jax.Array): An array containing the inputs to be applied for the forward pass.

        Returns:
            The result of the forward pass for the serial layers.
        """
        rng = kwargs.pop("rng", None)
        rngs = (random.split(rng, num_layers) if rng is not None else (None,) * num_layers)
        for fun, param, rng in zip(apply_funs, params, rngs):
            inputs = fun(param, inputs, rng=rng, **kwargs)
        return inputs

    return init_fun, apply_fun
