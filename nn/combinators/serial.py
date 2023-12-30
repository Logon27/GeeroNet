from jax import random
from jax.random import PRNGKey
from typing import Any, List, Tuple
from jax import Array
from jax.typing import ArrayLike
from nn.typing import Params
from nn.decorators.serial_decorator import debug_decorator
import jax

@debug_decorator
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

    def init_fun(rng: PRNGKey, input_shape) -> Tuple[Any, List[Params]]:
        """
        Args:
            rng: A PRNGKey used to initialize random values.
            input_shape: The input shape of serial combinator.

        Returns:
            An ``output_shape, params`` tuple. Where output_shape is the output shape of the combined layers.
            And params is a list of tuples of jax arrays representing parameters for each of the layers that were combined.
        """
        params = []
        states = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param, state = init_fun(layer_rng, input_shape)
            params.append(param)
            states.append(state)

        # input_shape at this point represents the final layer's output shape
        output_shape = input_shape
        return output_shape, params, states

    def apply_fun(params: List[Params], states, inputs: ArrayLike, **kwargs) -> Array:
        """
        Args:
            params: The list of parameters for the serial layer.
            inputs: An array containing the inputs to be applied for the forward pass.

        Returns:
            The result of the forward pass for the serial layers.
        """
        rng = kwargs.pop("rng", None)
        rngs = (random.split(rng, num_layers) if rng is not None else (None,) * num_layers)
        for index, (fun, param, state, rng) in enumerate(zip(apply_funs, params, states, rngs)):
            inputs, state = fun(param, state, inputs, rng=rng, **kwargs)
            states[index] = state
        
        return inputs, states

    return init_fun, apply_fun
