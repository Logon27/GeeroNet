from jax import random
from jax.random import PRNGKey
from typing import Any, List, Tuple
from jax import Array
from jax.typing import ArrayLike
from nn.typing import Params
from nn.decorators.serial_decorator import debug_decorator


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
        non_trainable_params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            input_shape, param, non_trainable_param = init_fun(layer_rng, input_shape)
            params.append(param)
            non_trainable_params.append(non_trainable_param)

        # input_shape at this point represents the final layer's output shape
        output_shape = input_shape
        return output_shape, params, non_trainable_params

    def apply_fun(params: List[Params], non_trainable_params, inputs: ArrayLike, **kwargs) -> Array:
        """
        Args:
            params: The list of parameters for the serial layer.
            inputs: An array containing the inputs to be applied for the forward pass.

        Returns:
            The result of the forward pass for the serial layers.
        """
        rng = kwargs.pop("rng", None)
        rngs = (random.split(rng, num_layers) if rng is not None else (None,) * num_layers)
        for index, (fun, param, non_trainable_param, rng) in enumerate(zip(apply_funs, params, non_trainable_params, rngs)):
            inputs, non_trainable_param = fun(param, non_trainable_param, inputs, rng=rng, **kwargs)
            non_trainable_params[index] = non_trainable_param
        return inputs, non_trainable_params

    return init_fun, apply_fun
