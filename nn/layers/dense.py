from jax import random
from jax.random import PRNGKey
from jax.nn.initializers import glorot_normal, normal
from jax import Array
from jax.typing import ArrayLike
from typing import Tuple
from nn.typing import Params, InitFun, ApplyFun
import jax.numpy as jnp
from nn.decorators.dense_decorator import debug_decorator


@debug_decorator
def Dense(output_shape: int, weight_init=glorot_normal(), bias_init=normal()) -> Tuple[InitFun, ApplyFun]:
    """Layer constructor function for a dense (fully-connected) layer.
    
    Args:
        output_shape: The 1 dimensional output shape.
        weight_init: A function used to initialize the weights
        bias_init: A function used to initialize the biases

    Returns:
        ``(init_fun, update_fun)``: Tuple of functions.
    """

    def init_fun(rng: PRNGKey, input_shape) -> Tuple[int, Params]:
        """
        Args:
            rng: A PRNGKey used to initialize random values. 
            input_shape: The input shape of the Dense layer.
        
        Returns:
            An ``output_shape, (weights, bias)`` tuple. Where output_shape is the calculated output shape of the layer.
            And (weights, bias) is a tuple of the initialized weight and bias arrays.
        """
        k1, k2 = random.split(rng)
        weights, bias = weight_init(k1, (input_shape, output_shape)), bias_init(k2, (1, output_shape))
        return output_shape, (weights, bias)

    # kwargs is necessary due to rng being passed to some apply functions.
    def apply_fun(params: Params, inputs: ArrayLike, **kwargs) -> Array:
        """
        Args:
            params: A tuple containing the weight and bias parameters represented as jax arrays.
            inputs: An array containing the inputs to be applied for the forward pass.
        
        Returns:
            The resulting jax.Array from the Dense forward pass.
        """
        if inputs.ndim <= 1:
            raise ValueError(
                "Input must be at least 2 dimensional. This helps eliminate any confusion with mixing vector and matrix multiplication."
            )

        weights, bias = params
        return jnp.matmul(inputs, weights) + bias

    return init_fun, apply_fun
