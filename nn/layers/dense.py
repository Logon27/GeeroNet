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
def Dense(out_dim: int, weight_init=glorot_normal(), bias_init=normal()) -> Tuple[InitFun, ApplyFun]:
    """Layer constructor function for a dense (fully-connected) layer.
    
    Args:
        out_dim: The 1 dimensional output shape.
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
        # The shape of the weight and bias arrays of the Dense layer are in no way dependent on the batch size.
        # However, convolutional layers appear to be dependent on batch size so I am strictly enforcing this format.
        # So you might not have any issues with using a model with only Dense layers, but if you mixed Dense and Conv layers you would have a problem.
        if isinstance(input_shape, int) or len(input_shape) <= 1:
            raise ValueError(
                "Input shape must be a tuple of length 2. Where input_shape = (batch_size, input_size)."
            )
        
        k1, k2 = random.split(rng)
        # Need to investigate why the output_shape needs to be this format.
        output_shape = input_shape[:-1] + (out_dim,)
        weights, bias = weight_init(k1, (input_shape[-1], out_dim)), bias_init(k2, (1, out_dim))
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
                "Input must be 2 dimensional. Where inputs.shape = (batch_size, input_size). This helps eliminate any confusion with mixing vector and matrix multiplication."
            )

        weights, bias = params
        return jnp.matmul(inputs, weights) + bias

    return init_fun, apply_fun
