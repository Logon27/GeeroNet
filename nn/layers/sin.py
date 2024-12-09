from jax import random
from jax.random import PRNGKey
from jax.nn.initializers import glorot_normal, normal
from jax import Array
from jax.typing import ArrayLike
from typing import Tuple
from nn.typing import Params, InitFun, ApplyFun
import jax.numpy as jnp
import jax


def Sin(out_dim: int, weight_init=glorot_normal(), bias_init=normal()) -> Tuple[InitFun, ApplyFun]:
    """Layer constructor function for a dense (fully-connected) layer.
    
    Args:
        out_dim: The 1 dimensional output shape.
        weight_init: A function used to initialize the weights
        bias_init: A function used to initialize the biases

    Returns:
        ``(init_fun, update_fun)``: Tuple of functions.
    """

    def init_fun(rng: PRNGKey, input_shape: Tuple) -> Tuple[int, Params]:
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
        if not isinstance(input_shape, tuple) or len(input_shape) > 2:
            msg = ("input_shape must be a tuple of length 1 or 2. Examples:.\n"
                "    input_shape = (input_size,)\n"
                "    input_shape = (-1, input_size)\n"
                "    -1 means a wildcard for the batch size.\n")
            raise ValueError(msg)
        
        k1, k2 = random.split(rng)
        output_shape = -1, out_dim
        weights1, weights2, weights3, bias1 = weight_init(k1, (input_shape[-1], out_dim)), weight_init(k1, (out_dim, input_shape[-1])), weight_init(k1, (input_shape[-1], out_dim)), bias_init(k2, (1, out_dim))
        return output_shape, (weights1, weights2, weights3, bias1), ()

    # kwargs is necessary due to rng being passed to some apply functions.
    def apply_fun(params: Params, state, inputs: ArrayLike, **kwargs) -> Array:
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

        weights1, weights2, weights3, bias1 = params

        # print(jnp.sin(jnp.matmul(inputs, weights2)).shape)
        # print(weights1.shape)
        # print(jnp.matmul(jnp.sin(jnp.matmul(inputs, weights2)), weights1).shape)
        # print("---")
        # print(inputs.shape)
        # print(jnp.matmul(jnp.sin(jnp.matmul(inputs, weights2)), weights1).shape)
        # print("---")
        # print((inputs - jnp.matmul(jnp.sin(jnp.matmul(inputs, weights2)), weights1)).shape)
        # print(weights3.shape)
        # exit()
        # w_{3}\left(x-w_{2}\sin\left(x\cdot w_{1}\right)\right)+b_{1}
        result = jnp.matmul((inputs - jnp.matmul(jnp.sin(jnp.matmul(inputs, weights1)), weights2)), weights3) + bias1, state
        return result
    
        # Mimic cycloid
        # result = jnp.matmul(1-jnp.cos(jnp.matmul(inputs, weights2)), weights1) + bias1, state

        # regular sin
        # weights1, weights2, weights3, bias1, bias2, bias3, bias4, bias5 = params
        # return jnp.sin(jnp.matmul(inputs, weights1) + bias1) + jnp.sin(jnp.matmul(inputs, weights2) + bias2) + jnp.sin(jnp.matmul(inputs, weights3) + bias3) + bias4, state
    
        # return jnp.sin(jnp.matmul(inputs, weights1)) + jnp.sin(jnp.matmul(inputs, weights2)) + jnp.sin(jnp.matmul(inputs, weights3)) + bias1, state
        #return ((jnp.matmul(jnp.sin(inputs), weights1) + bias1) + (jnp.matmul(jnp.sin(inputs), weights2) + bias2) + (jnp.matmul(jnp.sin(inputs), weights3) + bias3)) / 3, state
        #return (jnp.matmul(inputs, jnp.sin(weights1)) + bias1) + jnp.sin(jnp.matmul(inputs, weights2) + bias2), state
        #return jnp.sin(jnp.matmul(inputs, weights1) + bias1) + jnp.sin(jnp.matmul(inputs, weights2) + bias2), state
        # 10/10 Epochs | 00:14 Elapsed | Accuracy Train = 98.81%, Accuracy Test = 97.60%

    return init_fun, apply_fun
