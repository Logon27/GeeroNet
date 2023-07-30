from jax import random
from jax.nn.initializers import glorot_normal, normal
import jax.numpy as jnp
import logging
import jax


def Dense(output_shape, weight_init=glorot_normal(), bias_init=normal()):
    """Layer constructor function for a dense (fully-connected) layer.
    
    Args:
        output_shape (int): The 1 dimensional output shape
        weight_init (func): A function used to initialize the weights
        bias_init (func): A function used to initialize the biases

    Example:
        Usage for mnist::

        net_init, net_predict = serial(
            Dense(70),
            Sigmoid,
            Dense(35),
            Sigmoid,
            Dense(10),
            LogSoftmax
        )

    Returns:
        ``(init_fun, update_fun)``: Tuple of functions.
    """

    def init_fun(rng, input_shape):
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
    def apply_fun(params, inputs, **kwargs):
        """
        Args:
            params (jax.Array): An array containing the weight and bias parameters.
            inputs (jax.Array): An array containing the inputs to be applied for the forward pass.
        
        Returns:
            The resulting jax.Array from the Dense forward pass.
        """
        if inputs.ndim <= 1:
            raise ValueError(
                "Input must be at least 2 dimensional. This helps eliminate any confusion with mixing vector and matrix multiplication."
            )

        weights, bias = params
        if logging.getLevelName(logging.root.level) == "INFO2":
            jax.debug.print("I({}, {}) @ W({}, {}) + B({}, {}) = output_shape: {}",
                inputs.shape[0], inputs.shape[1], weights.shape[0], weights.shape[1], bias.shape[0], bias.shape[1], output_shape
            )
        return jnp.matmul(inputs, weights) + bias

    return init_fun, apply_fun
