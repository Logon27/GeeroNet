import jax.numpy as jnp
from nn.decorators.reshape_decorator import debug_decorator


# https://github.com/google/jax/issues/1796
@debug_decorator
def Reshape(new_shape):
    """Layer construction function for a reshape layer."""
    init_fun = lambda rng, input_shape: (new_shape, (), None)
    apply_fun = lambda params, state, inputs, **kwargs: (jnp.reshape(inputs, new_shape), state)
    return init_fun, apply_fun