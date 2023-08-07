import jax.numpy as jnp


# https://github.com/google/jax/issues/1796
def Reshape(new_shape):
  """Layer construction function for a reshape layer."""
  init_fun = lambda rng, input_shape: (new_shape, ())
  apply_fun = lambda params, inputs, **kwargs: jnp.reshape(inputs, new_shape)
  return init_fun, apply_fun