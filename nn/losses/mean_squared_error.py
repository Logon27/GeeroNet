import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array

def mean_squared_error(predictions: ArrayLike, targets: ArrayLike) -> Array:
    return jnp.mean(jnp.square(targets - predictions))
