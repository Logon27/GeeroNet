import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array


# https://sparrow.dev/binary-cross-entropy/
def binary_cross_entropy(predictions: ArrayLike, targets: ArrayLike) -> Array:
    return jnp.mean(targets * jnp.log(predictions) + (1 - targets) * jnp.log(1 - predictions))