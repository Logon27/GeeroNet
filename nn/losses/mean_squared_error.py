import jax.numpy as jnp


def mean_squared_error(predictions, targets):
    return jnp.mean(jnp.power(jnp.sum(predictions * targets, axis=1), 2))
