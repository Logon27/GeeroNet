import jax.numpy as jnp


# def binary_cross_entropy(predictions, targets):
#     predictions = jnp.clip(predictions, 0.000001, 0.9999999)
#     logits = jnp.log(predictions/(1 - predictions))
#     max_logit = jnp.clip(logits, 0, None)
#     bces = logits - logits * targets + max_logit + jnp.log(jnp.exp(-max_logit) + jnp.exp(-logits - max_logit))
#     return jnp.mean(bces)

# https://sparrow.dev/binary-cross-entropy/
def binary_cross_entropy(y_pred, y_true):
    return jnp.mean(y_true * jnp.log(y_pred) + (1 - y_true) * jnp.log(1 - y_pred))