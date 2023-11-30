# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from jax import random
import jax.numpy as jnp
from jax.nn.initializers import ones, zeros
from jax.nn import standardize


def BatchNorm(axis=(0, 1, 2), epsilon=1e-5, center=True, scale=True, beta_init=zeros, gamma_init=ones):
    """Layer construction function for a batch normalization layer."""
    _beta_init = lambda rng, shape: beta_init(rng, shape) if center else ()
    _gamma_init = lambda rng, shape: gamma_init(rng, shape) if scale else ()
    axis = (axis,) if jnp.isscalar(axis) else axis
    def init_fun(rng, input_shape):
        shape = tuple(d for i, d in enumerate(input_shape) if i not in axis)
        k1, k2 = random.split(rng)
        beta, gamma = _beta_init(k1, shape), _gamma_init(k2, shape)
        moving_mean, moving_var = jnp.zeros(shape), jnp.zeros(shape)
        return input_shape, (beta, gamma, moving_mean, moving_var)
    def apply_fun(params, x, **kwargs):
        beta, gamma, moving_mean, moving_var = params
        mode = kwargs.get("mode", "train")
        if mode != "train":
            # In prediction mode, use mean and variance obtained by moving average
            # `linen.Module.variables` have a `value` attribute containing the array
            X_hat = (x - moving_mean) / jnp.sqrt(moving_var + epsilon)
        else:
            assert len(x.shape) in (2, 4)
            if len(x.shape) == 2:
                # When using a fully connected layer, calculate the mean and
                # variance on the feature dimension
                mean = x.mean(axis=0)
                var = ((x - mean) ** 2).mean(axis=0)
            else:
                # When using a two-dimensional convolutional layer, calculate the
                # mean and variance on the channel dimension (axis=1). Here we
                # need to maintain the shape of `X`, so that the broadcasting
                # operation can be carried out later
                mean = x.mean(axis=(0, 2, 3), keepdims=True)
                var = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
            # In training mode, the current mean and variance are used
            X_hat = (x - mean) / jnp.sqrt(var + epsilon)
            # Update the mean and variance using moving average
            # moving_mean = momentum * moving_mean.value + (1.0 - momentum) * mean
            # moving_var.value = momentum * moving_var.value + (1.0 - momentum) * var
        Y = gamma * X_hat + beta  # Scale and shift
        return Y
    return init_fun, apply_fun

# def batch_norm(X, deterministic, gamma, beta, moving_mean, moving_var, eps,
#                momentum):
#     # Use `deterministic` to determine whether the current mode is training
#     # mode or prediction mode
#     if deterministic:
#         # In prediction mode, use mean and variance obtained by moving average
#         # `linen.Module.variables` have a `value` attribute containing the array
#         X_hat = (X - moving_mean.value) / jnp.sqrt(moving_var.value + eps)
#     else:
#         assert len(X.shape) in (2, 4)
#         if len(X.shape) == 2:
#             # When using a fully connected layer, calculate the mean and
#             # variance on the feature dimension
#             mean = X.mean(axis=0)
#             var = ((X - mean) ** 2).mean(axis=0)
#         else:
#             # When using a two-dimensional convolutional layer, calculate the
#             # mean and variance on the channel dimension (axis=1). Here we
#             # need to maintain the shape of `X`, so that the broadcasting
#             # operation can be carried out later
#             mean = X.mean(axis=(0, 2, 3), keepdims=True)
#             var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
#         # In training mode, the current mean and variance are used
#         X_hat = (X - mean) / jnp.sqrt(var + eps)
#         # Update the mean and variance using moving average
#         moving_mean.value = momentum * moving_mean.value + (1.0 - momentum) * mean
#         moving_var.value = momentum * moving_var.value + (1.0 - momentum) * var
#     Y = gamma * X_hat + beta  # Scale and shift
#     return Y