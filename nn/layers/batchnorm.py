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
        return input_shape, (beta, gamma)
    def apply_fun(params, x, **kwargs):
        beta, gamma = params
        # TODO(phawkins): jnp.expand_dims should accept an axis tuple.
        # (https://github.com/numpy/numpy/issues/12290)
        ed = tuple(None if i in axis else slice(None) for i in range(jnp.ndim(x)))
        z = standardize(x, axis, epsilon=epsilon)
        if center and scale: return gamma[ed] * z + beta[ed]
        if center: return z + beta[ed]
        if scale: return gamma[ed] * z
        return z
    return init_fun, apply_fun