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

from jax import lax
import jax.numpy as jnp
from nn.decorators.pooling_decorator import debug_decorator


def _pooling_layer(reducer, init_val, rescaler=None):
    def PoolingLayer(window_shape, strides=None, padding='VALID', spec=None):
        """Layer construction function for a pooling layer."""
        strides = strides or (1,) * len(window_shape)
        rescale = rescaler(window_shape, strides, padding) if rescaler else None

        if spec is None:
            non_spatial_axes = 0, len(window_shape) + 1
        else:
            non_spatial_axes = spec.index('N'), spec.index('C')

        for i in sorted(non_spatial_axes):
            window_shape = window_shape[:i] + (1,) + window_shape[i:]
            strides = strides[:i] + (1,) + strides[i:]

        def init_fun(rng, input_shape):
            padding_vals = lax.padtype_to_pads(input_shape, window_shape, strides, padding)
            ones = (1,) * len(window_shape)
            out_shape = lax.reduce_window_shape_tuple(input_shape, window_shape, strides, padding_vals, ones, ones)
            # Converting the 0 in the output shape to -1 for consistency
            if out_shape[0] == 0:
                out_shape = (-1, *out_shape[1:])
            return out_shape, (), None
        def apply_fun(params, state, inputs, **kwargs):
            out = lax.reduce_window(inputs, init_val, reducer, window_shape, strides, padding)
            return (rescale(out, inputs, spec), state) if rescale else (out, state)
        return init_fun, apply_fun
    return PoolingLayer

def _normalize_by_window_size(dims, strides, padding):
    def rescale(outputs, inputs, spec):
        if spec is None:
            non_spatial_axes = 0, inputs.ndim - 1
        else:
            non_spatial_axes = spec.index('N'), spec.index('C')

        spatial_shape = tuple(inputs.shape[i]
                            for i in range(inputs.ndim)
                            if i not in non_spatial_axes)
        one = jnp.ones(spatial_shape, dtype=inputs.dtype)
        window_sizes = lax.reduce_window(one, 0., lax.add, dims, strides, padding)
        for i in sorted(non_spatial_axes):
            window_sizes = jnp.expand_dims(window_sizes, i)

        return outputs / window_sizes
    return rescale

MaxPool = _pooling_layer(lax.max, -jnp.inf)
SumPool = _pooling_layer(lax.add, 0.)
AvgPool = _pooling_layer(lax.add, 0., _normalize_by_window_size)

MaxPool = debug_decorator(MaxPool, "MaxPool")
SumPool = debug_decorator(SumPool, "SumPool")
AvgPool = debug_decorator(AvgPool, "AvgPool")