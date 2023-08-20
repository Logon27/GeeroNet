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


import functools
from jax import lax
from jax import random
from jax.nn.initializers import glorot_normal, normal
from nn.decorators.convolution_decorator import convolution_decorator

# N - batch dimension
# H - spatial height
# W - spatial width
# C - channel dimension
# I - kernel input channel dimension
# O - kernel output channel dimension

# The first string "NHWC" indicates that your inputs are batch, height, width, channel. 
# The second string "HWIO" indicates that your kernel weights are height, width, kernel input, kernel output
# The last string "NHWC" indicates that the outputs of the convolution are batch, height, width, channel.

# N "images" of C channels of H x W feature maps

# Filters are always one dimension more than the kernels.
# For example, in 2D convolutions, filters are 3D matrices (which is essentially a
# concatenation of 2D matrices i.e. the kernels). So for a CNN layer with kernel dimensions
# h*w and input channels k, the filter dimensions are k*h*w.

# https://jax.readthedocs.io/en/latest/notebooks/convolutions.html

def GeneralConv(dimension_numbers, num_filters, kernel_shape, strides=None, padding='VALID', weight_init=None, bias_init=normal(1e-6)):
  """Layer construction function for a general convolution layer."""
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  one = (1,) * len(kernel_shape)
  strides = strides or one
  weight_init = weight_init or glorot_normal(rhs_spec.index('I'), rhs_spec.index('O'))

  # The init_fun is not dependent on the batch size, input height, or input width. Only the number of input channels.
  # However, when mixing layers like Dense -> Reshape -> Conv. Then the input height and width must be known.
  # Because the Reshape operation cannot have a wildcard in 3 dimensions, only 1.
  def init_fun(rng, input_shape):
    kernel_shape_iter = iter(kernel_shape)
    filter_shape = [num_filters if c == 'O' else
                    input_shape[lhs_spec.index('C')] if c == 'I' else
                    next(kernel_shape_iter) for c in rhs_spec]
    output_shape = lax.conv_general_shape_tuple(input_shape, filter_shape, strides, padding, dimension_numbers)
    bias_shape = [num_filters if c == 'C' else 1 for c in out_spec]
    k1, k2 = random.split(rng)
    weights, bias = weight_init(k1, filter_shape), bias_init(k2, bias_shape)
    return output_shape, (weights, bias)
  
  def apply_fun(params, inputs, **kwargs):
    weights, bias = params
    return lax.conv_general_dilated(inputs, weights, strides, padding, one, one, dimension_numbers=dimension_numbers) + bias
  
  return init_fun, apply_fun

# input_shape = (-1, input_height, input_width, input_channels)
Conv = functools.partial(GeneralConv, ('NHWC', 'HWIO', 'NHWC'))
# Cannot use the decorator annotation on the main function because of partial function definition.
Conv = convolution_decorator(Conv)