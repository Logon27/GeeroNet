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

# Modified the apply_fun reshape and the init_fun output_shape

import functools
import operator as op
import jax.numpy as jnp
from nn.decorators.flatten_decorator import debug_decorator


@debug_decorator
def Flatten():
  """Layer construction function for flattening all but the leading dim."""
  def init_fun(rng, input_shape):
    output_shape = -1, functools.reduce(op.mul, input_shape[1:], 1)
    return output_shape, ()
  def apply_fun(params, inputs, **kwargs):
    return jnp.reshape(inputs, (inputs.shape[0], -1))
  return init_fun, apply_fun