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

import jax.numpy as jnp
from nn.decorators.fan_in_sum_decorator import debug_decorator as faninsum_debug_decorator
from nn.decorators.fan_out_decorator import debug_decorator as fanout_debug_decorator
from nn.decorators.fan_in_concat_decorator import debug_decorator as faninconcat_decorator


@fanout_debug_decorator
def FanOut(num):
  """Layer construction function for a fan-out layer."""
  init_fun = lambda rng, input_shape: ([input_shape] * num, ())
  apply_fun = lambda params, inputs, **kwargs: [inputs] * num
  return init_fun, apply_fun


def FanInSum():
  """Layer construction function for a fan-in sum layer."""
  init_fun = lambda rng, input_shape: (input_shape[0], ())
  apply_fun = lambda params, inputs, **kwargs: sum(inputs)
  return init_fun, apply_fun
FanInSum = faninsum_debug_decorator(FanInSum)()


@faninconcat_decorator
def FanInConcat(axis=-1):
  """Layer construction function for a fan-in concatenation layer."""
  def init_fun(rng, input_shape):
    ax = axis % len(input_shape[0])
    concat_size = sum(shape[ax] for shape in input_shape)
    out_shape = input_shape[0][:ax] + (concat_size,) + input_shape[0][ax+1:]
    return out_shape, ()
  def apply_fun(params, inputs, **kwargs):
    return jnp.concatenate(inputs, axis)
  return init_fun, apply_fun