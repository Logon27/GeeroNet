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
from nn.decorators.dropout_decorator import debug_decorator


# Using a somewhat modified implementation
# https://github.com/google/jax/issues/16218
@debug_decorator
def Dropout(drop_probability=0.25):
  """Layer construction function for a dropout layer with given rate.
  
  The probability parmeter is the drop probability.
  If you want a success rate of 75% then you would set a probability of 0.25 or a 25% drop rate.
  """
  def init_fun(rng, input_shape):
    return input_shape, ()
  def apply_fun(params, inputs, **kwargs):
    rng = kwargs.get('rng', None)
    if rng is None:
      msg = ("Dropout layer requires apply_fun to be called with a PRNG key "
             "argument. That is, instead of `apply_fun(params, inputs)`, call "
             "it like `apply_fun(params, inputs, rng)` where `rng` is a "
             "jax.random.PRNGKey value.")
      raise ValueError(msg)
    mode = kwargs.get('mode', 'train')
    if mode == 'train':
      # Invert the probability because the implementation operates off the keep rate.
      rate = 1 - drop_probability
      keep = random.bernoulli(rng, rate, inputs.shape)
      return jnp.where(keep, inputs, 0)
    else:
      return inputs
  return init_fun, apply_fun