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

# Copyright 2023 Logon27
# Separated activations into their own file.
# To see the activation function documentation and implementations please reference the link below.
# https://jax.readthedocs.io/en/latest/jax.nn.html#activation-functions

from jax.nn import (relu, log_softmax, softmax, softplus, sigmoid, elu,
                    leaky_relu, selu, gelu)
import jax.numpy as jnp

def elementwise(fun, **fun_kwargs):
  """Layer that applies a scalar function elementwise on its inputs."""
  init_fun = lambda rng, input_shape: (input_shape, ())
  apply_fun = lambda params, inputs, **kwargs: fun(inputs, **fun_kwargs)
  return init_fun, apply_fun

Tanh = elementwise(jnp.tanh)
Relu = elementwise(relu)
Exp = elementwise(jnp.exp)
LogSoftmax = elementwise(log_softmax, axis=-1)
Softmax = elementwise(softmax, axis=-1)
Softplus = elementwise(softplus)
Sigmoid = elementwise(sigmoid)
Elu = elementwise(elu)
LeakyRelu = elementwise(leaky_relu)
Selu = elementwise(selu)
Gelu = elementwise(gelu)