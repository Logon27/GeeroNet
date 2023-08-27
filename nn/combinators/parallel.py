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
from nn.decorators.parallel_decorator import debug_decorator


@debug_decorator
def parallel(*layers):
    """Combinator for composing layers in parallel.

    The layer resulting from this combinator is often used with the FanOut and
    FanInSum layers.

    Args:
      *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

    Returns:
      A new layer, meaning an (init_fun, apply_fun) pair, representing the
      parallel composition of the given sequence of layers. In particular, the
      returned layer takes a sequence of inputs and returns a sequence of outputs
      with the same length as the argument `layers`.
    """
    nlayers = len(layers)
    init_funs, apply_funs = zip(*layers)

    def init_fun(rng, input_shape):
        rngs = random.split(rng, nlayers)
        return zip(
            *[
                init(rng, shape)
                for init, rng, shape in zip(init_funs, rngs, input_shape)
            ]
        )

    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.pop("rng", None)
        rngs = random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        return [
            f(p, x, rng=r, **kwargs)
            for f, p, x, r in zip(apply_funs, params, inputs, rngs)
        ]

    return init_fun, apply_fun
