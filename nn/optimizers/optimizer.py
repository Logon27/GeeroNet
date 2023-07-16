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

"""Examples of how to write optimizers with JAX.

You likely do not mean to import this module! The optimizers in this library
are intended as examples only. If you are looking for a fully featured optimizer
library, two good options are JAXopt_ and Optax_.

This module contains some convenient optimizer definitions, specifically
initialization and update functions, which can be used with ndarrays or
arbitrarily-nested tuple/list/dicts of ndarrays.

An optimizer is modeled as an ``(init_fun, update_fun, get_params)`` triple of
functions, where the component functions have these signatures:

::

  init_fun(params)

  Args:
    params: pytree representing the initial parameters.

  Returns:
    A pytree representing the initial optimizer state, which includes the
    initial parameters and may also include auxiliary values like initial
    momentum. The optimizer state pytree structure generally differs from that
    of `params`.

::

  update_fun(step, grads, opt_state)

  Args:
    step: integer representing the step index.
    grads: a pytree with the same structure as `get_params(opt_state)`
      representing the gradients to be used in updating the optimizer state.
    opt_state: a pytree representing the optimizer state to be updated.

  Returns:
    A pytree with the same structure as the `opt_state` argument representing
    the updated optimizer state.

::

  get_params(opt_state)

  Args:
    opt_state: pytree representing an optimizer state.

  Returns:
    A pytree representing the parameters extracted from `opt_state`, such that
    the invariant `params == get_params(init_fun(params))` holds true.


Notice that an optimizer implementation has a lot of flexibility in the form of
opt_state: it just has to be a pytree of JaxTypes (so that it can be passed to
the JAX transforms defined in api.py) and it has to be consumable by update_fun
and get_params.

Example Usage:

.. code-block:: python

  opt_init, opt_update, get_params = optimizers.sgd(learning_rate)
  opt_state = opt_init(params)

  def step(step, opt_state):
    value, grads = jax.value_and_grad(loss_fn)(get_params(opt_state))
    opt_state = opt_update(step, grads, opt_state)
    return value, opt_state

  for i in range(num_steps):
    value, opt_state = step(i, opt_state)


.. _JAXopt: https://github.com/google/jaxopt
.. _Optax: https://github.com/deepmind/optax
"""

from typing import Any, Callable, NamedTuple, Tuple, Union

from collections import namedtuple
import functools
from functools import partial

import jax.numpy as jnp
from jax._src.util import safe_zip, safe_map, unzip2
from jax import tree_util
from jax.tree_util import tree_map, tree_flatten, tree_unflatten, register_pytree_node


map = safe_map
zip = safe_zip

# The implementation here basically works by flattening pytrees. There are two
# levels of pytrees to think about: the pytree of params, which we can think of
# as defining an "outer pytree", and a pytree produced by applying init_fun to
# each leaf of the params pytree, which we can think of as the "inner pytrees".
# Since pytrees can be flattened, that structure is isomorphic to a list of
# lists (with no further nesting).

OptimizerState = namedtuple(
    "OptimizerState", ["packed_state", "tree_def", "subtree_defs"]
)
register_pytree_node(
    OptimizerState,
    lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs)),
    lambda data, xs: OptimizerState(xs[0], data[0], data[1]),
)  # type: ignore[index]


Array = Any
Params = Any  # Parameters are arbitrary nests of `jnp.ndarrays`.
State = Any  # internal State
Updates = Params  # Gradient updates are of the same type as parameters.

InitFn = Callable[[Params], OptimizerState]
Step = int
UpdateFn = Callable[[Step, Updates, OptimizerState], OptimizerState]
ParamsFn = Callable[[OptimizerState], Params]


# The Optimizer class is never used inside the jitted update function.
# Therefore it does not need to be a pytree. The Optimizer named tuple is unpacked before being used in the update function.
class Optimizer(NamedTuple):
    init_fn: InitFn
    update_fn: UpdateFn
    params_fn: ParamsFn

Schedule = Callable[[Step], float]

def optimizer(
    opt_maker: Callable[
        ...,  # A function that takes in Any parameter.
        # Which returns a Tuple with the following 3 values. Which stands for... (init_fun, update_fun, get_params)
        Tuple[
            Callable[[Params], State],
            Callable[[Step, Updates, Params], Params],
            Callable[[State], Params],
        ],
    ]
) -> Callable[..., Optimizer]:
    """Decorator to make an optimizer defined for arrays generalize to containers.

    With this decorator, you can write init, update, and get_params functions that
    each operate only on single arrays, and convert them to corresponding
    functions that operate on pytrees of parameters. See the optimizers defined in
    optimizers.py for examples.

    Args:
      opt_maker: a function that returns an ``(init_fun, update_fun, get_params)``
        triple of functions that might only work with ndarrays, as per

        .. code-block:: haskell

            init_fun :: ndarray -> OptStatePytree ndarray
            update_fun :: OptStatePytree ndarray -> OptStatePytree ndarray
            get_params :: OptStatePytree ndarray -> ndarray

    Returns:
      An ``(init_fun, update_fun, get_params)`` triple of functions that work on
      arbitrary pytrees, as per

      .. code-block:: haskell

            init_fun :: ParameterPytree ndarray -> OptimizerState
            update_fun :: OptimizerState -> OptimizerState
            get_params :: OptimizerState -> ParameterPytree ndarray

      The OptimizerState pytree type used by the returned functions is isomorphic
      to ``ParameterPytree (OptStatePytree ndarray)``, but may store the state
      instead as e.g. a partially-flattened data structure for performance.
    """

    # The wrappers are necessary because we JIT the update function.
    # And jax has to work with pytrees for transformation functions like jit.
    @functools.wraps(opt_maker)
    def tree_opt_maker(*args, **kwargs):
        init, update, get_params = opt_maker(*args, **kwargs)

        @functools.wraps(init)
        def tree_init(x0_tree):
            x0_flat, tree = tree_flatten(x0_tree)
            initial_states = [init(x0) for x0 in x0_flat]
            states_flat, subtrees = unzip2(map(tree_flatten, initial_states))
            return OptimizerState(states_flat, tree, subtrees)

        @functools.wraps(update)
        def tree_update(i, grad_tree, opt_state):
            # opt_init returns a pytree of the OptimizerState object. So you must be able to just parse out the three list elements seen here...
            # E.g. OptimizerState = namedtuple("OptimizerState", ["packed_state", "tree_def", "subtree_defs"])
            states_flat, tree, subtrees = opt_state
            # the grad_tree is flattened to ensure it has the same definition as the opt_state pytree.
            grad_flat, tree2 = tree_flatten(grad_tree)
            if tree2 != tree:
                msg = (
                    "optimizer update function was passed a gradient tree that did "
                    "not match the parameter tree structure with which it was "
                    "initialized: parameter tree {} and grad tree {}."
                )
                raise TypeError(msg.format(tree, tree2))
            states = map(tree_unflatten, subtrees, states_flat)
            # The second and third parameters to the update function are a list of leafs.
            # And each leaf needs to be mapped to the partial function of update with 'i' already applied.
            # The map is necessary because you cannot multiply or subtract by a list inside the update function. You have to map the individual elements.
            # swapped map to tree_map
            new_states = tree_map(partial(update, i), grad_flat, states)
            new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
            for subtree, subtree2 in zip(subtrees, subtrees2):
                if subtree2 != subtree:
                    msg = (
                        "optimizer update function produced an output structure that "
                        "did not match its input structure: input {} and output {}."
                    )
                    raise TypeError(msg.format(subtree, subtree2))
            return OptimizerState(new_states_flat, tree, subtrees)

        @functools.wraps(get_params)
        def tree_get_params(opt_state):
            states_flat, tree, subtrees = opt_state
            states = map(tree_unflatten, subtrees, states_flat)
            params = map(get_params, states)
            return tree_unflatten(tree, params)

        return Optimizer(tree_init, tree_update, tree_get_params)

    return tree_opt_maker


def constant(step_size) -> Schedule:
    def schedule(i):
        return step_size
    return schedule


def make_schedule(scalar_or_schedule: Union[float, Schedule]) -> Schedule:
    if callable(scalar_or_schedule):
        return scalar_or_schedule
    elif jnp.ndim(scalar_or_schedule) == 0:
        return constant(scalar_or_schedule)
    else:
        raise TypeError(type(scalar_or_schedule))
