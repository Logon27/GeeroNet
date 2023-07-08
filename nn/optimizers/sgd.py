from typing import Any, Callable, NamedTuple, Tuple, Union

from collections import namedtuple
import functools
from functools import partial

import jax.numpy as jnp
from jax._src.util import safe_zip, safe_map, unzip2
from jax import tree_util
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           register_pytree_node)

map = safe_map
zip = safe_zip

# The implementation here basically works by flattening pytrees. There are two
# levels of pytrees to think about: the pytree of params, which we can think of
# as defining an "outer pytree", and a pytree produced by applying init_fun to
# each leaf of the params pytree, which we can think of as the "inner pytrees".
# Since pytrees can be flattened, that structure is isomorphic to a list of
# lists (with no further nesting).

OptimizerState = namedtuple("OptimizerState",
                            ["packed_state", "tree_def", "subtree_defs"])
register_pytree_node(
    OptimizerState,
    lambda xs: ((xs.packed_state,), (xs.tree_def, xs.subtree_defs)),
    lambda data, xs: OptimizerState(xs[0], data[0], data[1]))  # type: ignore[index]


Array = Any
Params = Any  # Parameters are arbitrary nests of `jnp.ndarrays`.
State = Any   # internal State
Updates = Params  # Gradient updates are of the same type as parameters.

InitFn = Callable[[Params], OptimizerState]
Step = int
UpdateFn = Callable[[Step, Updates, OptimizerState], OptimizerState]
ParamsFn = Callable[[OptimizerState], Params]

class Optimizer(NamedTuple):
  init_fn: InitFn
  update_fn: UpdateFn
  params_fn: ParamsFn

Schedule = Callable[[Step], float]

def optimizer(opt_maker: Callable[...,
  Tuple[Callable[[Params], State],
        Callable[[Step, Updates, Params], Params],
        Callable[[State], Params]]]) -> Callable[..., Optimizer]:
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
      states_flat, tree, subtrees = opt_state
      grad_flat, tree2 = tree_flatten(grad_tree)
      if tree2 != tree:
        msg = ("optimizer update function was passed a gradient tree that did "
               "not match the parameter tree structure with which it was "
               "initialized: parameter tree {} and grad tree {}.")
        raise TypeError(msg.format(tree, tree2))
      states = map(tree_unflatten, subtrees, states_flat)
      new_states = map(partial(update, i), grad_flat, states)
      new_states_flat, subtrees2 = unzip2(map(tree_flatten, new_states))
      for subtree, subtree2 in zip(subtrees, subtrees2):
        if subtree2 != subtree:
          msg = ("optimizer update function produced an output structure that "
                 "did not match its input structure: input {} and output {}.")
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

def sgd(step_size):
  """Construct optimizer triple for stochastic gradient descent.

  Args:
    step_size: positive scalar, or a callable representing a step size schedule
      that maps the iteration index to a positive scalar.

  Returns:
    An (init_fun, update_fun, get_params) triple.
  """
  step_size = make_schedule(step_size)
  def init(x0):
    return x0
  def update(i, g, x):
    return x - step_size(i) * g
  def get_params(x):
    return x
  return Optimizer(init, update, get_params)