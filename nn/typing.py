from jax import Array
from jax.typing import ArrayLike
from typing import Any, Tuple, Callable
from jax.random import PRNGKey

Params = Tuple[Array, Array]
InitFun = Callable[[PRNGKey, ArrayLike], Tuple[Any, Params]]
ApplyFun = Callable[[Params, ArrayLike], Array]