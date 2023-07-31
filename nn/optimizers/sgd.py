from .optimizer import *


@optimizer
def sgd(step_size):
    """Construct optimizer triple for stochastic gradient descent.

    Args:
      step_size: positive scalar, or a callable representing a step size schedule
        that maps the iteration index to a positive scalar.

    Returns:
      An (init_fun, update_fun, get_params) triple of functions.
    """
    step_size = make_schedule(step_size)

    def init(x0):
        return x0

    def update(i, g, x):
        return x - step_size(i) * g

    def get_params(x):
        return x

    return init, update, get_params
