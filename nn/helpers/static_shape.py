import jax.tree_util as jtu

# import jax
# from nn.helpers.static_shape import StaticShape
# jax.debug.print("shape={x}", x=StaticShape(result.shape))

# This class is necessary because the shape attribute in jit prints (array(128, dtype=int32), array(1024, dtype=int32)) instead of an actual shape (128, 1024)
@jtu.register_pytree_node_class
class StaticShape:
    def __init__(self, value):
        self.value = value

    def tree_flatten(self):
        return (), self.value

    @classmethod
    def tree_unflatten(cls, aux_data, _):
        self = object.__new__(cls)
        self.value = aux_data
        return self

    def __repr__(self):
        return repr(self.value)

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return (self.value == other.value) if isinstance(other, StaticShape) else False