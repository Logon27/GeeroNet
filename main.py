from jax import random
import jax.numpy as jnp
from jax.example_libraries.stax import (Sigmoid)
from dense import Dense
from serial import serial

# Initialize parameters, not committing to a batch shape
rng = random.PRNGKey(111)
init_fun, apply_fun = Dense(4)
params = init_fun(rng, 3)
# print(params[0].shape)
arr = jnp.array([1,2,3])
arr = jnp.reshape(arr, (3, 1))
# print(arr.shape)
# print(apply_fun(params, arr))
net_init, net_apply = serial(
    Dense(4),
    Sigmoid,
    Dense(6),
    Sigmoid,
)
net_params = net_init(rng, arr.shape[0])
# print(net_params)
print(net_apply(net_params, arr))