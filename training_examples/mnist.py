"""A basic MNIST example using JAX with the mini-libraries stax and optimizers.

The mini-library jax.example_libraries.stax is for neural network building, and
the mini-library jax.example_libraries.optimizers is for first-order stochastic
optimization.
"""
import sys

sys.path.append("..")

import time
import itertools

import numpy.random as npr

import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries.stax import Relu, LogSoftmax, Sigmoid
import datasets as datasets

from nn import *


def loss(params, batch):
    inputs, targets = batch
    predictions = net_predict(params, inputs)
    return categorical_cross_entropy(predictions, targets)

def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(net_predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)

net_init, net_predict = serial(
    Dense(1024),
    Relu,
    Dense(1024),
    Relu,
    Dense(10),
    LogSoftmax
)

# net_init, net_predict = serial(
#     Dense(70),
#     Sigmoid,
#     Dense(35),
#     Sigmoid,
#     Dense(10),
#     LogSoftmax
# )

if __name__ == "__main__":
    rng = random.PRNGKey(0)

    step_size = 0.001  # Learning rate???
    num_epochs = 10
    batch_size = 128 # 128
    momentum_mass = 0.9

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]

    batches = data_stream()

    opt_init, opt_update, get_params = momentum(step_size, mass=momentum_mass)
    # opt_init, opt_update, get_params = sgd(step_size)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    _, init_params = net_init(rng, 28 * 28)
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    print("\nStarting training...")
    training_start_time = time.time()
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))
        epoch_time = time.time() - start_time

        params = get_params(opt_state)
        train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        print(
            "{:>{}}/{}, Accuracy Train = {:.2%}, Accuracy Test = {:.2%}, in {:.2f} seconds".format(
                (epoch + 1), len(str(num_epochs)), num_epochs, train_acc, test_acc, epoch_time
            )
        )

    training_end_time = time.time()
    time_elapsed_mins = (training_end_time - training_start_time) / 60
    print(
        "Training Complete. Elapsed Time = {:.2f} seconds. Or {:.2f} minutes.".format(
            training_end_time - training_start_time, time_elapsed_mins
        )
    )
