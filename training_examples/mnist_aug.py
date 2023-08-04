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
import datasets as datasets
import matplotlib.pyplot as plt
import jax

from nn import *

# The loss function produces a single error value representing the effiency of the network. 
# The gradient of the error with respect to the output of the final layer is just...
# the derivative of the loss function applied elementwise to the output (prediction) array of the network.

# The gradient of the error with respect to the input of the last layer is equivalent to...
# the gradient of the error with respect to the output of the second to last layer.
# Which means by using automatic differentiation (the chain rule) to calculate the gradient of the error with respect to the input...
# we can calculate all the gradients of the network by just knowning the error function.
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
    rng = random.PRNGKey(85)

    learning_rate = 0.001  # Learning rate???
    num_epochs = 10
    batch_size = 128 # 128
    momentum_mass = 0.9

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def augment(rng, batch):
        # Generate the same number of keys as the array size. In this case, 5.
        subkeys = random.split(rng, batch.shape[0])
        # Calculate random degrees between 0 and 20
        random_angles = jax.vmap(lambda x: jax.random.uniform(x, minval=-20, maxval=20), in_axes=(0), out_axes=0)(subkeys)
        random_vertical_shifts = jax.vmap(lambda x: jax.random.uniform(x, minval=-3, maxval=3), in_axes=(0), out_axes=0)(subkeys)
        random_horizontal_shifts = jax.vmap(lambda x: jax.random.uniform(x, minval=-3, maxval=3), in_axes=(0), out_axes=0)(subkeys)
        batch = jnp.reshape(batch * 256, (batch.shape[0], 28,28))
        batch = jax.vmap(translate_grayscale_image, in_axes=(0,0,0), out_axes=0)(batch, random_vertical_shifts, random_horizontal_shifts)
        batch = jax.vmap(rotate_grayscale_image, in_axes=(0,0), out_axes=0)(batch, random_angles)
        batch = jax.vmap(noisify_grayscale_image, in_axes=(0,0), out_axes=0)(subkeys, batch)
        save_grayscale_image(batch[0], "test_image.png")
        exit()
        batch = jnp.reshape(batch, (batch.shape[0], 28*28))
        return batch

    def data_stream():
        np_rng = npr.RandomState(0)
        while True:
            perm = np_rng.permutation(num_train)
            for i in range(num_batches):
                # batch_idx is a list of indices.
                # That means this function yields an array of training images equal to the batch size when 'next' is called.
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                train_images_aug = augment(rng, train_images[batch_idx])
                yield train_images_aug, train_labels[batch_idx]

    batches = data_stream()

    opt_init, opt_update, get_params = momentum(learning_rate, mass=momentum_mass)
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

    # Visual Debug After Training
    rows = 5
    columns = 10
    fig, axes = plt.subplots(nrows=rows, ncols=columns, sharex=False, sharey=True, figsize=(12, 8))
    fig.canvas.manager.set_window_title('Network Predictions')
    # "i" represents the test set starting index.
    i = 0
    params = get_params(opt_state)
    for j in range(rows):
        for k in range(columns):
            output = net_predict(params, test_images[i].reshape(1, test_images[i].shape[0]))
            prediction = jnp.argmax(output, axis=1)
            # Convert to a string to prevent an error with cupy
            prediction = str(prediction)
            axes[j][k].set_title(prediction)
            axes[j][k].imshow(test_images[i].reshape(28, 28), cmap='gray')
            axes[j][k].get_xaxis().set_visible(False)
            axes[j][k].get_yaxis().set_visible(False)
            i += 1
    plt.show()