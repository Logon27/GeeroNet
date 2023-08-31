"""A basic MNIST example with data augmentation

Performance is worse due to the data augementation."""
import sys
sys.path.append("..")

# Import the TQDM config for cleaner progress bars
import training_examples.tqdm_config # pyright: ignore
from tqdm import trange

import itertools

import numpy.random as npr

import jax.numpy as jnp
from jax import jit, grad, random
import datasets as datasets
import matplotlib.pyplot as plt
import jax
from functools import partial

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

net_init, net_predict = model_decorator(
    serial(
        Dense(1024),
        Relu,
        Dense(1024),
        Relu,
        Dense(10),
        LogSoftmax
    )
)

def main():
    rng = random.PRNGKey(85)

    step_size = 0.001
    num_epochs = 10
    batch_size = 128
    momentum_mass = 0.9

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def augment(rng, batch):
        # Generate the same number of keys as the array size. In this case, 5.
        subkeys = random.split(rng, batch.shape[0])
        # Calculate random degrees between -20 and 20
        random_angles = jax.vmap(lambda x: jax.random.uniform(x, minval=-20, maxval=20), in_axes=(0), out_axes=0)(subkeys)
        random_vertical_shifts = jax.vmap(lambda x: jax.random.uniform(x, minval=-3, maxval=3), in_axes=(0), out_axes=0)(subkeys)
        random_horizontal_shifts = jax.vmap(lambda x: jax.random.uniform(x, minval=-3, maxval=3), in_axes=(0), out_axes=0)(subkeys)

        # Each batch uses the same fixed zoom value. This is a limitation due to vmap not allowing dynamically shaped arrays.
        random_zoom = jax.random.uniform(subkeys[0], minval=0.75, maxval=1.45)
        # random_zoom = float(random_zoom) # if you want to jit the zoom function
        zoom_grayscale_image_fixed = partial(zoom_grayscale_image, zoom_factor=random_zoom)

        batch = jnp.reshape(batch * 256, (batch.shape[0], 28,28))
        # batch = jax.vmap(jit(zoom_grayscale_image_fixed), in_axes=(0), out_axes=0)(batch) # if you want to jit the zoom function
        batch = jax.vmap(zoom_grayscale_image_fixed, in_axes=(0), out_axes=0)(batch)
        batch = jax.vmap(jit(translate_grayscale_image), in_axes=(0,0,0), out_axes=0)(batch, random_vertical_shifts, random_horizontal_shifts)
        batch = jax.vmap(jit(rotate_grayscale_image), in_axes=(0,0), out_axes=0)(batch, random_angles)
        batch = jax.vmap(jit(noisify_grayscale_image), in_axes=(0,0), out_axes=0)(subkeys, batch)
        batch = jnp.reshape(batch, (batch.shape[0], 28*28))
        return batch

    def data_stream():
        # Need to modify this np_rng to be jax PRNG
        np_rng = npr.RandomState(0)
        key = jax.random.PRNGKey(0)
        while True:
            perm = np_rng.permutation(num_train)
            for i in range(num_batches):
                # batch_idx is a list of indices.
                # That means this function yields an array of training images equal to the batch size when 'next' is called.
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                # Augment the training data using key.
                key, subkey = jax.random.split(key)
                train_images_aug = augment(subkey, train_images[batch_idx])
                yield train_images_aug, train_labels[batch_idx]

    batches = data_stream()

    opt_init, opt_update, get_params = momentum(step_size, mass=momentum_mass)
    # opt_init, opt_update, get_params = sgd(step_size)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    _, init_params = net_init(rng, (-1, 28 * 28))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    print("Starting training...")
    for epoch in (t := trange(num_epochs)):
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))

        params = get_params(opt_state)
        train_acc = accuracy(params, (train_images, train_labels))
        test_acc = accuracy(params, (test_images, test_labels))
        t.set_description_str("Accuracy Train = {:.2%}, Accuracy Test = {:.2%}".format(train_acc, test_acc))
    print("Training Complete.")

    # Visual Debug After Training
    visual_debug(get_params(opt_state), test_images, test_labels)

def visual_debug(params, test_images, test_labels, starting_index=0, rows=5, columns=10):
    fig, axes = plt.subplots(nrows=rows, ncols=columns, sharex=False, sharey=True, figsize=(12, 8))
    fig.canvas.manager.set_window_title('Network Predictions')
    # "i" represents the test set starting index.
    i = starting_index
    for j in range(rows):
        for k in range(columns):
            output = net_predict(params, test_images[i].reshape(1, test_images[i].shape[0]))
            prediction = int(jnp.argmax(output, axis=1)[0])
            target = int(jnp.argmax(test_labels[i], axis=0))
            prediction_color = "green" if prediction == target else "red"
            axes[j][k].set_title(prediction, color=prediction_color)
            axes[j][k].imshow(test_images[i].reshape(28, 28), cmap='gray')
            axes[j][k].get_xaxis().set_visible(False)
            axes[j][k].get_yaxis().set_visible(False)
            i += 1
    plt.show()

if __name__ == "__main__":
    main()