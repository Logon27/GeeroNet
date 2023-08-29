"""A Convolutional Neural Network (FCNN) example for MNIST"""
import sys
sys.path.append("..")

# Import the TQDM config for cleaner progress bars
import training_examples.tqdm_config # pyright: ignore
from tqdm import trange

import time
import itertools

import numpy.random as npr

import jax.numpy as jnp
from jax import jit, grad, random
import datasets as datasets
import matplotlib.pyplot as plt

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
        Conv(6, (5, 5), padding='SAME'), Elu,
        Conv(16, (3, 3), padding='SAME'), Elu,
        Flatten(),
        Dense(120), Elu,
        Dense(84), Elu,
        Dense(10), LogSoftmax,
    )
)

def main():
    rng = random.PRNGKey(0)

    step_size = 0.001
    num_epochs = 10
    batch_size = 128
    momentum_mass = 0.9
    # IMPORTANT
    # If your network is larger and you test against the entire dataset for the accuracy.
    # Then you will run out or RAM and get a std::bad_alloc error.
    accuracy_batch_size = 1000

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    train_images = jnp.reshape(train_images, (train_images.shape[0], 28, 28, 1))
    test_images = jnp.reshape(test_images, (test_images.shape[0], 28, 28, 1))

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                # batch_idx is a list of indices.
                # That means this function yields an array of training images equal to the batch size when 'next' is called.
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]

    batches = data_stream()

    opt_init, opt_update, get_params = momentum(step_size, mass=momentum_mass)

    @jit
    def update(i, opt_state, batch):
        params = get_params(opt_state)
        return opt_update(i, grad(loss)(params, batch), opt_state)

    _, init_params = net_init(rng, (-1, 28, 28, 1))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    print("Starting training...")
    for epoch in (t := trange(num_epochs)):
        for _ in range(num_batches):
            opt_state = update(next(itercount), opt_state, next(batches))

        params = get_params(opt_state)
        train_acc = accuracy(params, (train_images[:accuracy_batch_size], train_labels[:accuracy_batch_size]))
        test_acc = accuracy(params, (test_images[:accuracy_batch_size], test_labels[:accuracy_batch_size]))
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
            output = net_predict(params, test_images[i].reshape(1, *test_images[i].shape))
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