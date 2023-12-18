"""A Convolutional Neural Network example for MNIST"""
import sys
sys.path.append("..")

# Import the TQDM config for cleaner progress bars
import training_examples.helpers.tqdm_config # pyright: ignore
from tqdm import trange

import itertools
import jax.numpy as jnp
from jax import jit, grad, random
import training_examples.helpers.datasets as datasets
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from nn import *


def accuracy(params, states, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(net_predict(params, states, inputs)[0], axis=1)
    return jnp.mean(predicted_class == target_class)

net_init, net_predict = model_decorator(
    serial(
        Conv(6, (5, 5), padding='SAME'), Elu,
        Conv(16, (3, 3), padding='SAME'), Elu,
        Flatten,
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
    # Then you will run out of RAM and get a std::bad_alloc error.
    accuracy_batch_size = 1000

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    train_images = jnp.reshape(train_images, (train_images.shape[0], 28, 28, 1))
    test_images = jnp.reshape(test_images, (test_images.shape[0], 28, 28, 1))

    def data_stream(rng):
        while True:
            rng, subkey = random.split(rng)
            perm = random.permutation(subkey, num_train)
            for i in range(num_batches):
                # batch_idx is a list of indices.
                # That means this function yields an array of training images equal to the batch size when 'next' is called.
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]

    batches = data_stream(rng)

    opt_init, opt_update, get_params = momentum(step_size, mass=momentum_mass)

    @jit
    def update(i, opt_state, states, batch):
        def loss(params, states, batch):
            """Calculates the loss of the network as a single value / float"""
            inputs, targets = batch
            predictions, states = net_predict(params, states, inputs)
            return categorical_cross_entropy(predictions, targets), states

        params = get_params(opt_state)
        grads, states = grad(loss, has_aux=True)(params, states, batch)
        return opt_update(i, grads, opt_state), states

    _, init_params, states = net_init(rng, (-1, 28, 28, 1))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    print("Starting training...")
    for epoch in (t := trange(num_epochs)):
        for batch in range(num_batches):
            opt_state, states = update(next(itercount), opt_state, states, next(batches))

        params = get_params(opt_state)
        train_acc = accuracy(params, states, (train_images[:accuracy_batch_size], train_labels[:accuracy_batch_size]))
        test_acc = accuracy(params, states, (test_images[:accuracy_batch_size], test_labels[:accuracy_batch_size]))
        t.set_description_str("Accuracy Train = {:.2%}, Accuracy Test = {:.2%}".format(train_acc, test_acc))
    print("Training Complete.")

    # Visual Debug After Training
    visual_debug(get_params(opt_state), states, test_images, test_labels)

def visual_debug(params, states, test_images, test_labels, starting_index=0, rows=5, columns=10):
    """Visually displays a number of images along with the network prediction. Green means a correct guess. Red means an incorrect guess"""
    print("Displaying Visual Debug...")
    fig, axes = plt.subplots(nrows=rows, ncols=columns, sharex=False, sharey=True, figsize=(12, 8))
    # Set a bottom margin to space out the buttons from the figures
    fig.subplots_adjust(bottom=0.15)
    fig.canvas.manager.set_window_title('Network Predictions')
    class Index:
        def __init__(self, starting_index):
            self.starting_index = starting_index
        
        def render_images(self):
            i = self.starting_index
            for j in range(rows):
                for k in range(columns):
                    output = net_predict(params, states, test_images[i].reshape(1, *test_images[i].shape))[0]
                    prediction = int(jnp.argmax(output, axis=1)[0])
                    target = int(jnp.argmax(test_labels[i], axis=0))
                    prediction_color = "green" if prediction == target else "red"
                    axes[j][k].set_title(prediction, color=prediction_color)
                    axes[j][k].imshow(test_images[i].reshape(28, 28), cmap='gray')
                    axes[j][k].get_xaxis().set_visible(False)
                    axes[j][k].get_yaxis().set_visible(False)
                    i += 1
            plt.draw()
            fig.suptitle("Displaying Images: {} - {}".format(self.starting_index, (self.starting_index + (rows * columns))), fontsize=14)
        
        def next(self, event):
            self.starting_index += (rows * columns)
            self.render_images()
        
        def prev(self, event):
            self.starting_index -= (rows * columns)
            self.render_images()

    callback = Index(starting_index)
    axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
    axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
    bnext = Button(axnext, 'Next', hovercolor="green")
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous', hovercolor="green")
    bprev.on_clicked(callback.prev)
    # Run an initial render before buttons are pressed
    callback.render_images()
    plt.show()

if __name__ == "__main__":
    main()