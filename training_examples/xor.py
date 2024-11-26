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
import matplotlib.pyplot as plt


def accuracy(params, states, batch):
    """Calculates accuracy (or number of correct guesses) for a given batch"""
    inputs, targets = batch
    predicted_class = jnp.round(net_predict(params, states, inputs)[0])
    return jnp.mean(predicted_class == targets)

net_init, net_predict = model_decorator(
    serial(
        Sin(100),
        # Relu,
        Sin(1),
        # Relu
    )
)

def main():
    rng = random.PRNGKey(0)

    step_size = 0.03
    num_epochs = 1000
    batch_size = 1
    momentum_mass = 0.9

    X = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = jnp.array([[0], [1], [1], [0]])
    X = jnp.reshape(X, (4, 2))
    Y = jnp.reshape(Y, (4, 1))
    num_train = X.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    opt_init, opt_update, get_params = sgd(step_size)

    @jit
    def update(i, opt_state, states, batch):
        # The loss function produces a single error value representing the effiency of the network. 
        # The gradient of the error with respect to the output of the final layer is just...
        # the derivative of the loss function applied elementwise to the output (prediction) array of the network.

        # The gradient of the error with respect to the input of the last layer is equivalent to...
        # the gradient of the error with respect to the output of the second to last layer.
        # Which means by using automatic differentiation (the chain rule) to calculate the gradient of the error with respect to the input...
        # we can calculate all the gradients of the network by just knowing the error function.
        def loss(params, states, batch):
            """Calculates the loss of the network as a single value / float"""
            inputs, targets = batch
            predictions, states = net_predict(params, states, inputs)
            return mean_squared_error(predictions, targets), states

        params = get_params(opt_state)
        grads, states = grad(loss, has_aux=True)(params, states, batch)
        return opt_update(i, grads, opt_state), states

    _, init_params, states = net_init(rng, (-1, 2))
    opt_state = opt_init(init_params)
    itercount = itertools.count()

    print("Starting training...")
    for epoch in (t := trange(num_epochs)):
        for batch in range(num_batches):
            # states is not really necessary since the model uses no running parameters.
            # However, the states variable must be passed anyway to satisfy the serial return parameters.
            opt_state, states = update(next(itercount), opt_state, states, (X,Y))

        params = get_params(opt_state)
        train_acc = accuracy(params, states, (X, Y))
        t.set_description_str("Accuracy Train = {:.2%}".format(train_acc))
    print("Training Complete.")

    # # Decision Boundary 3D Plot
    points = []
    for x in jnp.linspace(0, 1, 20):
        for y in jnp.linspace(0, 1, 20):
            inputArray = jnp.array([[x, y]])
            inputArray = jnp.reshape(inputArray, (1, 2))
            predictions, states = net_predict(params, states, inputArray)
            points.append([x, y, predictions[0][0]])
    points = jnp.array(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ax.scatter(x, y, z, c=z, cmap="winter")
    plt.show()

if __name__ == "__main__":
    main()