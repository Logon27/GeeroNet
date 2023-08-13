# Layers
from .layers.dense import Dense
from .layers.convolution import Conv
from .layers.flatten import Flatten
from .layers.reshape import Reshape
from .layers.dropout import Dropout

# Layer Combinators
from .combinators.serial import serial

# Activation Functions
from .activations.activations import Tanh, Relu, Exp, LogSoftmax, Softmax, Softplus, Sigmoid, Elu, LeakyRelu, Selu, Gelu

# Optimizers
from .optimizers.sgd import sgd
from .optimizers.momentum import momentum

# Loss Functions
from .losses.mean_squared_error import mean_squared_error
from .losses.categorical_cross_entropy import categorical_cross_entropy
from .losses.binary_cross_entropy import binary_cross_entropy

# Image processing functionality
from .data_processing.image_utils import save_grayscale_image, upscale_grayscale_image, scale_grayscale_image, translate_grayscale_image, \
    noisify_grayscale_image, rotate_grayscale_image, zoom_grayscale_image

# Saving and loading weights from file
from .data_processing.file_io import save_params, load_params

# Set logging level based on environment variable. See Readme for level details.
import logging
import os
from jax.config import config
import jax

def getenv(key, default=0):
    # Gets the environment variable or the default value if the env var is not set.
    # Then casts the value to the type of 'default'
    return type(default)(os.getenv(key, default))

LOG_LEVEL = getenv('LOGLEVEL', 'WARNING')
DISABLE_JIT = getenv('DISABLE_JIT', 0)

valid_debug_modes = {'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'INFO2', 'DEBUG', 'NOTSET'}
if LOG_LEVEL in valid_debug_modes:
    logging.addLevelName(19, "INFO2")
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=LOG_LEVEL.upper()
    )
    if LOG_LEVEL == "INFO2":
        logging.info("Disabling JIT for better debugging.")
        logging.info("Expect first iteration time estimates to be longer.")
        config.update('jax_disable_jit', True)
        logging.info("The '*' character in debug prints is a wildcard representing the batch size.")
        logging.info("The wildcard was added because the batch size can vary between forward passes after initialization.")
else:
    if LOG_LEVEL is not None:
        print("Unknown Log Level! Using basic config.")
    logging.basicConfig()

# Add the ability to toggle jit when DISABLE_JIT = 1
if DISABLE_JIT == 1:
    print("Disabling JIT.")
    config.update('jax_disable_jit', True)

print("Running On Jax Platform: {}".format(jax.default_backend().upper()))
