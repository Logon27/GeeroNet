# Layers
from .layers.dense import Dense

# Layer Containers
from .composers.serial import serial

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

# Set logging level based on environment variable. See Readme for level details.
import logging
import os
from jax.config import config

valid_debug_modes = {'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'INFO2', 'DEBUG', 'NOTSET'}
LOG_LEVEL = os.environ.get('LOGLEVEL')
if LOG_LEVEL is not None and LOG_LEVEL in valid_debug_modes:
    logging.addLevelName(19, "INFO2")
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=LOG_LEVEL.upper()
    )
    if LOG_LEVEL == "INFO2":
        logging.info("Disabling JIT for better debugging.")
        logging.info("Expect first iteration time estimates to be longer.")
        config.update('jax_disable_jit', True)
else:
    if LOG_LEVEL is not None:
        print("Unknown Log Level! Using basic config.")
    logging.basicConfig()

# Add the ability to toggle jit when DISABLE_JIT = 1
DISABLE_JIT = os.environ.get('DISABLE_JIT')
if DISABLE_JIT is not None and DISABLE_JIT == "1":
    print("Disabling JIT.")
    config.update('jax_disable_jit', True)