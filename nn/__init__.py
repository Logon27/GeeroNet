# Layers
from .layers.dense import Dense

# Layer Containers
from .composers.serial import serial

# Optimizers
from .optimizers.sgd import sgd
from .optimizers.momentum import momentum

# Loss Functions
from .losses.mean_squared_error import mean_squared_error
from .losses.categorical_cross_entropy import categorical_cross_entropy
from .losses.binary_cross_entropy import binary_cross_entropy

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
