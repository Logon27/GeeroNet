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

# Set logging level based on environment variable
import logging
import os
valid_debug_modes = {'WARN', 'INFO', 'DEBUG'}
if os.environ.get('LOGLEVEL') is not None and os.environ.get('LOGLEVEL') in valid_debug_modes:
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=os.environ.get('LOGLEVEL').upper()
    )
else:
    logging.basicConfig()