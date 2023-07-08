# Layers
from .layers.dense import Dense

# Layer Containers
from .containers.serial import serial

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