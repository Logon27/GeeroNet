# Set logging level based on environment variable. See Readme for level details.
import logging
from jax.config import config
import jax
from .helpers.envvar import getenv

LOG_LEVEL = getenv('LOGLEVEL', 'WARNING')
DISABLE_JIT = getenv('DISABLE_JIT', 0)

valid_debug_modes = {'CRITICAL', 'ERROR', 'WARNING', 'INFO', 'INFO2', 'DEBUG', 'NOTSET'}
if LOG_LEVEL in valid_debug_modes:
    logging.addLevelName(21, "INFO2")
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=LOG_LEVEL.upper()
    )
    if LOG_LEVEL == "INFO2":
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

print("Running On Jax Platform: {}\n".format(jax.default_backend().upper()))
