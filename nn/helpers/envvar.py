import os


def getenv(key, default=0):
    # Gets the environment variable or the default value if the env var is not set.
    # Then casts the value to the type of 'default'
    return type(default)(os.getenv(key, default))