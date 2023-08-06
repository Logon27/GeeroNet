import dill
from jax.typing import ArrayLike
from jax import Array

# Provides the ability to save and load network paramaters from files.
# The params array is saved as an object to a file.

def save_params(params: ArrayLike, file_path: str):
    with open(file_path, "wb") as file:
        print("Saving params to file {}...".format(file_path))
        dill.dump(params, file, dill.HIGHEST_PROTOCOL)
        print("Save complete.")

def load_params(file_path: str) -> Array:
    with open(file_path, "rb") as file:
        print("Loading params from file {}...".format(file_path))
        params = dill.load(file)
        print("Loading complete.")
        return params