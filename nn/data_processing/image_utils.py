import jax.numpy as jnp
from jax.image import scale_and_translate
# For image debugging
from PIL import Image as im
import numpy as np

# Helper Functions To Randomize Training Inputs

# Noise method

# Rotation method

# Translation method
def translate_image(device_array, vertical_shift: float, horizontal_shift: float):
    """Translate jax images for experimentation and debugging.

    Args:
        device_array: 2 dimensional jax array.
        vertical_shift (float): Positive values shift down. Negative values shift up.
        horizontal_shift (float): Positive values shift right. Negative values shift left.

    Example:
        Usage for mnist. Shift right and up::

        image_array = jnp.reshape(train_images[0] * 256, (28,28))
        image_array = translate_image(image_array, -3, 3)
    
    Returns:
        jax_array: The translated image as an array.
    """
    if len(device_array.shape) != 2:
        raise ValueError("Array shape {} dimensions, but expected 2 dimensions.".format(len(device_array.shape)))
    
    return scale_and_translate(
        device_array,
        shape=device_array.shape,
        spatial_dims=(0, 1),
        translation=jnp.array([vertical_shift, horizontal_shift]),
        scale=jnp.array([1, 1]),
        method='linear'
    )

def scale_image(device_array, scale_factor: float, method='linear'):
    """Scale jax images (up or down) for experimentation and debugging. This function utilizes interpolation for resizing the image.
    
    Args:
        device_array: 2 dimensional jax array.
        scale_factor (float): Multiplier to scale image up or down.
        method (str): Interpolation method.
    
    Examples:
        Usage for mnist::

        # Triple image size
        image_array = jnp.reshape(train_images[0] * 256, (28,28))
        image_array = scale_image(image_array, 3)

        # Half the image size
        image_array = jnp.reshape(train_images[0] * 256, (28,28))
        image_array = scale_image(image_array, 0.5)

    Returns:
        jax_array: The scaled image as an array.
    """
    valid_methods = {'linear', 'bilinear', 'trilinear', 'cubic', 'bicubic', 'tricubic', 'lanczos3', 'lanczos5'}
    if method not in valid_methods:
        raise ValueError("Resize method must be one of %r." % valid_methods)

    if len(device_array.shape) != 2:
        raise ValueError("Array shape {} dimensions, but expected 2 dimensions.".format(len(device_array.shape)))
    
    return scale_and_translate(
        device_array,
        shape=jnp.array([int(device_array.shape[0] * scale_factor), int(device_array.shape[1] * scale_factor)]),
        spatial_dims=(0, 1),
        translation=jnp.array([0, 0]),
        scale=jnp.array([scale_factor, scale_factor]),
        method=method
    )

# Need to replace this method with something more elaborate so i can scale down images too.
def upscale_image(device_array, scale_factor: int):
    """Used to scale up jax images for experimentation and debugging. This function enlarges the array by repeating existing values. There is no interpolation.

    Args:
        device_array: 2 dimensional jax array.
        scale_factor (int): Multiplier to scale image up.

    Example:
        Usage for mnist::

        # Triple the image size
        image_array = jnp.reshape(train_images[0] * 256, (28,28))
        scale_image(image_array, 3)
    
    Returns:
        jax_array: The scaled up image as an array.
    """
    return jnp.repeat(jnp.repeat(device_array, scale_factor, axis=0), scale_factor, axis=1)

def save_image(device_array, file_name: str):
    """A function to save jax arrays to disk as an image for debugging purposes.
    
    Args:
        device_array: 2 dimensional jax array.

    Example:
        Usage for mnist::

        image_array = jnp.reshape(train_images[0] * 256, (28,28))
        save_image(image_array, "test_image.png")

    Returns:
        None
    """
    # Create an image from the array
    numpy_array = np.asarray(device_array)
    data = im.fromarray(numpy_array)
    data = data.convert("L")
    
    # Saving the final output to file
    data.save(file_name)
    print("Saved Image... {}".format(file_name))