import jax.numpy as jnp
from jax.image import scale_and_translate
from jax import random
# For image rotation
from scipy.ndimage import rotate
# For image debugging
from PIL import Image as im
import numpy as np

# Need to implement clipped zoom.

# JIT IN-COMPATIBLE
def rotate_image(device_array, angle: float):
    """Rotate jax images for experimentation and debugging.

    Args:
        device_array: 2 dimensional jax array.
        angle (float): The angle to rotate the image.

    Example:
        Usage for mnist. Rotate image 45 degrees::

        image_array = jnp.reshape(train_images[0] * 256, (28,28))
        image_array = rotate_image(image_array, 45)
    
    Returns:
        jax_array: The rotated image as an array.
    """
    device_array = rotate(device_array, angle=angle, reshape=False)
    return device_array

# Performance and parameters can probably be improved.
def noisify_image(device_array, rng, num_noise_iterations: int = 5, percentage_noise = 0.5, noise_value_low: int = 50, noise_value_high: int = 255):
    """Add random noise to jax images for experimentation and debugging.

    For each noise iteration a single noise value is picked. Then that single noise value is applied to the image based on the noise percentage.
    This means for each noise iteration, a noise value can be applied to multiple pixels based on the percentage.

    Args:
        device_array: 2 dimensional jax array.
        rng (PRNGKey): The PRNGKey to pull random values from.
        num_noise_iterations: The number of unique noise values applied to the image.
        percentage_noise: The percentage chance that a single noise value replaces a pixel in the image.
            A single noise value can be applied more than once.
        noise_value_low: The minimum possible noise value.
        noise_value_high: The maximum possible noise value.

    Example:
        Usage for mnist::

        image_array = jnp.reshape(train_images[0] * 256, (28,28))
        image_array = noisify_image(image_array, rng)
    
    Returns:
        jax_array: The noisified image as an array.
    """
    frac = percentage_noise / 100
    # Each iteration only applies a single noise value (possibly multiple times).
    # So typically you want a really low noise percentage and a higher number of iterations.
    # That way you get a bunch of unique noise values applied a small number of times.
    for _ in range(num_noise_iterations):
        rng, noise_rng = random.split(rng)
        random_int = random.randint(noise_rng, (1,), noise_value_low, noise_value_high)
        device_array[random.uniform(noise_rng, device_array.shape) < frac] = random_int
    return device_array

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

# Only used for upscaling images with no interpolation.
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