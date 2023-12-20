import jax.numpy as jnp
from jax.image import scale_and_translate, resize
from jax import jit, random
from jax import Array
from jax.random import PRNGKey
from jax.typing import ArrayLike
# Deep mind library that provides jax compatible image processing functions
from dm_pix import rotate, resize_with_crop_or_pad
# For image debugging
from PIL import Image as im
import numpy as np

def zoom_grayscale_image(device_array: ArrayLike, zoom_factor: float) -> Array:
    """Zoom jax images for experimentation and debugging.

    Args:
        device_array: 2 dimensional jax array.
        zoom_factor: The multiplier used to zoom the image. Can be less than or greater than 1.

    Example:
        Usage for mnist. Make contents of image 25% smaller::

        image_array = jnp.reshape(train_images[0] * 256, (28,28))
        image_array = zoom_grayscale_image(image_array, 0.75)
    
    Returns:
        jax_array: The zoomed image as an array.
    """
    out = resize(device_array, (round(device_array.shape[0] * zoom_factor), round(device_array.shape[1] * zoom_factor)), 'nearest')
    # Greyscale to 3 channel image. https://stackoverflow.com/questions/40119743/convert-a-grayscale-image-to-a-3-channel-image
    stacked_img = jnp.stack((out,)*3, axis=-1)
    out = resize_with_crop_or_pad(stacked_img, device_array.shape[0], device_array.shape[1])
    out = jnp.reshape(out[:,:,:1], (device_array.shape[0],device_array.shape[1]))
    return out

def rotate_grayscale_image(device_array: ArrayLike, angle_degrees: float) -> Array:
    """Rotate jax images for experimentation and debugging.

    Args:
        device_array: 2 dimensional jax array.
        angle: The angle (in degrees) to rotate the image.

    Example:
        Usage for mnist. Rotate image 45 degrees::

        image_array = jnp.reshape(train_images[0] * 256, (28,28))
        image_array = rotate_grayscale_image(image_array, 45)
    
    Returns:
        jax_array: The rotated image as an array.
    """
    stacked_img = jnp.stack((device_array,)*3, axis=-1)
    # Need to convert the angle in degrees to radians. Because that is what the rotate function accepts.
    out = rotate(stacked_img, angle=(angle_degrees * (jnp.pi / 180)))
    out = jnp.reshape(out[:,:,:1], (device_array.shape[0],device_array.shape[1]))
    return out

def noisify_grayscale_image(rng: PRNGKey, device_array: ArrayLike, num_noise_iterations: int = 5,
                            percentage_noise: float = 0.5, noise_value_low: float = 0, noise_value_high: float = 255) -> Array:
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
        image_array = noisify_grayscale_image(image_array, rng)
    
    Returns:
        jax_array: The noisified image as an array.
    """
    frac = percentage_noise / 100
    # Each iteration only applies a single noise value (possibly multiple times).
    # So typically you want a really low noise percentage and a higher number of iterations.
    # That way you get a bunch of unique noise values applied a small number of times.
    for _ in range(num_noise_iterations):
        rng, noise_rng = random.split(rng)
        random_int = random.uniform(noise_rng, (1,), minval=noise_value_low, maxval=noise_value_high)
        device_array = jnp.where(random.uniform(noise_rng, device_array.shape) > frac, device_array, random_int)
    return device_array

def translate_grayscale_image(device_array: ArrayLike, vertical_shift: float, horizontal_shift: float) -> Array:
    """Translate jax images for experimentation and debugging.

    Args:
        device_array: 2 dimensional jax array.
        vertical_shift: Positive values shift down. Negative values shift up.
        horizontal_shift: Positive values shift right. Negative values shift left.

    Example:
        Usage for mnist. Shift right and up::

        image_array = jnp.reshape(train_images[0] * 256, (28,28))
        image_array = translate_grayscale_image(image_array, -3, 3)
    
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

def scale_grayscale_image(device_array: ArrayLike, scale_factor: float, method='linear') -> Array:
    """Scale jax images (up or down) for experimentation and debugging. This function utilizes interpolation for resizing the image.
    
    Args:
        device_array: 2 dimensional jax array.
        scale_factor: Multiplier to scale image up or down.
        method (str): Interpolation method.
    
    Examples:
        Usage for mnist::

        # Triple image size
        image_array = jnp.reshape(train_images[0] * 256, (28,28))
        image_array = scale_grayscale_image(image_array, 3)

        # Half the image size
        image_array = jnp.reshape(train_images[0] * 256, (28,28))
        image_array = scale_grayscale_image(image_array, 0.5)

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

# Only used for upscaling images with no interpolation. This can probably be reimplemented with jax.image.resize
def upscale_grayscale_image(device_array: ArrayLike, scale_factor: int) -> Array:
    """Used to scale up jax images for experimentation and debugging. This function enlarges the array by repeating existing values. There is no interpolation.

    Args:
        device_array: 2 dimensional jax array.
        scale_factor: Multiplier to scale image up.

    Example:
        Usage for mnist::

        # Triple the image size
        image_array = jnp.reshape(train_images[0] * 256, (28,28))
        upscale_grayscale_image(image_array, 3)
    
    Returns:
        jax_array: The scaled up image as an array.
    """
    return jnp.repeat(jnp.repeat(device_array, scale_factor, axis=0), scale_factor, axis=1)

def save_grayscale_image(device_array: ArrayLike, file_name: str):
    """A function to save jax arrays to disk as an image for debugging purposes.
    
    Args:
        device_array: 2 dimensional jax array.
        file_name: File name for the saved image. Alternatively you can provide a filepath.

    Example:
        Usage for mnist::

        # Times 255 because the image values are normalized.
        image_array = jnp.reshape(train_images[0] * 255, (28,28))
        save_grayscale_image(image_array, "test_image.png")

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

def save_rbg_image(device_array: ArrayLike, file_name: str):
    """A function to save jax arrays to disk as an image for debugging purposes.
    
    Args:
        device_array: 3 dimensional jax array. Channel being the last dimension
        file_name: File name for the saved image. Alternatively you can provide a filepath.

    Example:
        Usage for cifar::

        image_array = jnp.array((train_images[0] * 255), dtype="int8")
        save_rbg_image(image_array, "test_image.png")

    Returns:
        None
    """
    # Create an image from the array
    numpy_array = np.asarray(device_array)
    data = im.fromarray(numpy_array, "RGB")
    
    # Saving the final output to file
    data.save(file_name)
    print("Saved Image... {}".format(file_name))