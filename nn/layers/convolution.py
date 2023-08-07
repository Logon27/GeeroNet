import functools
from jax import lax
from jax import random
from jax.nn.initializers import glorot_normal, normal
from nn.decorators.convolution_decorator import convolution_decorator

# N - batch dimension
# H - spatial height
# W - spatial width
# C - channel dimension
# I - kernel input channel dimension
# O - kernel output channel dimension

# The first string "NHWC" indicates that your inputs are batch, height, width, channel. 
# The second string "HWIO" indicates that your kernel weights are height, width, kernel input, kernel output
# The last string "NHWC" indicates that the outputs of the convolution are batch, height, width, channel.

# N "images" of C channels of H x W feature maps
@convolution_decorator
def GeneralConv(dimension_numbers, out_chan, filter_shape, strides=None, padding='VALID', W_init=None, b_init=normal(1e-6)):
  """Layer construction function for a general convolution layer."""
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  one = (1,) * len(filter_shape)
  strides = strides or one
  W_init = W_init or glorot_normal(rhs_spec.index('I'), rhs_spec.index('O'))

  def init_fun(rng, input_shape):
    # Investigate if the kernels shapes are based on the batch sizing.
    filter_shape_iter = iter(filter_shape)
    kernel_shape = [out_chan if c == 'O' else
                    input_shape[lhs_spec.index('C')] if c == 'I' else
                    next(filter_shape_iter) for c in rhs_spec]
    output_shape = lax.conv_general_shape_tuple(input_shape, kernel_shape, strides, padding, dimension_numbers)
    bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
    k1, k2 = random.split(rng)
    weights, bias = W_init(k1, kernel_shape), b_init(k2, bias_shape)
    return output_shape, (weights, bias)
  
  def apply_fun(params, inputs, **kwargs):
    weights, bias = params
    return lax.conv_general_dilated(inputs, weights, strides, padding, one, one, dimension_numbers=dimension_numbers) + bias
  return init_fun, apply_fun
Conv = functools.partial(GeneralConv, ('NHWC', 'HWIO', 'NHWC'))