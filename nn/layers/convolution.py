import functools
from jax import lax
from jax import random
from jax.nn.initializers import glorot_normal, normal, ones, zeros

# N "images" of C channels of H x W feature maps
def GeneralConv(dimension_numbers, out_chan, filter_shape,
                strides=None, padding='VALID', W_init=None,
                b_init=normal(1e-6)):
  """Layer construction function for a general convolution layer."""
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  one = (1,) * len(filter_shape)
  strides = strides or one
  W_init = W_init or glorot_normal(rhs_spec.index('I'), rhs_spec.index('O'))
  def init_fun(rng, input_shape):
    filter_shape_iter = iter(filter_shape)
    kernel_shape = [out_chan if c == 'O' else
                    input_shape[lhs_spec.index('C')] if c == 'I' else
                    next(filter_shape_iter) for c in rhs_spec]
    output_shape = lax.conv_general_shape_tuple(
        input_shape, kernel_shape, strides, padding, dimension_numbers)
    bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
    k1, k2 = random.split(rng)
    W, b = W_init(k1, kernel_shape), b_init(k2, bias_shape)
    return output_shape, (W, b)
  def apply_fun(params, inputs, **kwargs):
    W, b = params
    return lax.conv_general_dilated(inputs, W, strides, padding, one, one,
                                    dimension_numbers=dimension_numbers) + b
  return init_fun, apply_fun
Conv = functools.partial(GeneralConv, ('NHWC', 'HWIO', 'NHWC'))