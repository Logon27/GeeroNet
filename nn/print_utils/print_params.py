import logging


def print_params(params):
    """Helper function to print the parameter shapes of all trainable layers

    Args:
    params: A list of all layer parameters (tuples of arrays). The tuples contain the weight and bias arrays
    """
    weight_max_len_1 = 1
    weight_max_len_2 = 1
    bias_max_len_1 = 1
    bias_max_len_2 = 1
    for param in params:
        if len(param) != 0:
            weight_max_len_1 = max(len(str(param[0].shape[0])), weight_max_len_1)
            weight_max_len_2 = max(len(str(param[0].shape[1])), weight_max_len_2)
            bias_max_len_1 = max(len(str(param[1].shape[0])), bias_max_len_1)
            bias_max_len_2 = max(len(str(param[1].shape[1])), bias_max_len_2)

    logging.info("Printing Parameter Shapes:")
    for param in params:
        # If the layer is not an activation layer
        if len(param) != 0:
            logging.info("Weight Shape: ({:<{}}, {:<{}}), Bias Shape: ({:<{}}, {:<{}})".format(
                param[0].shape[0], weight_max_len_1, param[0].shape[1], weight_max_len_2,
                param[1].shape[0], bias_max_len_1, param[1].shape[1], bias_max_len_2,
            ))