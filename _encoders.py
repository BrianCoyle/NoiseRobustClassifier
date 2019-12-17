from math import pi
import numpy as np


def angle_param_linear(feature_vector, params):
    """Returns the "parameterized linear encoding" of the feature vectors

    theta * feature_vector[0]
    2 * phi * feature_vector[1]
    """

    [theta, phi] = params
    return (theta * feature_vector[0], phi * feature_vector[1])
    # return (theta * feature_vector[0], 0) # For TTN structure


def angle_param_combination(feature_vector, params):
    """Returns the "parameterized linear encoding" of the feature vectors

    theta * feature_vector[0]
    2 * phi * feature_vector[1]
    """
    [theta, phi] = params
    return (theta * feature_vector[0] + phi * feature_vector[1], theta * feature_vector[0] + phi * feature_vector[1])
