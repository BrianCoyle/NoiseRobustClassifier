import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from matplotlib import cm


import numpy as np
from numpy import ndarray
import nisqai
from pyquil import get_qc, Program
from pyquil.api import WavefunctionSimulator
from nisqai.encode._feature_maps import FeatureMap
from nisqai.encode._encoders import angle_simple_linear
from nisqai.layer._base_ansatz import BaseAnsatz
from nisqai.layer._params import Parameters
from nisqai.data._cdata import CData
from nisqai.cost._classical_costs import indicator
from pyquil.gates import *
from scipy.optimize import minimize
from collections import Counter
import random
from sklearn.datasets import make_circles, make_moons


from single_qubit_classifier import  generate_noisy_classification
from data_gen import *


def plot_correct_classifications(true_labels, predicted_labels, data):
    """
    Plot the ideal, correctly classified data. Should be entirely blue in ideal case
    """
    correct_labels = []
    for ii in range(data.shape[0]):    
        if indicator(true_labels[ii], predicted_labels[ii]) == 0: 
            #If label is correct for particular point
            correct_labels.append(0)
        else:
            correct_labels.append(1)
    
    plt.scatter(data.T[0], data.T[1], c=correct_labels, cmap='rainbow')
    plt.show()
    return


def plot_number_misclassified_shot_point(ideal_params, noise_type, noise_strength, qc):

    points_data_inc = []
    num_shots = 500
    for n_x in [2*i for i in range(2, 11)]:
        n_y = n_x
        data, true_labels = generate_random_data_labels(n_x, n_y)
        noisy_predictions, number_misclassified = generate_noisy_classification(ideal_params, noise_type, noise_strength, qc, num_shots, data, true_labels)
        points_data_inc.append([n_x*n_y, number_misclassified])
    
    points_data_inc_arr = np.array(points_data_inc)

    data, true_labels = generate_random_data_labels(10, 10)
    points_shots_inc = []

    for num_shots in [20*i for i in range(1, 20)]:
        
        noisy_predictions, number_misclassified = generate_noisy_classification(ideal_params, noise_type, noise_strength, qc, num_shots, data, true_labels)
        points_shots_inc.append([num_shots, number_misclassified])

    points_shot_inc_arr = np.array(points_shots_inc)

    fig, axs = plt.subplots(2)

    axs[0].plot(points_data_inc_arr[: , 0], points_data_inc_arr[: , 1]) 
    axs[1].plot(points_shot_inc_arr[: , 0], points_shot_inc_arr[: , 1])

    plt.show()

def plot_number_misclassified_singleparam_noise_stren(ideal_params, encoding_choice, encoding_params, data, true_labels, noise_choice, num_shots, qc):
    points_noise_inc = []
    
    noise_strengths = np.arange(0, 1, 0.05)

    points_to_plot = []
    for noise_str in noise_strengths:

        if noise_choice.lower() == "pauli_before_measurement":
            p = [1-noise_str, noise_str, 0, 0]
        elif noise_choice.lower() == "amp_damp_before_measurement":
            p = noise_str
        noisy_predictions, number_misclassified = generate_noisy_classification(ideal_params, noise_choice, p, encoding_choice, encoding_params, qc, num_shots, data, true_labels)
        points_to_plot.append([noise_str, number_misclassified])
        print(points_to_plot)

    points_to_plot_array = np.array(points_to_plot)

    fig, axs = plt.subplots(1)

    axs.plot(points_to_plot_array[:, 0], points_to_plot_array[:, 1]) 

    plt.show()

    return

def plot_number_misclassified_meas_noise_3d(ideal_params, num_shots, encoding_choice, encoding_params, num_points, qc):

    points_noise_inc = []
    
    data, true_labels = generate_random_data_labels(num_points, num_points)
    noise_strengths00 = np.arange(0, 1, 0.02)
    noise_strengths11 = np.arange(0, 1, 0.02)
    X, Y = np.meshgrid(noise_strengths00, noise_strengths11)

    noise_type = "Measurement"
    proportion_misclassified = np.zeros((noise_strengths00.shape[0], noise_strengths11.shape[0]), dtype=float)
    for ii, p00 in enumerate(noise_strengths00):
        for jj, p11 in enumerate(noise_strengths11):
            p = [p00, p11] # measurement noise for both outcomes
            noisy_predictions, proportion_misclassified[ii,jj]  = generate_noisy_classification(ideal_params, noise_type, p, encoding_choice, encoding_params, qc, num_shots, data, true_labels)
            print(proportion_misclassified)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(X, Y, proportion_misclassified, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    ax.set_zlim(-0.01, 1.01)
    fig.colorbar(surf)
    plt.show()

    return

def scatter(data, labels=None, predictions=None, show=False, **kwargs):
    """Shows a scatter plot for two-dimensional data. Points are colored by label if labels are provided.

    Args:
        data : numpy.ndarray
            Two-dimensional data to visualize. Each row should be a data point, and the number of columns is the
            total number of data points.

        labels : array-like
            Array of labels (nominally valued 0 or 1). Must be of the same linear dimension as data.

        predictions : array-like
            Array of predictions (nominally valued 0 or 1). Must be of the same linear dimension as data.

        color1 : str
            Color to use for first class of data when plotting.

        color2: str
            Color to use for second class of data when plotting.
    """
    # TODO: Allow for these in **kwargs
    if kwargs is not None and 'colors' in kwargs: COLORS = kwargs['colors']
    else: COLORS =  ["blue", "orange", "green", "salmon", "red", "black", "purple"]
    
    if kwargs is not None and 'alpha' in kwargs: ALPHA = kwargs['alpha']
    else: ALPHA =  0.8

    if kwargs is not None and 'size' in kwargs: SIZE = kwargs['size']
    else: SIZE = 100

    if kwargs is not None and 'linewidth' in kwargs: LINEWIDTH = kwargs['linewidth']
    else: LINEWIDTH = 2

    plt.rcParams.update({"font.size": 16, "font.family": "serif", "font.weight": "bold"})

    # Make sure we have the correct input type
    if not isinstance(data, ndarray):
        raise ValueError("data must be of type numpy.ndarray.")

    # Get the shape of the data
    num_points, num_features = data.shape

    # Make sure the data dimension is supported
    if num_features != 2:
        raise DimensionError(
            "Data must be two-dimensional. (Number of features must be two)."
        )

    # If we just get predictions and no labels, color the plot as if the predictions were labels
    if labels is None and predictions is not None:
        labels = predictions
        predictions = None

    # If labels are provided
    if labels is not None:
        # Make sure there is at least one unique label
        if len(set(labels)) < 1:
            raise ValueError("Invalid number of labels. There should be at least one unique label.")

        # Get the unique labels
        unique_labels = list(set(labels))
        if len(unique_labels) > len(COLORS):
            RuntimeWarning(
                "There are too many classes for supported colors. Duplicate colors will be used." +
                "To avoid this, pass in a list of string colors to scatter as a kwarg."
            )
            COLORS = COLORS * len(unique_labels)
        color_for_label = dict(zip(unique_labels, COLORS[:len(unique_labels)]))

        # Scatter the data points with labels but no predictions
        if predictions is None:
            for point, label in zip(data, labels):
                plt.scatter(point[0], point[1], color=color_for_label[label], s=SIZE, alpha=ALPHA)

        # Scatter the data points with labels and predictions
        else:
            for point, label, prediction in zip(data, labels, predictions):
                plt.scatter(
                    point[0], point[1], color=color_for_label[label], edgecolor=color_for_label[prediction],
                    linewidth=LINEWIDTH, s=SIZE, alpha=ALPHA
                )

    # If neither labels nor predictions are not provided
    else:
        for point in data:
            # Scatter the points
            plt.scatter(point[0], point[1], s=SIZE, alpha=ALPHA)

    # Put the score on the title
    if predictions is not None and labels is not None:
        num_wrong = sum(abs(labels - predictions))
        percent_correct = 100.0 - num_wrong / len(labels) * 100.0
        plt.title("Test Accuracy: %0.3f" % percent_correct + "%")
    
    if show: plt.show()
    
