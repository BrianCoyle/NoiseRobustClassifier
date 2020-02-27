import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from matplotlib import cm

plt.rcParams.update({"font.size": 14, "font.serif": "Computer Modern Roman"})


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

"""
    This file contains plotting functions & generates figures for Measurement & Pauli Noise
"""

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

def plot_number_misclassified_meas_noise(ideal_params, num_shots, encoding_choice, encoding_params, qc):

    points_noise_inc = []
    
    data_train, data_test, true_labels_train, true_labels_test = generate_data('random_vertical_boundary', num_points=500, split=True)
    n_layers = 1
    device_qubits = qc.qubits()

    classifier_qubits = [device_qubits[0]]

    noise_strengths00 = np.arange(0, 1, 0.02)
    noise_strengths11 = np.arange(0, 1, 0.02)
    X, Y = np.meshgrid(noise_strengths00, noise_strengths11)

    noise_type = "Measurement"
    proportion_correctly_classified = np.zeros((noise_strengths00.shape[0], noise_strengths11.shape[0]), dtype=float)
    for ii, p00 in enumerate(noise_strengths00):
        for jj, p11 in enumerate(noise_strengths11):
            p = [p00, p11] # measurement noise for both outcomes
    
            noisy_predictions, proportion_correctly_classified[ii,jj]  = generate_noisy_classification(ideal_params, n_layers, noise_type, p,\
                                                                                                        encoding_choice, encoding_params, qc, classifier_qubits,\
                                                                                                        num_shots, data_test, true_labels_test)
            print('For parameters: ', p00, p11, ', the proportion misclassified is:', 1 - proportion_correctly_classified[ii,jj])

    plt.imshow(proportion_correctly_classified, cmap=cm.coolwarm_r, extent=[0,1,0,1], origin=(0,0))

    plt.xlabel(r'$p_{00}$')
    plt.ylabel(r'$p_{11}$')

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('% Classified Correctly')
    plt.savefig('meas_noise_flat.pdf')
    plt.show()

    return

def plot_number_misclassified_pauli_noise(ideal_params, num_shots, encoding_choice, encoding_params, qc):
    
    points_noise_inc = []
    
    data_train, data_test, true_labels_train, true_labels_test = generate_data('random_vertical_boundary', num_points=500, split=True)
    
    n_layers = 1
    device_qubits = qc.qubits()

    classifier_qubits = [device_qubits[0]]

    noise_strengthsX = np.arange(0, 0.5, 0.01)
    noise_strengthsY = np.arange(0, 0.5, 0.01)
    X, Y = np.meshgrid(noise_strengthsX, noise_strengthsY)

    noise_type = "pauli_before_measurement"
    proportion_correctly_classified = np.zeros((noise_strengthsX.shape[0], noise_strengthsY.shape[0]), dtype=float)
    for ii, pX in enumerate(noise_strengthsX):
        for jj, pY in enumerate(noise_strengthsY):
            p = [1-pX-pY, pX, pY, 0] # pauli X and Y noise
            noisy_predictions, proportion_correctly_classified[ii,jj]  = generate_noisy_classification(ideal_params, n_layers,
                                                                                                    noise_type, p, encoding_choice, encoding_params,\
                                                                                                    qc, classifier_qubits, num_shots,\
                                                                                                    data_test, true_labels_test)
            print('For parameters: ', pX, pY, ', the proportion misclassified is:', 100*(1 - proportion_correctly_classified[ii,jj]))

    plt.imshow(np.rot90(proportion_correctly_classified), cmap=cm.coolwarm_r, extent=[0,0.5,0,0.5])

    plt.xlabel(r'$p_{X}$')
    plt.ylabel(r'$p_{Y}$')

    cbar = plt.colorbar()
    cbar.ax.set_ylabel('% Classified Correctly')
    plt.savefig('pauli_noise_flat.pdf')
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

    # plt.rcParams.update({"font.size": 16, "font.family": "serif", "font.weight": "bold"})
    plt.rcParams.update({"font.size": 12, "font.serif": "Computer Modern Roman"})

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
        if type(predictions) is not np.array or type(labels) is not np.array:
            num_wrong = 0
            for ii, pred_label in enumerate(predictions):
                num_wrong += abs( labels[ii] - pred_label )
        else:
            num_wrong = sum(abs(labels - predictions))
            percent_correct = 100.0 - num_wrong / len(labels) * 100.0
            plt.title("Test Accuracy: %0.3f" % percent_correct + "%")
    
    if show: plt.show()

    
def plot_encoding_algo(datasets, encodings, ideal_params, init_encoding_params, ideal_encoding_params, costs):

    fig, ax = plt.subplots(1, len(datasets), sharex=True, sharey=False)
    x = np.arange(len(encodings)) / 10

    encoding_labels = [r'$\mathsf{DAE}$', r'$\mathsf{WF}$', r'$\mathsf{SDAE}$']
    legend_labels = ['Ideal', 'Untrained Encoding + Noise', 'Trained Encoding + Noise']

    [ideal_costs_mean, ideal_costs_std,\
    noisy_costs_mean, noisy_costs_std,\
    noisy_costs_uncorrected_mean, noisy_costs_uncorrected_std] = costs
    data_names = []
    colors = ['blue', 'green', 'orange']
    for ii, data in enumerate(datasets):
        print('For dataset', data, 'the ideal mean is:', ideal_costs_mean[ii])
        print('For dataset', data, 'the ideal std is:', ideal_costs_std[ii])
        
        print('For dataset', data, 'the uncorrected mean is:', noisy_costs_uncorrected_mean[ii])
        print('For dataset', data, 'the uncorrected std is:', noisy_costs_uncorrected_std[ii])

        print('For dataset', data, 'the corrected noisy mean is:', noisy_costs_mean[ii])
        print('For dataset', data, 'the corrected noisy std is:', noisy_costs_std[ii])

        ax[ii].scatter(x, ideal_costs_mean[ii], s = 150 , marker = "x", color='blue')
        ax[ii].scatter(x, noisy_costs_uncorrected_mean[ii], s = 150 , marker = "^", color='green')
        ax[ii].scatter(x, noisy_costs_mean[ii], s = 150 , marker = "o", color='orange')

        ax[ii].errorbar(x, ideal_costs_mean[ii], yerr=ideal_costs_std[ii], fmt='x', linewidth=3, capsize=10, capthick=3, color='blue')
        ax[ii].errorbar(x, noisy_costs_uncorrected_mean[ii], yerr=noisy_costs_uncorrected_std[ii], fmt='^', linewidth=3, capsize=10, capthick=3, color='green')
        ax[ii].errorbar(x, noisy_costs_mean[ii], yerr=noisy_costs_std[ii], fmt='o', linewidth=3, capsize=10, capthick=3, color='orange')

        if data.lower() == 'moons':
            data_names.append('moons')
        elif data.lower() == 'random_vertical_boundary':
            data_names.append('vertical')
        elif data.lower() == 'random_diagonal_boundary':
            data_names.append('diagonal')
        
        ax[ii].set_xticks(x)
        ax[ii].set_xticklabels([],  rotation='30')
       
        ideal_params_rounded = [ ['%.2f' % param for param in encoding] for encoding in ideal_params[ii] ]
        init_encoding_params_rounded = [ ['%.2f' % param for param in encoding] for encoding in init_encoding_params[ii] ]
        ideal_encoding_params_rounded = [ ['%.2f' % param for param in encoding] for encoding in ideal_encoding_params[ii] ]

    # ax[-2].legend(legend_labels, loc='lower right')
    plt.show()
    return

def fidelity_bound_plot(average_fidelity, encoding_params, noise_choice='amp_damp',show=False, num_points=50, num_qbs=1, legend=False): 

    [encoding_params_dae, encoding_params_wf, encoding_params_sdae] =  encoding_params
    if noise_choice.lower()== 'global_depolarizing':
        noise_values_arr = np.linspace(0, 0.9, num=num_points, endpoint=True) # Cut off endpoint - misclassified due to numerical imprecision
    else:
        noise_values_arr = np.linspace(0, 1, num=num_points, endpoint=False)

    [cost_diff_plot_dae, mixed_bound_plot_dae, avg_bound_plot_dae] = [np.zeros(len(noise_values_arr)) for _ in range(3)]
    [cost_diff_plot_wf, mixed_bound_plot_wf, avg_bound_plot_wf] = [np.zeros(len(noise_values_arr)) for _ in range(3)]
    [cost_diff_plot_sdae, mixed_bound_plot_sdae, avg_bound_plot_sdae] = [np.zeros(len(noise_values_arr)) for _ in range(3)]

    for ii, noise_value in enumerate(noise_values_arr):
        if num_qbs == 1:
            noise_values = [ noise_value ]
        elif num_qbs == 2:
            if noise_choice.lower() == 'amp_damp':             noise_values = [ noise_value, noise_value ]
            elif noise_choice.lower() == 'bit_flip':           noise_values = [ noise_value, noise_value ]
            elif noise_choice.lower() == 'global_depolarizing':noise_values = noise_value 
            elif noise_choice.lower() == 'dephasing':          noise_values = [ noise_value, noise_value ]

        else: raise NotImplementedError('Only have 1 or 2 qubits')
        cost_diff_plot_dae[ii], avg_bound_plot_dae[ii], mixed_bound_plot_dae[ii], _, _ = \
            average_fidelity(encoding_choice='denseangle_param', encoding_params=encoding_params_dae, noise_choice=noise_choice, noise_values=noise_values)

        cost_diff_plot_wf[ii], avg_bound_plot_wf[ii], mixed_bound_plot_wf[ii], _, _ = \
            average_fidelity(encoding_choice='wavefunction_param', encoding_params=encoding_params_wf, noise_choice=noise_choice, noise_values=noise_values)
        
        cost_diff_plot_sdae[ii], avg_bound_plot_sdae[ii], mixed_bound_plot_sdae[ii], _, _ = \
            average_fidelity(encoding_choice='superdenseangle_param', encoding_params=encoding_params_sdae, noise_choice=noise_choice, noise_values=noise_values)
    
    plt.plot(noise_values_arr, mixed_bound_plot_dae, '-o', color='blue', label= r'2$\sqrt{1-F(\rho_{{mix}}, \rho_{{mix}}^{{noise}})}$, DAE')
    plt.plot(noise_values_arr, avg_bound_plot_dae, '-o', color='cornflowerblue', label= r'2 Avg$(\sqrt{1-F(\tilde{\rho}_{\mathbf{x}_i}, \mathcal{E}(\tilde{\rho}_{\mathbf{x}_i}))}$], DAE')
    plt.plot(noise_values_arr, cost_diff_plot_dae, '--o', color='darkturquoise', label='Cost Function Error, DAE')
    
    plt.plot(noise_values_arr, mixed_bound_plot_wf, '-x',color='red',  label= r'2$\sqrt{1-F(\rho_{{mix}}, \rho_{{mix}}^{{noise}})}$, WF')
    plt.plot(noise_values_arr, avg_bound_plot_wf, '-x', color='lightcoral', label= r'2 Avg$(\sqrt{1-F(\tilde{\rho}_{\mathbf{x}_i}, \mathcal{E}(\tilde{\rho}_{\mathbf{x}_i}))}$], WF')
    plt.plot(noise_values_arr, cost_diff_plot_wf, '--x',color='magenta',  label='Cost Function Error, WF')

    plt.plot(noise_values_arr, mixed_bound_plot_sdae, '-s',color='saddlebrown',  label= r'2$\sqrt{1-F(\rho_{{mix}}, \rho_{{mix}}^{{noise}})}$, SDAE')
    plt.plot(noise_values_arr, avg_bound_plot_sdae, '-s', color='goldenrod', label= r'2 Avg$(\sqrt{1-F(\tilde{\rho}_{\mathbf{x}_i}, \mathcal{E}(\tilde{\rho}_{\mathbf{x}_i}))}$], SDAE')
    plt.plot(noise_values_arr, cost_diff_plot_sdae, '--s', color='darkorange', label='Cost Function Error, SDAE')
    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    if show:
        plt.show()
    return


def fidelity_compare_plot(average_fidelity, encoding_params, noise_choice='amp_damp',show=False, num_points=50, num_qbs=1, legend=False): 

    [encoding_params_dae, encoding_params_wf, encoding_params_sdae] =  encoding_params
    if noise_choice.lower()== 'global_depolarizing':
        noise_values_arr = np.linspace(0, 0.9, num=num_points, endpoint=True) # Cut off endpoint - misclassified due to numerical imprecision
    else:
        noise_values_arr = np.linspace(0, 1, num=num_points, endpoint=False)

    [avg_fidelity_dae, avg_fidelity_mixed_dae] = [np.zeros(len(noise_values_arr)) for _ in range(2)]
    [avg_fidelity_wf, avg_fidelity_mixed_wf] = [np.zeros(len(noise_values_arr)) for _ in range(2)]
    [avg_fidelity_sdae, avg_fidelity_mixed_sdae] = [np.zeros(len(noise_values_arr)) for _ in range(2)]

    for ii, noise_value in enumerate(noise_values_arr):
        if num_qbs == 1:
            noise_values = [ noise_value ]
        elif num_qbs == 2:
            if noise_choice.lower() == 'amp_damp':             noise_values = [ noise_value, noise_value ]
            elif noise_choice.lower() == 'bit_flip':           noise_values = [ noise_value, noise_value ]
            elif noise_choice.lower() == 'global_depolarizing':noise_values = noise_value 
            elif noise_choice.lower() == 'dephasing':          noise_values = [ noise_value, noise_value ]
        else: raise NotImplementedError('Only have 1 or 2 qubits')
        _, _, _, avg_fidelity_dae[ii], avg_fidelity_mixed_dae[ii] = \
            average_fidelity(encoding_choice='denseangle_param', encoding_params=encoding_params_dae, noise_choice=noise_choice, noise_values=noise_values)

        _, _, _, avg_fidelity_wf[ii], avg_fidelity_mixed_wf[ii] = \
            average_fidelity(encoding_choice='wavefunction_param', encoding_params=encoding_params_wf, noise_choice=noise_choice, noise_values=noise_values)
        
        _, _, _, avg_fidelity_sdae[ii], avg_fidelity_mixed_sdae[ii] = \
            average_fidelity(encoding_choice='superdenseangle_param', encoding_params=encoding_params_sdae, noise_choice=noise_choice, noise_values=noise_values)
    
    plt.plot(noise_values_arr, avg_fidelity_dae, '-o',color='darkturquoise', label= r'Average Fidelity, DAE')
    plt.plot(noise_values_arr, avg_fidelity_mixed_dae, '--o', color = 'blue', label= r'$F(\rho_{{mix}}, \rho_{{mix}}^{{noise}})}$, DAE')
    
    plt.plot(noise_values_arr, avg_fidelity_wf, '-x', color='magenta', label= r'Average Fidelity, WF')
    plt.plot(noise_values_arr, avg_fidelity_mixed_wf, '--x', color='red', label= r'$F(\rho_{{mix}}, \rho_{{mix}}^{{noise}})}$, WF')

    plt.plot(noise_values_arr, avg_fidelity_sdae, '-s', color='darkorange', label= r'Average Fidelity, SDAE')
    plt.plot(noise_values_arr, avg_fidelity_mixed_sdae, '--s', color='saddlebrown', label= r'$F(\rho_{{mix}}, \rho_{{mix}}^{{noise}})}$, SDAE')
    if legend:
        plt.legend(loc='center left', bbox_to_anchor=(1.05, 0.5))

    if show:
        plt.show()
    return
    
if __name__ == "__main__":

    """
        Generate Pauli and Measurement Noise Plots
    """
    ideal_params = [1.67814786, 1.56516469, 1.77820848]
    encoding_choice = 'denseangle_param'
    encoding_params = [np.pi, 2*np.pi]

    qc_name = '1q-qvm'
    qc = get_qc(qc_name)
    num_shots = 512
    pauli = True
    Meas = True
    if pauli:
        plot_number_misclassified_pauli_noise(ideal_params, num_shots, encoding_choice, encoding_params, qc)
    if meas:
        plot_number_misclassified_meas_noise(ideal_params, num_shots, encoding_choice, encoding_params, qc)


