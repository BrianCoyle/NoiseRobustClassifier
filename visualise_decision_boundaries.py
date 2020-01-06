import matplotlib.pyplot as plt

from matplotlib import cm
import numpy as np

from pyquil import get_qc

from nisqai.data._cdata import CData
from pyquil.gates import *
from scipy.optimize import minimize
from collections import Counter
import random
from sklearn.datasets import make_circles, make_moons

from classifier_circuits import *
from data_gen import *
from single_qubit_classifier import train_classifier, ClassificationCircuit, classifier_params
from plots import plot_correct_classifications, scatter
from data_gen import generate_data, remove_zeros

def main(encoding='denseangle_param'):
    """
    This function prints out a random decision boundary generated by a particular encoding
    over a grid with two features. The encoding parameters are fixed as standard.
    """

    qc_name = '1q-qvm'
    qc = get_qc(qc_name)
    num_shots = 1024
    device_qubits = qc.qubits()
    classifier_qubits = device_qubits
    if encoding.lower() == 'wavefunction_param':
        params = np.array([0.45811744, 0.2575122, 0.52902198])
    else:
        params = np.random.rand(3)
    n_layers = 1
    if encoding.lower() == 'denseangle_param':
        encoding_choice = 'denseangle_param'
        init_encoding_params = [np.pi, 2*np.pi]
    elif encoding.lower() == 'wavefunction_param':
        encoding_choice = 'wavefunction_param'
        init_encoding_params = [0]
    elif encoding.lower() == 'superdenseangle_param':
        encoding_choice = 'superdenseangle_param'
        init_encoding_params = [np.pi, 2*np.pi]
    else: raise NotImplementedError
    '''
    # Generate Grid of datapoints to determine and visualise ideal decision boundary
    '''
    data_choice = 'full_vertical_boundary'
    num_grid_points = 2000
    data_grid, grid_true_labels = generate_data(data_choice, num_grid_points)
    data_grid, grid_true_labels = remove_zeros(data_grid, grid_true_labels)

    predicted_labels_grid = ClassificationCircuit(classifier_qubits, data_grid).make_predictions(params, n_layers, encoding_choice, init_encoding_params, \
                                                                num_shots, qc)
    plot_params = {'colors': ['blue', 'orange'], 'alpha': 1, 'size': 70}
    scatter(data_grid, predicted_labels_grid, **plot_params)
    plt.show()


if __name__ == "__main__":
    main(encoding='superdenseangle_param')
