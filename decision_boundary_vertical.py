from single_qubit_classifier import generate_noisy_classification
import matplotlib.pyplot as plt
import numpy as np

import nisqai
from pyquil import get_qc, Program
from pyquil.api import WavefunctionSimulator
from _dense_angle_encoding import DenseAngleEncoding
from nisqai.encode import WaveFunctionEncoding
from nisqai.encode._feature_maps import FeatureMap
from nisqai.encode._encoders import angle_simple_linear
from _encoders import angle_param_linear
from nisqai.layer._base_ansatz import BaseAnsatz
from nisqai.layer._params import Parameters
from nisqai.data._cdata import CData, LabeledCData
from nisqai.cost._classical_costs import indicator
from pyquil.gates import *
from scipy.optimize import minimize
from collections import Counter
import random
from sklearn.datasets import make_circles, make_moons

from noise_models import add_noisy_gate_to_circ, estimate_meas_noise
from classifier_circuits import *
from data_gen import *
from single_qubit_classifier import train_classifier, train_classifier_encoding, ClassificationCircuit
from plots import plot_correct_classifications, scatter
from data_gen import generate_data
"""
# Find optimal parameters for linear decision boundary and add noise
"""

### Firstly, generate for dataset:
'''
# We use the transpose of the (scaled to unit square) Moons dataset in order to see a non-linear decision boundary
'''
data_vertical, true_labels = generate_data('random_vertical_boundary', num_points = 200)


### Next, generate correct classification parameters for dataset (perfect classification):
'''
# Define parameters of model. Start with DenseAngle encoding with fixed parameters.
'''

qc_name = '1q-qvm'
qc = get_qc(qc_name)
num_shots = 300
qubits = qc.qubits()
init_params = np.random.rand(3)

encoding_choice = 'denseangle_param'
init_encoding_params = [np.pi, 2*np.pi]
optimiser = 'Powell' 

### Train model, and check classification result of ideal parameters found
'''
# Train model using scipy.optimize
'''
# params, result_unitary_param = train_classifier(qc, num_shots, init_params, encoding_choice, init_encoding_params, optimiser, data_vertical, true_labels)
# print('The optimised parameters are:', result_unitary_param.x)
# print('These give a cost of:', ClassificationCircuit(qubits, data_vertical).build_classifier(result_unitary_param.x, encoding_choice, init_encoding_params, num_shots, qc, true_labels))

### Define Ideal parameters for trained model learned from previous. Simple model can acheieve classification of about 90 %
'''
# 100% Classification parameters (modulo points on the boundary)
'''
ideal_params_vertical =  [3.82087234, 1.52522519, 0.08084322]


# print('These give a cost of:', ClassificationCircuit(qubits, data_vertical).build_classifier(ideal_params_vertical, encoding_choice, init_encoding_params, num_shots, qc, true_labels))
# predicted_labels = ClassificationCircuit(qubits, data_vertical).make_predictions(ideal_params_vertical, encoding_choice, init_encoding_params, num_shots, qc)
# plot_correct_classifications(true_labels, predicted_labels, data_vertical)
# nisqai.visual.scatter(data_vertical, true_labels, predicted_labels)


### Overlay decision bounday
'''
# Generate Grid of datapoints to determine and visualise ideal decision boundary
'''
data_choice = 'full_vertical_boundary'
num_points = 300
data_grid, grid_true_labels = generate_data(data_choice, num_points)

ideal_params_vertical = [3.82087234, 1.52522519, 0.08084322]

# ideal_encoding_params = [2.23855329, 7.57781576]

predicted_labels = ClassificationCircuit(qubits, data_vertical).make_predictions(ideal_params_vertical, encoding_choice, init_encoding_params, num_shots, qc)
scatter(data_vertical, predicted_labels)

predicted_labels_grid = ClassificationCircuit(qubits, data_grid).make_predictions(ideal_params_vertical, encoding_choice, init_encoding_params, num_shots, qc)
plot_params = {'colors': ['red', 'green']}
scatter(data_grid, predicted_labels_grid, **plot_params)
plt.show()


## Define noise parameters
'''
# Define noise parameters to add to model to determine how classification is affected.
'''

# noise  =None
# noise_values =None

noise  ='amp_damp_before_measurement'
noise_values = 0.4

### Add noise to circuit and classify
'''
# Add noise to circuit, and determine number of points classified differently (not mis-classified since we can't achieve perfect classification)
'''
# noisy_predictions, number_misclassified = generate_noisy_classification(ideal_params_vertical, noise, noise_values, encoding_choice, init_encoding_params, qc, num_shots, data_vertical, predicted_labels)
# print('The proportion classified differently after noise is:', number_misclassified)

## Overlay decision boundary
'''
# Generate Grid of datapoints to determine and visualise ideal decision boundary WITH noise added
'''


predicted_labels = ClassificationCircuit(qubits, data_vertical, noise, noise_values).make_predictions(ideal_params_vertical, encoding_choice, init_encoding_params, num_shots, qc)
scatter(data_vertical, predicted_labels)

predicted_labels_grid = ClassificationCircuit(qubits, data_grid, noise, noise_values).make_predictions(ideal_params_vertical, encoding_choice, init_encoding_params, num_shots, qc)

plot_params = {'colors': ['green', 'red']}
scatter(data_grid, predicted_labels_grid, **plot_params)
plt.show()

### Retrain circuit with noise
'''
# Given the noise in the circuit, train the parameters of encoding unitary to account for noise. Parameterised unitary parameters are fixed as the ideal ones learned.
'''

# encoding_params, result_encoding_param = train_classifier_encoding(qc, noise, noise_values, num_shots, ideal_params_vertical, encoding_choice, init_encoding_params, optimiser, data_vertical, true_labels)
# print('The optimised encoding parameters are:', result_encoding_param.x)
# encoding_params = [2.23855329, 7.57781576]

# ideal_encoding_params= [3.23177862, 7.21499604]

# print('These give a cost with the noisy circuit of:',\
#      ClassificationCircuit(qubits, data_vertical, noise, noise_values).build_classifier(ideal_params_vertical, encoding_choice,ideal_encoding_params , num_shots, qc, true_labels)
#      )

### Add noise to circuit and classify
'''
# Using learned encoding parameters, check again proportion misclassified
'''
# noisy_predictions, number_misclassified = generate_noisy_classification(ideal_params_vertical, noise, noise_values, encoding_choice, init_encoding_params, qc, num_shots, data_vertical, true_labels)
# print('The proportion classified differently after noise is:', number_misclassified)


### Retrain circuit with noise
'''
# Define function to compute points which will remian correctly classified after noise is added
'''

# def correct_function(data_point, params, encoding_choice, encoding_params):
#     [alpha_1, alpha_2, alpha_3] = params

#     [x_1, x_2] = data_point

#     [theta, phi] = encoding_params
#     print(theta, phi)
#     function = np.cos(alpha_3)**2 - (np.sin(2 * alpha_3) * np.sin(theta * x_1) * np.cos(theta * x_1) * np.exp(-1j*(2 * alpha_2 - phi * x_2))).real
#     print(np.cos(alpha_3)**2,  (1/2 * np.sin(2 * alpha_3) * np.sin(2 * theta * x_1) * np.exp(-1j*(2 * alpha_2 - phi * x_2)).real))
#     return function

# data_grid, grid_true_labels = generate_data(data_choice, num_points)

# data_point = data_grid[0]
# params = ideal_params_vertical 
# encoding_params = init_encoding_params
# encoding_choice = 'DenseAngle'

# function = correct_function(data_point, params, encoding_choice, encoding_params)

# def compute_misclassifed_condition(data, params, encoding_choice, encoding_params, noise_strength, true_labels):
#     correct_classification_labels = []
#     # params[2]= 1
#     for ii, data_point in enumerate(data):
    
#         function = correct_function(data_point, params, encoding_choice, encoding_params)
#         print(function, 1/(2*(1-noise_strength)), true_labels[ii])
#         print(data_point)
#         if true_labels[ii] == 0:
#             correct_classification_labels.append(0) # If datapoint was zero originally, it will be correctly classified regardless of noise
#         else:
#             if function > 1/(2*(1-noise_strength)) : # If datapoint satisfies theoretical bound, classified correctly
#                 correct_classification_labels.append(0)
#             else: correct_classification_labels.append(1)
#         print(correct_classification_labels)
#     return np.array(correct_classification_labels)

# correct_classification_labels = compute_misclassifed_condition(data_grid, params, encoding_choice, encoding_params, 0.4, grid_true_labels)

# scatter(data_grid, correct_classification_labels)
# plt.show()














'''
# IDEAL PARAMETERS FOR DATASETS
'''
# ideal_params_diagonal = [0.8268539878512121, -0.019890153474649244, 0.762840633753562]
# ideal_params = [1.552728382792277, 2.1669057463233097, -0.013736721729997667] #Vertical?

'''
NOISE PARAMETERS
'''
noise = "Decoherence_symmetric_ro"
# T1 = 30e-6
# T2 = 30e-6
# gate_time_1q = 50e-9
# gate_time_2q = 150e-09
# ro_fidelity = 0.95

# T1 = np.inf
# T2 = np.inf
# gate_time_1q = 0
# gate_time_2q = 0
# ro_fidelity = 0.95
# noise_values = [T1, T2, gate_time_1q, gate_time_2q, ro_fidelity]
# noise  ='measurement'
# noise_values = [0.95, 0.95]