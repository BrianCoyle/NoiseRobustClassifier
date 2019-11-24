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
from plots import plot_correct_classifications, scatter, scatter_red
from data_gen import generate_data

"""
# Test/Learn optimal encodings in presence of certain noise models
"""

### Firstly, generate for dataset:
'''
# We use the transpose of the (scaled to unit square) Moons dataset in order to see a non-linear decision boundary
'''
unscaled_data, true_labels = make_moons(n_samples=200, noise=0.05)
data_moons = np.zeros_like(unscaled_data)

for ii, point in enumerate(unscaled_data):
    data_moons[ii] = np.array([point[0]/3+1/3, 2*point[1]/3 + 1/3])

data_moons[:,[0, 1]] = data_moons[:,[1, 0]]


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
# init_params_encoding_and_unitary = np.random.rand(5) # all parameters to be trained
# params, result_unitary_param = train_classifier(qc, num_shots, init_params, encoding_choice, init_encoding_params, optimiser, data_moons, true_labels)

# print('The optimised parameters are:', result_unitary_param.x)
# print('These give a cost of:', ClassificationCircuit(qubits, data_moons).build_classifier(result_unitary_param.x, encoding_choice, init_encoding_params, num_shots, qc, true_labels))

### Define Ideal parameters for trained model. Simple model can acheieve classification of about 90 %
'''
# 90% Classification parameters
'''
# ideal_params_moons = [1.91798163, 0.83910137, 0.00612348]
ideal_params_moons = [ 2.19342064 , 1.32972029, -0.18308298]

# print('These give a cost of:', ClassificationCircuit(qubits, data_moons).build_classifier(ideal_params_diagonal, encoding_choice, init_encoding_params, num_shots, qc, true_labels))
# predicted_labels = ClassificationCircuit(qubits, data_moons).make_predictions(ideal_params_diagonal, encoding_choice, init_encoding_params, num_shots, qc)
# plot_correct_classifications(true_labels, predicted_labels, data_moons)
# nisqai.visual.scatter(data_moons, true_labels, predicted_labels)


### Overlay decision bounday
'''
# Generate Grid of datapoints to determine and visualise ideal decision boundary
'''
data_choice = 'full_vertical_boundary'
num_points = 300
data_grid, grid_true_labels = generate_data(data_choice, num_points)

# # ideal_params = [point_1, point_2, point_3]
# ideal_params_moons = [1.91798163, 0.83910137, 0.00612348]
ideal_params_moons = [ 2.19342064,  1.32972029, -0.18308298]

# ideal_encoding_params = [2.23855329, 7.57781576]
# ideal_params_moons =  result_unitary_param.x[0:3]
# ideal_encoding_params = result_unitary_param.x[4:5]

predicted_labels = ClassificationCircuit(qubits, data_moons).make_predictions(ideal_params_moons, encoding_choice, init_encoding_params, num_shots, qc)
scatter(data_moons, predicted_labels)

predicted_labels_grid = ClassificationCircuit(qubits, data_grid).make_predictions(ideal_params_moons, encoding_choice, init_encoding_params, num_shots, qc)
scatter_red(data_grid, predicted_labels_grid)
plt.show()


## Define noise parameters
'''
# Define noise parameters to add to model to determine how classification is affected.
'''

# noise  =None
# noise_values =None

noise  ='amp_damp_before_measurement'
noise_values = 0.3

### Add noise to circuit and classify
'''
# Add noise to circuit, and determine number of points classified differently (not mis-classified since we can't achieve perfect classification)
'''
noisy_predictions, number_misclassified = generate_noisy_classification(ideal_params_moons, noise, noise_values, encoding_choice, init_encoding_params, qc, num_shots, data_moons, predicted_labels)
print('The proportion classified differently after noise is:', number_misclassified)

## Overlay decision boundary
'''
# Generate Grid of datapoints to determine and visualise ideal decision boundary WITH noise added
'''

# ideal_params_moons = [1.91798163, 0.83910137, 0.00612348] # Same ideal parameters

# predicted_labels = ClassificationCircuit(qubits, data_moons, noise, noise_values).make_predictions(ideal_params_moons, encoding_choice, init_encoding_params, num_shots, qc)
# scatter(data_moons, predicted_labels)

# predicted_labels_grid = ClassificationCircuit(qubits, data_grid, noise, noise_values).make_predictions(ideal_params_moons, encoding_choice, init_encoding_params, num_shots, qc)
# scatter_red(data_grid, predicted_labels_grid)
# plt.show()

### Retrain circuit with noise
'''
# Given the noise in the circuit, train the parameters of encoding unitary to account for noise. Parameterised unitary parameters are fixed as the ideal ones learned.
'''

# encoding_params, result_encoding_param = train_classifier_encoding(qc, noise, noise_values, num_shots, ideal_params_moons, encoding_choice, init_encoding_params, optimiser, data_moons, true_labels)
# print('The optimised encoding parameters are:', result_encoding_param.x)
# encoding_params = [2.23855329, 7.57781576]

ideal_encoding_params= [3.23177862, 7.21499604]

print('These give a cost with the noisy circuit of:',\
     ClassificationCircuit(qubits, data_moons, noise, noise_values).build_classifier(ideal_params_moons, encoding_choice,ideal_encoding_params , num_shots, qc, true_labels)
     )

### Add noise to circuit and classify
'''
# Using learned encoding parameters, check again proportion misclassified
'''
noisy_predictions, number_misclassified = generate_noisy_classification(ideal_params_moons, noise, noise_values, encoding_choice, init_encoding_params, qc, num_shots, data_moons, true_labels)
print('The proportion classified differently after noise is:', number_misclassified)

## Overlay decision boundary
'''
# Generate Grid of datapoints to determine and visualise ideal decision boundary WITH noise added
'''

# ideal_params_moons = [1.91798163, 0.83910137, 0.00612348] # Same ideal parameters
ideal_params_moons = [ 2.19342064,  1.32972029, -0.18308298]

# ideal_encoding_params = [2.23855329, 7.57781576]
ideal_encoding_params= [3.23177862, 7.21499604]

predicted_labels = ClassificationCircuit(qubits, data_moons, noise, noise_values).make_predictions(ideal_params_moons, encoding_choice, ideal_encoding_params, num_shots, qc)
scatter(data_moons, predicted_labels)

predicted_labels_grid = ClassificationCircuit(qubits, data_grid, noise, noise_values).make_predictions(ideal_params_moons, encoding_choice, ideal_encoding_params, num_shots, qc)
scatter_red(data_grid, predicted_labels_grid)
plt.show()

# ClassificationCircuit(qubits, data).build_classifier(result.x, encoding_choice, encoding_params,num_shots, qc, true_labels)
# predicted_labels = ClassificationCircuit(qubits, data).make_predictions(ideal_params, encoding_choice, encoding_params, num_shots, qc, true_labels)
# print(true_labels, predicted_labels)

# plot_correct_classifications(true_labels, predicted_labels, data)

# noise_values = [T1, T2, gate_time_1q, gate_time_2q, ro_fidelity]
# noise  ='measurement'
# noise_values = [0.95, 0.95]

# noisy_predictions, number_misclassified = generate_noisy_classification(ideal_params, noise, noise_values, encoding_choice, encoding_params, qc, num_shots, data, true_labels)
# plot_correct_classifications(true_labels, noisy_predictions, data)
# print(noisy_predictions, true_labels)
# print(noisy_predictions, number_misclassified)
# ideal_wf = make_wf.wavefunction(ideal_circuit_circ.circuit)
# print('Ideal WF is:', ideal_wf)
# ideal_predictions = circ._make_predictions(num_shots, qc)
# print(ideal_predictions, 'TRUE:\n',true_labels)
# plot_correct_classifications(true_labels, ideal_predictions, data)


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