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
from data_gen import generate_data, remove_zeros


"""
# Test/Learn optimal encodings in presence of certain noise models
"""
    

def main(train=False, encoding_choice='denseangle_param', retrain=False, data_choice='moons', noise=False):
    
    ### Firstly, generate for dataset:
    '''
    # We use the transpose of the (scaled to unit square) Moons dataset in order to see a non-linear decision boundary
    '''
    data_train, data_test, true_labels_train, true_labels_test = generate_data(data_choice, num_points=500, split=True)

    # data_train, true_labels_train   = remove_zeros(data_train, true_labels_train)
    # data_test, true_labels_test     = remove_zeros(data_test, true_labels_test)

    ### Next, generate correct classification parameters for dataset (perfect classification):
    '''
    # Define parameters of model. Start with DenseAngle encoding with fixed parameters.
    '''

    qc_name = '1q-qvm'
    qc = get_qc(qc_name)
    num_shots = 1024
    qubits = qc.qubits()
    init_params = np.random.rand(3)
    if encoding_choice.lower() == 'wavefunction_param':
        init_encoding_params = [ 0 ] # Generalized Wavefunction Encoding initialised to Wavefunction encoding 
    else: 
        init_encoding_params = [np.pi, 2*np.pi]

    if train:
         
        optimiser = 'Powell' 
        params, result_unitary_param = train_classifier(qc, num_shots, init_params, encoding_choice, init_encoding_params, optimiser, data_train, true_labels_train)

        print('The optimised parameters are:', result_unitary_param.x)
        print('These give a cost of:', ClassificationCircuit(qubits, data_train).build_classifier(result_unitary_param.x, encoding_choice, init_encoding_params, num_shots, qc, true_labels_train))
        ideal_params =  result_unitary_param.x
    else:
        if data_choice.lower() == 'moons':
            ### Define Ideal parameters for trained model. Simple model can acheieve classification of about 90 %
            '''
            # 90% Classification parameters for dense angle encoding
            '''
            if encoding_choice.lower() == 'denseangle_param': ideal_params= [ 2.19342064 , 1.32972029, -0.18308298]

            ### Define Ideal parameters for trained model. Simple model can acheieve classification of about 75 %
            '''
            # 73% Classification parameters for superdense angle encoding
            '''
            if encoding_choice.lower() == 'superdenseangle_param': ideal_params =  [-0.27365492,  0.83278854,  3.00092961]

            ### Define Ideal parameters for trained model. Simple model can acheieve classification of about  %
            '''
            # 85% Classification parameters for wavefunction encoding
            '''
            if encoding_choice.lower() == 'wavefunction': ideal_params = [0.81647273, 0.41996708, 2.20603541]
            if encoding_choice.lower() == 'wavefunction_param': ideal_params = [0.81647273, 0.41996708, 2.20603541]
        
        elif data_choice.lower() == 'random_vertical_boundary':
            if encoding_choice.lower() == 'superdenseangle_param': ideal_params = [1.606422245361118, 0.23401504261014927, 5.694226283697996]
        
        elif data_choice.lower() == 'random_diagonal_boundary':

            ### Define Ideal parameters for trained model. Simple model can acheieve classification of about 90 %
            '''
            # 90% Classification parameters for dense angle encoding
            '''
            if encoding_choice.lower() == 'denseangle_param':   ideal_params = [0.8579214,  1.22952647, 4.99408074]
            
            ### Define Ideal parameters for trained model. Simple model can acheieve classification of about  %
            '''
            # % Classification parameters for superdense angle encoding
            '''
            if encoding_choice.lower() == 'superdenseangle_param': ideal_params = [2.0101407,  1.05916291, 1.14570489]
            
            ### Define Ideal parameters for trained model. Simple model can acheieve classification of about  97%
            '''
            # 97% Classification parameters for wavefunction encoding
            '''
            if encoding_choice.lower() == 'wavefunction':           ideal_params = [0.69409285 0.0862859  0.42872711]
            if encoding_choice.lower() == 'wavefunction_param':     ideal_params =  [0.69409285 0.0862859  0.42872711]

        
    print('These give a cost of:', ClassificationCircuit(qubits, data_test).build_classifier(ideal_params, encoding_choice, init_encoding_params, num_shots, qc, true_labels_test))
    predicted_labels_ideal = ClassificationCircuit(qubits, data_test).make_predictions(ideal_params, encoding_choice, init_encoding_params, num_shots, qc)

    # nisqai.visual.scatter(data_test, true_labels_test, predicted_labels)


    ### Overlay decision bounday
    '''
    # Generate Grid of datapoints to determine and visualise ideal decision boundary
    '''
    num_points = 400
    data_grid, grid_true_labels = generate_data('full_vertical_boundary', num_points)
    data_grid, grid_true_labels = remove_zeros(data_grid, grid_true_labels)

    predicted_labels = ClassificationCircuit(qubits, data_test).make_predictions(ideal_params, encoding_choice, init_encoding_params, num_shots, qc)
    plot_params = {'colors': ['blue', 'orange'], 'alpha': 1}
    scatter(data_test, true_labels_test, predicted_labels, **plot_params)

    predicted_labels_grid = ClassificationCircuit(qubits, data_grid).make_predictions(ideal_params, encoding_choice, init_encoding_params, num_shots, qc)

    plot_params = {'colors': ['red', 'green'], 'alpha': 0.2}

    scatter(data_grid, predicted_labels_grid, **plot_params)
    plt.show()


    ## Define noise parameters
    '''
    # Define noise parameters to add to model to determine how classification is affected.
    '''
    if noise: 
        noise_choice = 'amp_damp_before_measurement'
        noise_values = 0.3

    ### Add noise to circuit and classify
    '''
    # Add noise to circuit, and determine number of points classified differently (not mis-classified since we can't achieve perfect classification)
    '''
    if noise:
        noisy_predictions, number_classified_same = generate_noisy_classification(ideal_params, noise_choice, noise_values, encoding_choice, init_encoding_params, qc, num_shots, data_test, predicted_labels_ideal)
        print('The proportion classified differently after noise is:', 1- number_classified_same)

    ## Overlay decision boundary
    '''
    # Generate Grid of datapoints to determine and visualise ideal decision boundary WITH noise added
    '''

    if noise:
        print(noise_choice)
        predicted_labels = ClassificationCircuit(qubits, data_test, noise_choice, noise_values).make_predictions(ideal_params, encoding_choice, init_encoding_params, num_shots, qc)
        plot_params = {'colors': ['blue', 'orange'], 'alpha': 1}
        scatter(data_test, true_labels_test, predicted_labels, **plot_params)

        predicted_labels_grid = ClassificationCircuit(qubits, data_grid, noise_choice, noise_values).make_predictions(ideal_params, encoding_choice, init_encoding_params, num_shots, qc)
        plot_params = {'colors': ['red', 'green'], 'alpha': 0.2}
        scatter(data_grid, predicted_labels_grid, **plot_params)

        plt.show()

    ### Retrain circuit with noise
    '''
    # Given the noise in the circuit, train the parameters of encoding unitary to account for noise. Parameterised unitary parameters are fixed as the ideal ones learned.
    '''
    if retrain:

        if encoding_choice.lower() == 'wavefunction_param': optimiser = 'L-BFGS-B' 
        else:                                               optimiser = 'Powell' 

        if noise:
            encoding_params, result_encoding_param = train_classifier_encoding(qc, noise_choice, noise_values, num_shots, ideal_params, encoding_choice, init_encoding_params, optimiser, data_train, true_labels_train)
            print('The optimised encoding parameters with noise are:', result_encoding_param.x)
            ideal_encoding_params = result_encoding_param.x
        else:
            encoding_params, result_encoding_param = train_classifier_encoding(qc, None, None, num_shots, ideal_params, encoding_choice, init_encoding_params, optimiser, data_train, true_labels_train)
            print('The optimised encoding parameters without noise are:', result_encoding_param.x)
            ideal_encoding_params = result_encoding_param.x
    else:
        ### Define Ideal ENCODING parameters for trained model. Simple model can acheieve classification of about 90 with noise, 93% without noise %
        '''
        # 90% Classification parameters for dense angle encoding
        '''
        if data_choice.lower() == 'moons' and encoding_choice.lower() == 'denseangle_param' and noise:  
            ideal_encoding_params = [2.23855329, 7.57781576]
            '''
            # 93% Classification parameters for dense angle encoding without noise
            '''
        elif data_choice.lower() == 'moons' and encoding_choice.lower() == 'denseangle_param':
            ideal_encoding_params =  [3.05615259, 7.61215138]  # No noise

        ### Define Ideal ENCODING parameters for trained model. Simple model can acheieve classification of about 90 %
        '''
        # NO NOISE  - 74-77% Classification parameters with training for superdense angle encoding  
        # NOISE     - Classification parameters for superdense angle encoding (0.3 amp damp = 20% different classification - 69% accuracy with noise before encoding training)
        #             With learned encoding - 
        '''
        if data_choice.lower() == 'moons' and encoding_choice.lower() == 'superdenseangle_param' and noise: 
            ideal_encoding_params =  [3.31296568, 6.34142188]

        elif data_choice.lower() == 'moons' and encoding_choice.lower() == 'superdenseangle_param':
            ideal_encoding_params = [2.86603822, 6.14328274] # No noise
        
        ### Define Ideal ENCODING parameters for trained model. Simple model can acheieve classification of about 90 %
        '''
        # NO NOISE  - 82-84% Classification parameters with training for generalised wavefunction encoding  
        # NOISE     - Classification parameters for superdense angle encoding (0.3 amp damp = 20% different classification - 78% accuracy with noise before encoding training)
        #             With learned encoding - 
        '''
        print(data_choice.lower(), encoding_choice.lower())
        if data_choice.lower() == 'moons' and encoding_choice.lower() == 'wavefunction_param' and noise: 
            ideal_encoding_params =  [0.02884417]
        elif data_choice.lower() == 'moons' and encoding_choice.lower() == 'wavefunction_param':
            ideal_encoding_params = [0.01582773] # No noise
            

    if noise:
        print('These give a cost with the noisy circuit of:',\
            ClassificationCircuit(qubits, data_test, noise_choice, noise_values).build_classifier(ideal_params, encoding_choice, ideal_encoding_params , num_shots, qc, true_labels_test) )
    else:       
        print('These give a cost with the ideal circuit of:',\
            ClassificationCircuit(qubits, data_test).build_classifier(ideal_params, encoding_choice, ideal_encoding_params , num_shots, qc, true_labels_test) )

    ### Add noise to circuit and classify
    '''
    # Using learned encoding parameters, check again proportion misclassified
    '''
    if noise:
        noisy_predictions, number_classified_same = generate_noisy_classification(ideal_params, noise_choice, noise_values, encoding_choice, ideal_encoding_params, qc, num_shots, data_test, predicted_labels)
        print('The proportion classified differently after noise with learned encoding is:', 1 - number_classified_same)

    ## Overlay decision boundary
    '''
    # Generate Grid of datapoints to determine and visualise ideal decision boundary WITH/WITHOUT noise added
    '''
    if noise:
        predicted_labels = ClassificationCircuit(qubits, data_test, noise_choice, noise_values).make_predictions(ideal_params, encoding_choice, ideal_encoding_params, num_shots, qc)

        plot_params = {'colors': ['blue', 'orange'], 'alpha': 1}
        scatter(data_test,  true_labels_test, predicted_labels, **plot_params)

        predicted_labels_grid = ClassificationCircuit(qubits, data_grid, noise_choice, noise_values).make_predictions(ideal_params, encoding_choice, ideal_encoding_params, num_shots, qc)
        
        plot_params = {'colors': ['red', 'green'], 'alpha': 0.2}
        scatter(data_grid, predicted_labels_grid, **plot_params)
        plt.show()
    else:
        predicted_labels = ClassificationCircuit(qubits, data_test).make_predictions(ideal_params, encoding_choice, ideal_encoding_params, num_shots, qc)
        
        plot_params = {'colors': ['blue', 'orange'], 'alpha': 1}
        scatter(data_test, true_labels_test,  predicted_labels, **plot_params)
        
        predicted_labels_grid = ClassificationCircuit(qubits, data_grid).make_predictions(ideal_params, encoding_choice, ideal_encoding_params, num_shots, qc)
        
        plot_params = {'colors': ['red', 'green'], 'alpha': 0.2}
        scatter(data_grid, predicted_labels_grid, **plot_params)
        plt.show()


if __name__ == "__main__":
    main(train=False, encoding_choice='superdense_angle', retrain=False, data_choice='random_diagonal_boundary', noise=True)

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