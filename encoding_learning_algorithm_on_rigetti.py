from single_qubit_classifier import generate_noisy_classification
import matplotlib.pyplot as plt
import numpy as np

from math import isclose

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

from noise_models import add_noisy_gate_to_circ
from classifier_circuits import *
from data_gen import *
from single_qubit_classifier import train_classifier, train_classifier_encoding, ClassificationCircuit
from plots import plot_correct_classifications, scatter, plot_encoding_algo
from data_gen import generate_data, remove_zeros
from plots import scatter
from file_operations_out import make_dir

from DeviceCharacterisers.rigetti_devices import get_device_noise_params


def main(train=False, retrain=False, data_choice='moons', qc_name='1q-qvm', simulated=True):
    
    ### Firstly, generate for dataset:

    data_train, data_test, true_labels_train, true_labels_test = generate_data(data_choice, num_points=500, split=True)

    data_train, true_labels_train   = remove_zeros(data_train, true_labels_train)
    data_test, true_labels_test     = remove_zeros(data_test, true_labels_test)
    encodings = [ 'superdenseangle_param' ]

    # encodings = [ 'denseangle_param','superdenseangle_param', 'wavefunction_param' ]

    minimal_costs, ideal_costs, noisy_costs, noisy_costs_uncorrected = [np.ones(len(encodings)) for _ in range(4)]

    qc = get_qc(qc_name, as_qvm=simulated)
    num_shots = 512
    device_qubits = qc.qubits()

    classifier_qubits = [device_qubits[0]]
    n_layers = 1
    init_params = np.random.rand(1,1,3)
    ideal_params = []
    ideal_encoding_params = []
    init_encoding_params = []

    
    for ii, encoding_choice in enumerate(encodings):

        print('\n**********************************')
        print('\nThe encoding is:', encoding_choice)
        print('\n**********************************')

        if encoding_choice.lower() == 'wavefunction_param':
            init_encoding_params.append([ 0 ]) # Generalized Wavefunction Encoding initialised to Wavefunction encoding 
        else: 
            init_encoding_params.append([np.pi, 2*np.pi])

        if train:

            optimiser = 'Powell' 
            params, result_unitary_param = train_classifier(qc, num_shots, init_params, encoding_choice, init_encoding_params[ii], optimiser, data_train, true_labels_train)

            print('The optimised parameters are:', result_unitary_param.x)
            print('These give a cost of:', ClassificationCircuit(classifier_qubits, data_train).build_classifier(result_unitary_param.x, n_layers, encoding_choice, init_encoding_params[ii], num_shots, qc, true_labels_train))
            ideal_params.append(result_unitary_param.x)
        else:

            if data_choice.lower() == 'moons':
                if      encoding_choice.lower() == 'denseangle_param':              ideal_params.append([ 2.19342064 , 1.32972029, -0.18308298])
                elif    encoding_choice.lower() == 'superdenseangle_param':         ideal_params.append([-0.27365492,  0.83278854,  3.00092961])
                elif    encoding_choice.lower()  == 'wavefunction':                 ideal_params.append([0.81647273, 0.41996708, 2.20603541])
                elif    encoding_choice.lower() == 'wavefunction_param':            ideal_params.append([0.81647273, 0.41996708, 2.20603541])

            elif data_choice.lower() == 'random_vertical_boundary':
                if      encoding_choice.lower() == 'denseangle_param':              ideal_params.append([1.67814786, 1.56516469, 1.77820848])
                elif    encoding_choice.lower() == 'superdenseangle_param':         ideal_params.append([1.60642225, 0.23401504, 5.69422628])
                elif    encoding_choice.lower()  == 'wavefunction':                 ideal_params.append([0.96291484, 0.18133714, 0.35436732])
                elif    encoding_choice.lower() == 'wavefunction_param':            ideal_params.append([0.96291484, 0.18133714, 0.35436732])
            
            elif data_choice.lower() == 'random_diagonal_boundary':
                if      encoding_choice.lower() == 'denseangle_param':              ideal_params.append([0.8579214,  1.22952647, 4.99408074])
                elif    encoding_choice.lower() == 'superdenseangle_param':         ideal_params.append([2.0101407,  1.05916291, 1.14570489])
                elif    encoding_choice.lower()  == 'wavefunction':                 ideal_params.append([0.69409285, 0.0862859,  0.42872711])
                elif    encoding_choice.lower() == 'wavefunction_param':            ideal_params.append([0.69409285, 0.0862859,  0.42872711])
            
            else:
                print('THIS DATASET HAS NOT BEEN TRAINED FOR')
                if      encoding_choice.lower() == 'denseangle_param':              ideal_params.append(init_params)
                elif    encoding_choice.lower() == 'superdenseangle_param':         ideal_params.append(init_params)
                elif    encoding_choice.lower() == 'wavefunction':                  ideal_params.append(init_params)
                elif    encoding_choice.lower() == 'wavefunction_param':            ideal_params.append(init_params)

        # ideal_costs[ii] = ClassificationCircuit(classifier_qubits, data_test).build_classifier(ideal_params[ii],  n_layers,\
        #                                                                                     encoding_choice, init_encoding_params[ii],\
        #                                                                                     num_shots, qc, true_labels_test)
        
        # print('In the ideal case, the cost is:', ideal_costs[ii])
        predicted_labels_ideal = ClassificationCircuit(classifier_qubits, data_test).make_predictions(ideal_params[ii], n_layers,  encoding_choice, init_encoding_params[ii], num_shots, qc)
        
        noise_choice = 'decoherence_symmetric_ro'
        if simulated:
            noise_values = get_device_noise_params(qc_name+'-qvm')
        else:
            noise_values = get_device_noise_params(qc_name)

        # noisy_costs_uncorrected[ii] = ClassificationCircuit(classifier_qubits, data_test, noise_choice, noise_values).build_classifier(ideal_params[ii],  n_layers, \
        #                                                                                                                                 encoding_choice, init_encoding_params[ii],\
        #                                                                                                                                 num_shots, qc, true_labels_test) 
        # print('\nWithout encoding training, the noisy cost is:', noisy_costs_uncorrected[ii])

        # noisy_predictions, number_classified_same = generate_noisy_classification(ideal_params[ii], n_layers,  noise_choice, noise_values, \
        #                                                                             encoding_choice, init_encoding_params[ii],\
        #                                                                             qc,  classifier_qubits, num_shots, data_test, predicted_labels_ideal)

        # print('The proportion classified differently after noise is:', 1 - number_classified_same)

        # data_choice = 'full_vertical_boundary'

        # num__grid_points = 500
        # data_grid, grid_true_labels = generate_data(data_choice, num__grid_points)
        # data_grid, grid_true_labels = remove_zeros(data_grid, grid_true_labels)

        # ### IDEAL
        # predicted_labels_test = ClassificationCircuit(classifier_qubits, data_test).make_predictions(ideal_params[ii], n_layers, encoding_choice, init_encoding_params[ii], num_shots, qc)
        # plot_params = {'colors': ['blue', 'orange'], 'alpha': 1}
        # scatter(data_test, true_labels_test, predicted_labels_test, **plot_params)

        # predicted_labels_grid = ClassificationCircuit(classifier_qubits, data_grid).make_predictions(ideal_params[ii], n_layers, encoding_choice, init_encoding_params[ii], num_shots, qc)
        # plot_params = {'colors': ['red', 'green'], 'alpha': 0.2}
        # scatter(data_grid, predicted_labels_grid, **plot_params)
        # plt.show()

        #  ## Overlay decision boundary
        # '''
        # # Generate Grid of datapoints to determine and visualise ideal decision boundary WITH noise added
        # '''
        # predicted_labels_test_noise = ClassificationCircuit(classifier_qubits, data_test, noise_choice, noise_values).make_predictions(ideal_params[ii], n_layers, encoding_choice, init_encoding_params[ii], num_shots, qc)
        # plot_params = {'colors': ['blue', 'orange'], 'alpha': 1}
        # scatter(data_test, true_labels_test, predicted_labels_test_noise, **plot_params)

        # predicted_labels_grid_noise = ClassificationCircuit(classifier_qubits, data_grid, noise_choice, noise_values).make_predictions(ideal_params[ii], n_layers, encoding_choice, init_encoding_params[ii], num_shots, qc)

        # plot_params = {'colors': ['red', 'green'], 'alpha': 0.2}
        # scatter(data_grid, predicted_labels_grid_noise, **plot_params)
        # plt.show()

        if retrain:
            if encoding_choice.lower() == 'wavefunction_param': optimiser = 'L-BFGS-B' # Need a solver with bounds for generalized wf encoding
            else:                                               optimiser = 'Powell' 

            encoding_params, result_encoding_param = train_classifier_encoding(qc, classifier_qubits, noise_choice, noise_values, num_shots,\
                                                                                ideal_params[ii], encoding_choice, init_encoding_params[ii], optimiser,\
                                                                                data_train, true_labels_train)

            print('The optimised encoding parameters with noise are:', result_encoding_param.x)
            ideal_encoding_params.append(result_encoding_param.x)
  
        else:
            if qc_name.lower() == 'aspen-4-2q-a-qvm': # Trained encoding params for this chip with simulated noise model
                if data_choice.lower() == 'moons':
                    if      encoding_choice.lower() == 'denseangle_param': ideal_encoding_params.append([])
                    elif    encoding_choice.lower() == 'superdenseangle_param': ideal_encoding_params.append([])
                    elif    encoding_choice.lower() == 'wavefunction_param': ideal_encoding_params.append([])

                elif data_choice.lower() == 'random_vertical_boundary':
                    if      encoding_choice.lower() == 'denseangle_param': ideal_encoding_params.append([np.pi, 2*np.pi])
                    elif    encoding_choice.lower() == 'superdenseangle_param': ideal_encoding_params.append([])
                    elif    encoding_choice.lower() == 'wavefunction_param': ideal_encoding_params.append([])

                elif data_choice.lower() == 'random_diagonal_boundary':
                    if      encoding_choice.lower() == 'denseangle_param': ideal_encoding_params.append([])
                    elif    encoding_choice.lower() == 'superdenseangle_param': ideal_encoding_params.append([])
                    elif    encoding_choice.lower() == 'wavefunction_param': ideal_encoding_params.append([])

                else:
                    print('THIS DATASET HAS NOT BEEN TRAINED FOR')
                    if      encoding_choice.lower() == 'denseangle_param': ideal_encoding_params.append(init_encoding_params[ii])
                    elif    encoding_choice.lower() == 'superdenseangle_param': ideal_encoding_params.append(init_encoding_params[ii])
                    elif    encoding_choice.lower() == 'wavefunction_param': ideal_encoding_params.append(init_encoding_params[ii])
            
        noisy_costs[ii] = ClassificationCircuit(classifier_qubits, data_test, noise_choice, noise_values).build_classifier(ideal_params[ii], n_layers, \
                                                                                                        encoding_choice, ideal_encoding_params[ii],\
                                                                                                        num_shots, qc, true_labels_test) 

        print('\nWith encoding training, the noisy cost is:', noisy_costs[ii])

        noisy_predictions, number_classified_same = generate_noisy_classification(ideal_params[ii],  n_layers, \
                                                                                    noise_choice, noise_values, encoding_choice, ideal_encoding_params[ii],\
                                                                                    qc, classifier_qubits,  num_shots, data_test, predicted_labels_ideal)

        print('The proportion classified differently after noise with learned encoding is:', 1 - number_classified_same)

    for ii, encoding_choice in enumerate(encodings):
        print('\nThe encoding is:'                      , encodings[ii]                 ) 
        print('The ideal params are:'                   , ideal_params[ii]              )
        print('The ideal encoding params are'           , ideal_encoding_params[ii]     )

        print('The ideal cost for encoding'             , ideal_costs[ii]               )
        print('The noisy cost with untrained encoding'  , noisy_costs_uncorrected[ii]   )  
        print('The noisy cost with trained encoding'    , noisy_costs[ii]               )

    return encodings, ideal_params, init_encoding_params, ideal_encoding_params, ideal_costs, noisy_costs_uncorrected, noisy_costs

if __name__ == "__main__":

    data_choice = 'random_vertical_boundary'

    for run in range(5):

        print('\n**********************************')
        print('\nThis is Run:', run)
        print('\n**********************************')
        
        qc_name='Aspen-4-2Q-A'
        simulated = True
        
        output = main(train=False, retrain=True, data_choice=data_choice, qc_name=qc_name, simulated=simulated)
        if simulated:
            qc_name = qc_name + '-qvm'
    
        file_name = '%s/Data_choice_%s/Run_%i' % (qc_name, data_choice, run)
        make_dir(file_name)
        
        ideal_params, init_encoding_params, ideal_encoding_params, ideal_costs, noisy_costs_uncorrected, noisy_costs  = [[] for _ in range(6)]

        ideal_params.append(output[1])
        init_encoding_params.append(output[2]) 
        ideal_encoding_params.append(output[3]) 
        ideal_costs.append(output[4]) 
        noisy_costs_uncorrected.append(output[5])
        noisy_costs.append(output[6])

        encodings = output[0]
        
        np.savetxt('%s/%s' %(file_name, 'ideal_params'), ideal_params[run])
        np.savetxt('%s/%s' %(file_name, 'init_encoding_params'), init_encoding_params[run])
        np.savetxt('%s/%s' %(file_name, 'ideal_encoding_params'), ideal_encoding_params[run])
        np.savetxt('%s/%s' %(file_name, 'ideal_costs'), ideal_costs[run])
        np.savetxt('%s/%s' %(file_name, 'noisy_costs_uncorrected'), noisy_costs_uncorrected[run])
        np.savetxt('%s/%s' %(file_name, 'noisy_costs'), noisy_costs[run])


    # datasets = ['moons', 'random_vertical_boundary', 'random_diagonal_boundary']

    # ideal_params, init_encoding_params, ideal_encoding_params, ideal_costs, noisy_costs_uncorrected, noisy_costs  = [[] for _ in range(6)]

    # for data in datasets:
    #     print('\n**********************************')
    #     print('\nThe dataset is:', data)
    #     print('\n**********************************')

    #     output = main(train=False, retrain=False, data_choice=data, noise_choice='amp_damp_before_measurement', noise_values=0.3)

    #     ideal_params.append(output[1])
    #     init_encoding_params.append(output[2]) 
    #     ideal_encoding_params.append(output[3]) 
    #     ideal_costs.append(output[4]) 
    #     noisy_costs_uncorrected.append(output[5])
    #     noisy_costs.append(output[6])

    # encodings = output[0]

    # plt.rcParams.update({"font.size": 11, "font.serif":"Computer Moden Roman"})

    # plot_encoding_algo(datasets, encodings, ideal_params, init_encoding_params, ideal_encoding_params, ideal_costs, noisy_costs_uncorrected, noisy_costs)