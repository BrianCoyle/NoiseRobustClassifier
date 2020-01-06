from single_qubit_classifier import generate_noisy_classification
import matplotlib.pyplot as plt
import numpy as np
from numpy import cos, sin

from math import isclose

from scipy.linalg import sqrtm

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

"""
This file analyses the relative fidelity between the encoded states and the ideal states
over a particular dataset for several classes of noise.

This DOES NOT use the Rigetti simulator, but simulates the density matrices directly.

(1) Take ideal parameters for dataset for a particular encoding.

(2) Compute average cost over dataset based on this cost.

(3) Add noise to state and compute average state fidelity for each set of points.
"""
 
def main(dataset='random_vertical_boundary'):
    
    # Generate data
    data_train, data_test, true_labels_train, true_labels_test = generate_data(dataset, num_points = 500, split=True)

    # Pauli matrices
    imat = np.identity(2)
    xmat = np.array([[0, 1], [1, 0]])
    ymat = np.array([[0, -1j], [1j, 0]])
    zmat = np.array([[1, 0], [0, -1]])

    def rz(param):

        return np.array([[np.cos(param / 2) - 1j * np.sin(param / 2), 0],
           [0, np.cos(param / 2) + 1j * np.sin(param / 2)]])

    def ry(param):
        return np.array( [[cos(param / 2), -sin(param / 2)],
           [sin(param / 2), cos(param / 2)]])

    def unitary_evolved_noiseless(rho, params):
        '''
        rho should be an encoded state. This function returns the evolved state with the unitary params.
        '''
        unitary = rz(params[2]) @ ry(params[1]) @ rz(params[0]) 
        rho = unitary @ rho @ unitary.conj().T
        return rho

    def unitary_evolved_noisy(rho, params, noise_choice='Pauli', noise_values=None):
        '''
        rho should be an encoded state. This function returns the evolved state with the unitary params.
        The noise channel is applied after each step in the unitary
        '''
        rho = rz(params[0]) @ rho @ rz(params[0]).conj().T # First unitary

        rho = channels(rho, noise_choice=noise_choice, noise_values=noise_values) # Apply noise

        rho = ry(params[1]) @ rho @ ry(params[1]).conj().T # Second unitary

        rho = channels(rho, noise_choice=noise_choice, noise_values=noise_values) # Apply noise

        rho = rz(params[2]) @ rho @ rz(params[2]).conj().T # Third unitary

        return rho

    # Projectors
    pi0 = (imat + zmat) / 2
    pi1 = (imat - zmat) / 2


    # Encodings
    def state(f, g, x, y, encoding_params):
        rho = np.array([
                [abs(f(x, y, encoding_params))**2, f(x, y, encoding_params) * np.conj(g(x, y, encoding_params))],
                [np.conj(f(x, y, encoding_params)) * g(x, y, encoding_params), abs(g(x, y, encoding_params))**2]
                ])
        return rho / np.trace(rho)

    

    def f_dae(x, y, encoding_params):
        return np.cos(encoding_params[0] * x)


    def g_dae(x, y, encoding_params):
        return np.exp(2 * encoding_params[1] * 1j * y) * np.sin(encoding_params[0] * x)

    def f_wf(x, y, encoding_params=None):
        return x / np.sqrt( x**2 + y**2 )


    def g_wf(x, y, encoding_params=None):
        return y / np.sqrt( x**2 + y**2 )

    def f_sdae(x, y, encoding_params):
        return np.cos(encoding_params[0] * x + encoding_params[1] * y)


    def g_sdae(x, y, encoding_params):
        return np.sin(encoding_params[0] * x + encoding_params[1] * y)

    def pauli(rho, pi=0.4, px=0.1, py=0.1, pz=0.4):
        """Applies a single qubit pauli channel to the state rho."""
        assert np.isclose(sum((pi, px, py, pz)), 1.0)
        return (pi * rho + 
                px * xmat @ rho @ xmat +
                py * ymat @ rho @ ymat +
                pz * zmat @ rho @ zmat)

    def depolarizing(rho, p=0.1):
        return (1-p)* rho + (p) * imat

    def amplitude_damping(rho, p=0.1):
        E0 = np.array([ [1, 0], [0, np.sqrt(1-p)] ])
        E1 = np.array([ [0, np.sqrt( p )], [0, 0] ])

        return  E0 @ rho @ E0.conj().T  + E1 @ rho @ E1.conj().T

    # Noise channels
    def channels(rho, noise_choice='Pauli', noise_values=None):

        if noise_choice.lower() == 'pauli':
            [ pi, px , py , pz ] = noise_values
            rho = pauli(rho, pi, px, py, pz) 

        elif noise_choice.lower() == 'depolarizing':
            [ p ] = noise_values
            rho = pauli(rho, p)

        elif noise_choice.lower() == 'dephasing':
            [ pz ] = noise_values
            rho = pauli(rho, 1-pz, 0, 0, pz) 

        elif noise_choice.lower() == 'bit_flip':
            [ pz ] = noise_values
            rho = pauli(rho, 1-px, px, 0, 0) 
        
        elif noise_choice.lower() == 'amp_damp':
            [ p ] = noise_values
            rho = amplitude_damping(rho, p)

        return rho

    # Predictions
    def make_prediction(rho):
        # Compute prediction for data sample in state rho
        elt = rho[0, 0]
        if elt >= 0.5:
            return 0
        return 1


    data_choice = 'random_vertical_boundary'
    encoding_choice = 'denseangle_param' 
    ideal_params = []

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

    else: raise NotImplementedError

    fidelity = np.zeros(len(data_test)) # one element of fidelity per data point
    pred_labels_noisy = np.zeros(len(data_test), dtype=int)
    pred_labels_noiseless = np.zeros(len(data_test), dtype=int)


    def average_fidelity(params, encoding_choice='denseangle_param', encoding_params=None, noise_choice='Pauli', noise_values=None):

        params=ideal_params[0]
        for ii, point in enumerate(data_test):
                
            if encoding_choice.lower() == 'denseangle_param':
                rho_encoded = state(f_dae, g_dae, point[0], point[1], encoding_params) # encode data point
            print(point, rho_encoded)
            rho_noisy =  unitary_evolved_noisy(rho_encoded, params, noise_choice='Pauli', noise_values=noise_values)
            rho_noiseless =  unitary_evolved_noiseless(rho_encoded, params)
            print(rho_noiseless, rho_noisy)
            pred_labels_noiseless[ii] = make_prediction(rho_noiseless)
            pred_labels_noisy[ii] = make_prediction(rho_noisy)

            fidelity[ii] = ( np.trace(sqrtm(rho_noisy @ rho_noiseless)) )**2 # compute fidelity per sample
        print(fidelity)
        avg_fidelity = (1/len(data_test)) * np.sum(fidelity, axis=0) # compute average fidelity over dataset
        print('The Average Fidelity is:', avg_fidelity)
        return average_fidelity, pred_labels_noiseless, pred_labels_noisy

    params = ideal_params[0]
    encoding_params = [2 * np.pi, np.pi]
    noise_values = [0, 0.25, 0.25, 0.5]

    average_fidelity(params, encoding_choice='denseangle_param', encoding_params=encoding_params, noise_choice='Pauli', noise_values=noise_values)
   
   
    # def pauli_robustness():
    #     points = grid_pts(nx=12, ny=12)
    #     unitary = np.identity(2)
        
    #     ys = []
    #     yhats = []
        
    #     for point in points:
    #         # Get the state
    #         rho = state(f, g, *point)
            
    #         # Noiseless and noisy predictions
    #         y = noiseless(rho, unitary)
    #         yhat = noisy(rho, pauli, unitary)
            
    #         # Store them
    #         ys.append(y)
    #         yhats.append(yhat)
            
    #         if y == yhat:
    #             plt.scatter(*point, marker="o", color="green", s=50)
    #         else:
    #             plt.scatter(*point, marker="x", color="red", s=50)
            
    #     plt.show()




if __name__ == "__main__":
    main()

