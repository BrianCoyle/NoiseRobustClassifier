import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 12, "font.serif": "Computer Modern Roman"})

import numpy as np
from numpy import cos, sin

from math import isclose

from scipy.linalg import sqrtm

from pyquil.gates import *
from scipy.optimize import minimize
from collections import Counter
import random

from classifier_circuits import *
from data_gen import *
from data_gen import generate_data

from quantum_gates import ry, rz, rx, imat, xmat, zmat, ymat, pi0, pi1, hmat, cnotmat

from plots import fidelity_bound_plot, fidelity_compare_plot
"""
This file analyses the relative fidelity between the encoded states and the ideal states
over a particular dataset for several classes of noise.

This DOES NOT use the Rigetti simulator, but simulates the density matrices directly.

(1) Take ideal parameters for dataset for a particular encoding.

(2) Compute average cost over dataset based on this cost.

(3) Add noise to state and compute average state fidelity for each set of points.
"""
 
def main(data_choice='iris', amp_damp_noise=False, bit_flip_noise=False, dephasing_noise=False, global_depolarizing_noise=False, legend=False):
    
    # Generate data
    data_train, data_test, true_labels_train, true_labels_test = generate_data(data_choice, num_points = 100, split=True)
 

    def unitary_evolved_noiseless(rho, params):
        '''
        rho should be an encoded state. This function returns the evolved state with the unitary params.
        '''

        unitary_layer_1_1 = rz(2*params[2]) @ ry(2*params[1]) @ rz(2*params[0]) 
        unitary_layer_1_2 = rz(2*params[5]) @ ry(2*params[4]) @ rz(2*params[3]) 
        U_1 = np.kron( unitary_layer_1_1, unitary_layer_1_2) 
        
        # First layer
        rho = U_1 @ rho @ U_1.conj().T

        unitary_layer_2_1 = rx(2*params[6] + np.pi) @ hmat 
        unitary_layer_2_2 = rz(2*params[7]) 
        U_2 = np.kron( unitary_layer_2_1, unitary_layer_2_2) @ cnotmat 

        # Second Layer
        rho = U_2 @ rho @ U_2.conj().T

        unitary_layer_3_1 = hmat 
        unitary_layer_3_2 = rz(-2*params[8]) 
        U_3 = np.kron( unitary_layer_3_1, unitary_layer_3_2) @ cnotmat
        
        # Third Layer
        rho = U_3 @ rho @ U_3.conj().T

        unitary_layer_4_1 = rz(2*params[11]) @ ry(2*params[10]) @ rz(2*params[9]) 
        unitary_layer_4_2 = imat
        U_4 = np.kron( unitary_layer_4_1, unitary_layer_4_2) @ cnotmat

        # Fourth Layer
        rho = U_4 @ rho @ U_4.conj().T

        return rho


    def unitary_evolved_noisy(rho, params, noise_choice='Pauli', noise_values=None, noise=True):
        '''
        rho should be an encoded state. This function returns the evolved state with the unitary params.
        The noise channel is applied after each step in the unitary
        '''
        if noise:
            rho = channels(rho, noise_choice=noise_choice, noise_values=noise_values) # Apply noise

        unitary_layer_1_n = np.kron(rz(2*params[0]), rz(2*params[3])  )
        rho = unitary_layer_1_n @ rho @ unitary_layer_1_n.conj().T
        if noise:
            rho = channels(rho, noise_choice=noise_choice, noise_values=noise_values) # Apply noise

        unitary_layer_2_n = np.kron(ry(2*params[1]), ry(2*params[4])  )
        rho = unitary_layer_2_n @ rho @ unitary_layer_2_n.conj().T
        if noise:
            rho = channels(rho, noise_choice=noise_choice, noise_values=noise_values) # Apply noise

        unitary_layer_3_n = np.kron(rz(2*params[2]), rz(2*params[5])  )
        rho = unitary_layer_3_n @ rho @ unitary_layer_3_n.conj().T
        if noise:
            rho = channels(rho, noise_choice=noise_choice, noise_values=noise_values) # Apply noise

        unitary_layer_4_n = cnotmat
        rho = unitary_layer_4_n @ rho @ unitary_layer_4_n.conj().T

        if noise:
            rho = channels(rho, noise_choice=noise_choice, noise_values=noise_values) # Apply noise

        unitary_layer_2_1 = rx(2*params[6] + np.pi) @ hmat 
        unitary_layer_2_2 = rz(2*params[7]) 
        unitary_layer_5_n = np.kron( unitary_layer_2_1, unitary_layer_2_2) 
        rho = unitary_layer_5_n @ rho @ unitary_layer_5_n.conj().T
        if noise:
            rho = channels(rho, noise_choice=noise_choice, noise_values=noise_values) # Apply noise

        unitary_layer_6_n = cnotmat
        rho = unitary_layer_6_n @ rho @ unitary_layer_6_n.conj().T

        if noise:
            rho = channels(rho, noise_choice=noise_choice, noise_values=noise_values) # Apply noise

        unitary_layer_3_1 = hmat 
        unitary_layer_3_2 = rz(-2*params[8]) 
        unitary_layer_7_n = np.kron( unitary_layer_3_1, unitary_layer_3_2) 
        
        # Third Layer

        rho = unitary_layer_7_n @ rho @ unitary_layer_7_n.conj().T
        if noise:
            rho = channels(rho, noise_choice=noise_choice, noise_values=noise_values) # Apply noise
        
        unitary_layer_8_n = cnotmat
        rho = unitary_layer_8_n @ rho @ unitary_layer_8_n.conj().T

        if noise:
            rho = channels(rho, noise_choice=noise_choice, noise_values=noise_values) # Apply noise

        unitary_layer_9_n = np.kron(rz(2*params[9]) , imat )
        rho = unitary_layer_9_n @ rho @ unitary_layer_9_n.conj().T
        if noise:
            rho = channels(rho, noise_choice=noise_choice, noise_values=noise_values) # Apply noise
        
        unitary_layer_10_n = np.kron(ry(2*params[10]) , imat )
        rho = unitary_layer_10_n @ rho @ unitary_layer_10_n.conj().T
        if noise:
            rho = channels(rho, noise_choice=noise_choice, noise_values=noise_values) # Apply noise

        unitary_layer_11_n = np.kron(rz(2*params[11]) , imat )
        rho = unitary_layer_11_n @ rho @ unitary_layer_11_n.conj().T
        if noise:
            rho = channels(rho, noise_choice=noise_choice, noise_values=noise_values) # Apply noise

        return rho

    # Encodings
    def state(f, g, x_1, y_1, x_2, y_2, encoding_params):
        # Encode first two features, x_1, y_1 in first qubit
        rho_1 = np.array([
                [abs(f(x_1, y_1, encoding_params))**2, f(x_1, y_1, encoding_params) * np.conj(g(x_1, y_1, encoding_params))],
                [np.conj(f(x_1, y_1, encoding_params)) * g(x_1, y_1, encoding_params), abs(g(x_1, y_1, encoding_params))**2]
                ])

        # Encode second two features, x_2, y_2 in second qubit
        rho_2 = np.array([
                [abs(f(x_2, y_2, encoding_params))**2, f(x_2, y_2, encoding_params) * np.conj(g(x_2, y_2, encoding_params))],
                [np.conj(f(x_2, y_2, encoding_params)) * g(x_2, y_2, encoding_params), abs(g(x_2, y_2, encoding_params))**2]
                ])

        rho = np.kron( rho_1, rho_2 ) # Tensor product state

        return rho / np.trace(rho)

    
    def f_dae(x, y, encoding_params):
        return np.cos(encoding_params[0] * x)

    def g_dae(x, y, encoding_params):
        return np.exp(encoding_params[1] * 1j * y) * np.sin(encoding_params[0] * x)

    def f_wf(x, y, encoding_params=None):
        return ( np.sqrt( 1 + encoding_params[0] * y**2) * x ) / np.sqrt( x**2 + y**2 )

    def g_wf(x, y, encoding_params=None):
        return  ( np.sqrt( 1 - encoding_params[0] * x**2) * y )  / np.sqrt( x**2 + y**2 )

    def f_sdae(x, y, encoding_params):
        return np.cos(encoding_params[0] * x + encoding_params[1] * y)

    def g_sdae(x, y, encoding_params):
        return np.sin(encoding_params[0] * x + encoding_params[1] * y)

    def pauli(rho, pi=0.4, px=0.1, py=0.1, pz=0.4):
        """Applies a single qubit pauli channel to the state rho."""

        assert np.isclose(sum((pi, px, py, pz)), 1.0)
        
        return (pi * rho + px * xmat @ rho @ xmat + 
                        py * ymat @ rho @ ymat + pz * zmat @ rho @ zmat)
    
    def one_q_pauli_kraus(pi, px, py, pz):
        kraus = [np.sqrt(pi) * imat, np.sqrt(px) * xmat, \
                    np.sqrt(py) * ymat, np.sqrt(pz) * zmat]
        return kraus

    def two_q_pauli_kraus(pi, px, py, pz):
        [pi_1, pi_2] = pi
        [px_1, px_2] = px
        [py_1, py_2] = py
        [pz_1, pz_2] = pz

        assert np.isclose(sum((pi_1, px_1, py_1, pz_1)), 1.0)
        assert np.isclose(sum((pi_2, px_2, py_2, pz_2)), 1.0)

        kraus_1 = one_q_pauli_kraus(pi_1, px_1, py_1, pz_1)
        kraus_2 = one_q_pauli_kraus(pi_2, px_2, py_2, pz_2)
        pauli_kraus =  [np.kron(k1, k2) for k1 in kraus_1 for k2 in kraus_2]

        return pauli_kraus

    def two_q_pauli(rho, pi, px, py, pz):
        '''
            Applies random single qubit Pauli gates to 
            each qubit in the state with potentially different strengths
        '''
        pauli_kraus = two_q_pauli_kraus(pi, px, py, pz)
        # constuct list of all Kraus operators being applied to the state and sum to 
        # generate noisy state
        rho = np.array([k @ rho @ k.conj().T for k in pauli_kraus]).sum(axis=0) 
        return rho

    def two_q_global_depolarizing(rho, p=0.1):
        return (1-p)* rho + (p) * np.kron(imat, imat) / 2**2

    def amplitude_damping(rho, p):
        E_00 = np.array([ [1, 0], [0, np.sqrt( 1 - p[0] )] ])
        E_01 = np.array([ [0, np.sqrt( p[0] )], [0, 0] ])
        
        E_10 = np.array([ [1, 0], [0, np.sqrt( 1 - p[1] )] ])
        E_11 = np.array([ [0, np.sqrt( p[1] )], [0, 0] ])
        
        rho =  np.kron(E_00, E_10) @ rho @ np.kron(E_00, E_10).conj().T + np.kron(E_00, E_11) @ rho @ np.kron(E_00, E_11).conj().T \
            + np.kron(E_01, E_10) @ rho @ np.kron(E_01, E_10).conj().T + np.kron(E_01, E_11) @ rho @ np.kron(E_01, E_11).conj().T
        
        return rho 

    # Noise channels
    def channels(rho, noise_choice='Pauli', noise_values=None):
        if noise_choice.lower() == 'pauli':
            pi, px, py, pz = noise_values
            rho = two_q_pauli(rho, pi, px, py, pz) 

        elif noise_choice.lower() == 'global_depolarizing':
            p = noise_values
            rho = two_q_global_depolarizing(rho, p)

        elif noise_choice.lower() == 'dephasing':
            pz  = noise_values
            rho =  two_q_pauli(rho, [1-pz[0], 1-pz[1]], pz, [0, 0], [0, 0]) 

        elif noise_choice.lower() == 'bit_flip':
            px  = noise_values
            rho =  two_q_pauli(rho, [1-px[0], 1-px[1]], px, [0, 0], [0, 0]) 

        elif noise_choice.lower() == 'amp_damp':
            p = noise_values
            rho = amplitude_damping(rho, p)
        else: raise NotImplementedError

        return rho

    # Predictions
    def make_prediction(rho):
        # Compute prediction for data sample in state rho
        proj_0 = np.kron( np.kron( pi0, imat ), imat )

        elt = np.trace( proj_0 @ rho)
        if elt.real >= 0.499999999999999999:
            return 0
        return 1

    def compute_cost(ideal_labels, noisy_labels):
        """
        Computes the cost between the ideal case labels, and the labels
        produced in the presence of noise. Simple indicator cost.
        """
        assert len(ideal_labels) == len(noisy_labels)
        sum_count = 0
        for ii in range(len(ideal_labels)):
            if not ideal_labels[ii] == noisy_labels[ii]:
                sum_count += 1
        average_sum = sum_count / len(ideal_labels)
        return average_sum

    def set_params(data_choice='iris', encoding_choice='denseangle_param'):

        if data_choice.lower() == 'iris':
            if   encoding_choice.lower() == 'denseangle_param':         ideal_params = [ 2.02589489, 1.24358318, -0.6929718,\
                                                                                    0.85764484, 2.7572075, 1.12317156, \
                                                                                    4.01974889, 0.30921738, 0.88106973,\
                                                                                    1.461694, 0.367226, 5.01508911 ]         
                 
            elif    encoding_choice.lower() == 'superdenseangle_param': ideal_params = [1.1617383, -0.05837820, -0.7216498,\
                                                                                    1.3195103, 0.52933357, 1.2854939,\
                                                                                    1.2097700, 0.26920745, 0.4239539, \
                                                                                    1.2999367, 0.37921617, 0.790320211]

            elif    encoding_choice.lower()  == 'wavefunction':         ideal_params = [ 2.37732073, 1.01449711, 1.12025344,\
                                                                                    -0.087440021, 0.46937127, 2.14387135, \
                                                                                    0.4696964, 1.444409282, 0.14412614,\
                                                                                    1.4825742, 1.0817654, 6.30943537 ]

            elif    encoding_choice.lower() == 'wavefunction_param':    ideal_params = [ 2.37732073, 1.01449711, 1.12025344,\
                                                                                    -0.087440021, 0.46937127, 2.14387135, \
                                                                                    0.4696964, 1.444409282, 0.14412614,\
                                                                                    1.4825742, 1.0817654, 6.30943537]

            else: raise NotImplementedError
        else: raise ValueError('This dataset has not been trained for.')

        return ideal_params


    def average_fidelity(encoding_choice='denseangle_param', encoding_params=None, noise_choice='Pauli', noise_values=None):
        rho_noiseless, rho_noisy = [], []

        params = set_params(data_choice='iris', encoding_choice=encoding_choice)
        
        fidelity = np.zeros(len(data_test)) # one element of fidelity per data point
        pred_labels_noisy = np.zeros(len(data_test), dtype=int)
        pred_labels_noiseless = np.zeros(len(data_test), dtype=int)
        print('------------------------------------')
        print('Encoding is:', encoding_choice)
        print('------------------------------------')

        for ii, point in enumerate(data_test):
            if true_labels_test[ii] == 0:
                label_qubit = np.array([[1, 0], [0, 0]])           
            else: label_qubit = np.array([[0, 0], [0, 1]])

            if encoding_choice.lower() == 'denseangle_param':
                rho_encoded = state(f_dae, g_dae, point[0], point[1], point[2], point[3], encoding_params) # encode data point

            elif encoding_choice.lower() == 'wavefunction_param':
                rho_encoded = state(f_wf, g_wf, point[0], point[1], point[2], point[3],  encoding_params) # encode data point

            elif encoding_choice.lower() == 'superdenseangle_param':
                rho_encoded = state(f_sdae, g_sdae, point[0], point[1], point[2], point[3],  encoding_params) # encode data point

            else: raise NotImplementedError
            rhos_noiseless_indiv = unitary_evolved_noiseless(rho_encoded, params)
            rho_noisy_indiv = unitary_evolved_noisy(rho_encoded, params, noise_choice=noise_choice, noise_values=noise_values)

            rho_noiseless.append( np.kron( rhos_noiseless_indiv, label_qubit ))            
            rho_noisy.append(np.kron( rho_noisy_indiv, label_qubit ))
      
            pred_labels_noiseless[ii] = make_prediction(rho_noiseless[ii])
            pred_labels_noisy[ii] = make_prediction(rho_noisy[ii])
            
            fidelity[ii] = ( np.trace(sqrtm(rho_noisy[ii] @ rho_noiseless[ii] )) )**2 # compute fidelity per sample
        cost_ideal  = compute_cost(true_labels_test, pred_labels_noiseless)
        cost_noisy  = compute_cost(true_labels_test, pred_labels_noisy)


        cost_difference = cost_noisy - cost_ideal

        rho_noiseless_mixed = np.array((1 / len(rho_noiseless)) * np.array(rho_noiseless).sum(axis=0))
        rho_noisy_mixed = (1 / len(rho_noisy)) * np.array(rho_noisy).sum(axis=0)

        avg_fidelity_mixed = (( np.trace(sqrtm( sqrtm(rho_noisy_mixed) @ rho_noiseless_mixed @ sqrtm(rho_noisy_mixed)  )) )**2).real # compute fidelity for mixed state

        avg_fidelity = (1/len(data_test)) * np.sum(fidelity, axis=0) # compute average fidelity over dataset
        avg_bound = (2/len(data_test)) * np.sum(np.square( np.ones_like(fidelity) - fidelity ) , axis=0) # compute average bound over dataset
        print(avg_fidelity)
        if avg_fidelity_mixed > 1:
            avg_fidelity_mixed = 1 # reset fidelity less than one if numerical precision makes it > 1
        
        mixed_bound = 2*np.sqrt(1 - avg_fidelity_mixed) # Bound on cost function error
    
        return cost_difference, avg_bound, mixed_bound, avg_fidelity, avg_fidelity_mixed

    encoding_params_dae = [ np.pi/2, 2 * np.pi]
    encoding_params_wf = [ 0 ]
    encoding_params_sdae = [ np.pi, 2 * np.pi]

    encoding_params = [encoding_params_dae, encoding_params_wf, encoding_params_sdae]

    num_points = 50

    if amp_damp_noise:
        fidelity_bound_plot(average_fidelity, encoding_params=encoding_params, noise_choice='amp_damp',show=True, num_points=num_points, num_qbs=2, legend=legend)
        fidelity_compare_plot(average_fidelity, encoding_params=encoding_params, noise_choice='amp_damp',show=True, num_points=num_points, num_qbs=2, legend=legend)

    if bit_flip_noise:
        fidelity_bound_plot(average_fidelity, encoding_params=encoding_params, noise_choice='bit_flip',show=True, num_points=num_points, num_qbs=2, legend=legend)
        fidelity_compare_plot(average_fidelity, encoding_params=encoding_params, noise_choice='bit_flip',show=True, num_points=num_points, num_qbs=2, legend=legend)

    if dephasing_noise:
        fidelity_bound_plot(average_fidelity, encoding_params=encoding_params, noise_choice='dephasing',show=True, num_points=num_points, num_qbs=2, legend=legend)
        fidelity_compare_plot(average_fidelity, encoding_params=encoding_params, noise_choice='dephasing',show=True, num_points=num_points, num_qbs=2, legend=legend)

    if global_depolarizing_noise:
        '''
            Global depolarizing noise on both qubits together.
        '''
        fidelity_bound_plot(average_fidelity, encoding_params=encoding_params, noise_choice='global_depolarizing', show=True, num_points=num_points, num_qbs=2, legend=legend)
        fidelity_compare_plot(average_fidelity, encoding_params=encoding_params, noise_choice='global_depolarizing',show=True, num_points=num_points, num_qbs=2, legend=legend)


if __name__ == "__main__":
    amp_damp = False
    bit_flip = False
    dephasing = False
    global_depo = True

    main(amp_damp_noise=amp_damp,  bit_flip_noise=bit_flip, dephasing_noise=dephasing, global_depolarizing_noise=global_depo, legend=False)
    # Create seperate plot to get legend
    # main(amp_damp_noise=amp_damp,  bit_flip_noise=bit_flip, dephasing_noise=dephasing, global_depolarizing_noise=global_depo, legend=True)


