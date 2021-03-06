from single_qubit_classifier import generate_noisy_classification
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np

from pyquil import get_qc

from nisqai.data._cdata import CData
from pyquil.gates import *
from scipy.optimize import minimize
from collections import Counter
import random
from sklearn.datasets import make_circles, make_moons

from noise_models import add_noisy_gate_to_circ
from classifier_circuits import *
from data_gen import *
from single_qubit_classifier import train_classifier, ClassificationCircuit, classifier_params
from plots import plot_correct_classifications, scatter
from data_gen import generate_data, remove_zeros

def main(train=False, encoding='denseangle_param', ideal=False, noise=False, analytic=False, compare=False):

    """
    # Find optimal parameters for linear decision boundary and add noise
    """

    ### Firstly, generate for dataset:
    '''
    # We use the transpose of the (scaled to unit square) Moons dataset in order to see a non-linear decision boundary
    '''
    data_vertical_train, data_vertical_test, true_labels_train, true_labels_test = generate_data('random_vertical_boundary', num_points=500, split=True)

    ### Next, generate correct classification parameters for dataset (perfect classification):
    '''
    # Define parameters of model. Start with DenseAngle encoding with fixed parameters.
    '''

    qc_name = '1q-qvm'
    qc = get_qc(qc_name)
    num_shots = 1024
    device_qubits = qc.qubits()
    classifier_qubits = device_qubits
    n_layers = 1
    init_params = np.random.rand(3)
    if encoding.lower() == 'denseangle_param':
        encoding_choice = 'denseangle_param'
        # init_encoding_params = [np.pi, 2*np.pi]
        init_encoding_params = [np.pi, 2*np.pi]

    elif encoding.lower() == 'wavefunction' or encoding.lower() == 'wavefunction_param':
        encoding_choice = 'wavefunction_param'
        init_encoding_params = [0]

    optimiser = 'Powell' 

    if train:
        ### Train model, and check classification result of ideal parameters found
        '''
        # Train model using scipy.optimize
        '''
        params, result_unitary_param = train_classifier(qc, num_shots, init_params, encoding_choice, init_encoding_params, optimiser, data_vertical_train, true_labels_train)
        print('The optimised parameters are:', result_unitary_param.x)
        print('These give a cost of:', ClassificationCircuit(classifier_qubits, data_vertical_train).build_classifier(result_unitary_param.x, n_layers, \
                                                                            encoding_choice, init_encoding_params, num_shots, qc, true_labels_train))
        ideal_params_vertical = result_unitary_param.x
    else:
        ### Define Ideal parameters for trained model learned from previous. Simple model can acheieve classification of about 90 %
    
        if encoding_choice.lower() == 'denseangle_param':
            '''
            # 100% Classification parameters (modulo points on the boundary)
            '''
            # ideal_params_vertical = [3.8208,1.525,0.0808]
            ideal_params_vertical = [1.67814786, 1.56516469, 1.77820848]
        elif encoding_choice.lower() == 'wavefunction_param':
            '''
            # 78% Classification parameters (modulo points on the boundary)
            '''
            ideal_params_vertical =  [ 2.2921198,  0.61375299, -5.15252796]

    plt.rcParams.update({"font.size": 20, "font.serif": "Computer Modern Roman"})

    ### Overlay decision bounday
    '''
    # Generate Grid of datapoints to determine and visualise ideal decision boundary
    '''
    data_choice = 'full_vertical_boundary'
    num__grid_points = 1000
    data_grid, grid_true_labels = generate_data(data_choice, num__grid_points)
    data_grid, grid_true_labels = remove_zeros(data_grid, grid_true_labels)
    
    if ideal:

        predicted_labels_test = ClassificationCircuit(classifier_qubits, data_vertical_test, qc).make_predictions(ideal_params_vertical,  n_layers, \
                                                                                        encoding_choice, init_encoding_params, num_shots)
        plot_params = {'colors': ['blue', 'orange'], 'alpha': 1}
        scatter(data_vertical_test, true_labels_test, predicted_labels_test, **plot_params)

        predicted_labels_grid = ClassificationCircuit(classifier_qubits, data_grid, qc).make_predictions(ideal_params_vertical, n_layers,\
                                                                             encoding_choice, init_encoding_params, num_shots)
        plot_params = {'colors': ['red', 'green'], 'alpha': 0.2}
        scatter(data_grid, predicted_labels_grid, **plot_params)
        plt.show()


    ### Define noise parameters
    '''
        # Define noise parameters to add to model to determine how classification is affected.
    '''

    noise_choice  ='amp_damp_before_measurement'
    noise_values = 0.4

    ### Add noise to circuit and classify
    '''
        # Add noise to circuit, and determine number of points classified differently (not mis-classified since we can't achieve perfect classification)
    '''
    
    if noise:
        ## Overlay decision boundary
        '''
        # Generate Grid of datapoints to determine and visualise ideal decision boundary WITH noise added
        '''
        predicted_labels_test_noise = ClassificationCircuit(classifier_qubits, data_vertical_test, qc,\
                     noise_choice, noise_values).make_predictions(ideal_params_vertical, n_layers, encoding_choice, init_encoding_params, num_shots)
        plot_params = {'colors': ['blue', 'orange'], 'alpha': 1}
        scatter(data_vertical_test, true_labels_test, predicted_labels_test_noise, **plot_params)

        predicted_labels_grid_noise = ClassificationCircuit(classifier_qubits, data_grid, qc,\
                                                            noise_choice, noise_values).make_predictions(ideal_params_vertical, n_layers, \
                                                            encoding_choice, init_encoding_params, num_shots)

        plot_params = {'colors': ['red', 'green'], 'alpha': 0.2}
        scatter(data_grid, predicted_labels_grid_noise, **plot_params)
        plt.show()

    '''
    # Define function to compute points which will remian correctly classified after noise is added
    '''

    def correct_function(data_point, params, encoding_choice, encoding_params):
        [alpha_1, alpha_2, alpha_3] = params
        [x_1, x_2] = data_point

        if encoding_choice.lower() == 'denseangle_param':
            [theta, phi] = encoding_params
            function = (np.sin(alpha_2) )**2 * ( np.cos(theta * x_1) )**2  + (np.cos(alpha_2))**2 * (np.sin(theta * x_1))**2 \
                        + ((1/2)*(np.sin(2 * alpha_2) * np.sin(2 * theta * x_1) * np.exp(-1j*(2 * alpha_3 + phi * x_2)))).real
        elif encoding_choice.lower() == 'wavefunction_param':
            [theta] = encoding_params
            l2_norm = np.linalg.norm(np.array([x_1, x_2]))**2
            function = (np.sin(alpha_2)**2 ) * ( x_1**2/(l2_norm) )  + (np.cos(alpha_2)**2) * (x_2**2/(l2_norm)) \
                        + ((1/(2*l2_norm))*(np.sin(2 * alpha_2) * (x_1) * (x_2) * np.exp(-1j*(2 * alpha_3)))).real

        return function

    def compute_analytic_misclassifed_condition(data, params, encoding_choice, encoding_params, noise_strength, true_labels):
        correct_classification_labels = []
        for ii, data_point in enumerate(data):
        
            function = correct_function(data_point, params, encoding_choice, encoding_params)
            if true_labels[ii] == 0:
                correct_classification_labels.append(0) # If datapoint was zero originally, it will be correctly classified regardless of noise
 
            else:
                if function > 1/(2*(1-noise_strength)): # If data point was classified as 1, it will be correctly classified if condition is met.
                    correct_classification_labels.append(0) 

                else: correct_classification_labels.append(1)
        number_robust = 1- sum(correct_classification_labels)/len(correct_classification_labels) # percentage of misclassified points
        return np.array(correct_classification_labels), number_robust




    def plot_number_misclassified_amp_damp(ideal_params, num_shots, num_points, qc, noise_values):

        points_noise_inc = []

        data_vertical_train, data_vertical_test, true_labels_train, true_labels_test = generate_data('random_vertical_boundary',\
                                                                                                     num_points=num_points, split=True)
        interval = 0.2
        encoding_choice = 'denseangle_param'
        theta = np.arange(0, 2*np.pi, interval)
        phi = np.arange(0, 2*np.pi, interval)
        X, Y = np.meshgrid(theta, phi)
        noise_choice = 'amp_damp_before_measurement'
        test_acc_ideal = np.zeros((theta.shape[0], phi.shape[0]), dtype=float)

        test_acc_noise = np.zeros((theta.shape[0], phi.shape[0]), dtype=float)
        number_robust = np.zeros((theta.shape[0], phi.shape[0]), dtype=float)

        for ii, t in enumerate(theta):
            for jj, p in enumerate(phi):
                temp_encoding_params = [t, p]

                # Classification of encoding parameters *without* noise
                ideal_predictions, test_acc_ideal[ii,jj]  = generate_noisy_classification(ideal_params, 1, None, None,\
                                                                                                    encoding_choice, temp_encoding_params, qc,\
                                                                                                    classifier_qubits, num_shots, data_vertical_test, true_labels_test)
            
                # Learned encoding parameters *with* noise
                noisy_predictions, test_acc_noise[ii,jj]  = generate_noisy_classification(ideal_params, 1, noise_choice, noise_values,\
                                                                                                    encoding_choice, temp_encoding_params, qc,\
                                                                                                    classifier_qubits, num_shots, data_vertical_test, true_labels_test)
                # Number expected to be robust under analytic condition
                correct_classification_labels, number_robust[ii, jj] = compute_analytic_misclassifed_condition(data_vertical_test, ideal_params_vertical,\
                                                                                                                encoding_choice, temp_encoding_params,\
                                                                                                                noise_values, true_labels_test)

                print('Theta, Phi is:', t, p)
                print('Test accuracy ideal:', test_acc_ideal[ii, jj])
                print('Test accuracy with noise:' ,test_acc_noise[ii, jj])
                print('Proportion robust:', number_robust[ii, jj])

        max_acc_indices_ideal = np.unravel_index(np.argmax(test_acc_ideal, axis=None), test_acc_ideal.shape)
        max_acc_indices = np.unravel_index(np.argmax(test_acc_noise, axis=None), test_acc_noise.shape)
        max_robust_indices = np.unravel_index(np.argmax(number_robust, axis=None), number_robust.shape)

        plt.rcParams.update({"font.size": 14, "font.family": "serif" })

        # ----------------------
        # Uncomment below for 3d plots
        # ----------------------

        # fig = plt.figure(figsize=plt.figaspect(0.33))
        # ax1 = fig.add_subplot(1, 3, 1, projection='3d')
        # surf1 = ax1.plot_surface(X, Y, test_acc_ideal, cmap=cm.coolwarm_r,linewidth=0, antialiased=False)
        # # ax1.set_zlim(0.45, 1.01)
        # cbar1 =fig.colorbar(surf1)
        # cbar1.ax.set_ylabel('Test Accuracy')

        # ax2 = fig.add_subplot(1, 3, 2, projection='3d')
        # surf2 = ax2.plot_surface(X, Y, test_acc_noise, cmap=cm.coolwarm_r, linewidth=0, antialiased=False)
        # # ax2.set_zlim(0.45, 1.01)
        # cbar2 = fig.colorbar(surf2)
        # cbar2.ax.set_ylabel('Test Accuracy')

        # ax3 = fig.add_subplot(1, 3, 3, projection='3d')

        # surf3 = ax3.plot_surface(X, Y, number_robust, cmap=cm.PuOr, linewidth=0, antialiased=False)
        # cbar3 = fig.colorbar(surf3)
        # cbar3.ax.set_ylabel('Proportion robust')

        # ax1.set_ylabel(r'$\theta (rads)$')
        # ax1.set_xlabel(r'$\phi (rads)$' )
        # ax1.set_title(  'Best accuracy ideal: '             + str( round( test_acc_ideal[max_acc_indices_ideal] , 2) ) \
        #                 + '\nBest accuracy with noise: '    + str( round( test_acc_noise[max_acc_indices_ideal] , 2) ) \
        #                 + '\nRobustness: '                  + str( round( number_robust[max_acc_indices_ideal]  , 2) ) + '\n' \
        #                 + r'$[\theta, \phi]$ = '            + '['+str(round(theta [ max_acc_indices_ideal[0] ], 2) )+ ', ' + str( round( phi [ max_acc_indices_ideal[1] ] , 2) ) + ']' )
        
        # ax2.set_ylabel(r'$\theta (rads)$')
        # ax2.set_xlabel(r'$\phi (rads)$' )
        # ax2.set_title(  'Best accuracy with noise: '    + str( round( test_acc_noise[max_acc_indices]   , 2) ) \
        #                 + '\nBest accuracy ideal: '     + str( round( test_acc_ideal[max_acc_indices]   , 2) ) \
        #                 + '\nRobustness: '               + str( round( number_robust[max_acc_indices]    , 2) ) + '\n' \
        #                 + r'$[\theta, \phi]$ = '        + '['+str(theta [ max_acc_indices[0] ])+ ', ' + str(round( phi [ max_acc_indices[1] ], 2) ) + ']' )

        # ax3.set_ylabel(r'$\theta (rads)$')
        # ax3.set_xlabel(r'$\phi (rads)$' )
        # ax3.set_title('Max. robustness: '               + str( round( number_robust[max_robust_indices]  , 2) ) \
        #                 +'\nBest accuracy with noise: ' + str( round( test_acc_noise[max_robust_indices] , 2) ) \
        #                 +'\nBest accuracy ideal: '      + str( round( test_acc_ideal[max_robust_indices] , 2) ) + '\n'\
        #                 +r'$[\theta, \phi]$ = '         + '[' + str(theta [ max_robust_indices[0] ]) \
        #                                                 + ', ' + str(phi [ max_robust_indices[1] ] ) + ']' )
        
        
        ## 2D PLOTS
        fig, ax = plt.subplots(1, 3)
        im0 = ax[0].imshow(test_acc_ideal, cmap=cm.coolwarm_r, extent=[0, 2*np.pi,2*np.pi, 0])
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.1)
        cbar0 = fig.colorbar(im0, cax=cax, orientation='vertical')

        cbar0.ax.set_ylabel('Test Accuracy')

        im1 =  ax[1].imshow(test_acc_noise, cmap=cm.coolwarm_r, extent=[0, 2*np.pi,2*np.pi, 0])
        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.1)
        cbar1 = fig.colorbar(im1, cax=cax, orientation='vertical')

        cbar1.ax.set_ylabel('Test Accuracy')

        im2 = ax[2].imshow(number_robust, cmap=cm.PuOr, extent=[0, 2*np.pi, 2*np.pi, 0])
        divider = make_axes_locatable(ax[2])
        cax = divider.append_axes('right', size='5%', pad=0.1)
        cbar2 = fig.colorbar(im2, cax=cax, orientation='vertical')

        cbar2.ax.set_ylabel('Proportion robust')

        ax[0].set_title(  'Best accuracy ideal: '             + str( round( test_acc_ideal[max_acc_indices_ideal] , 2) ) \
                        + '\nBest accuracy with noise: '    + str( round( test_acc_noise[max_acc_indices_ideal] , 2) ) \
                        + '\nRobustness: '                  + str( round( number_robust[max_acc_indices_ideal]  , 2) ) + '\n' \
                        + r'$[\theta, \phi]$ = '            + '['+str(round(theta [ max_acc_indices_ideal[0] ], 2) )+ ', ' + str( round( phi [ max_acc_indices_ideal[1] ] , 2) ) + ']' )

        ax[1].set_title(  'Best accuracy with noise: '    + str( round( test_acc_noise[max_acc_indices]   , 2) ) \
                        + '\nBest accuracy ideal: '     + str( round( test_acc_ideal[max_acc_indices]   , 2) ) \
                        + '\nRobustness: '               + str( round( number_robust[max_acc_indices]    , 2) ) + '\n' \
                        + r'$[\theta, \phi]$ = '        + '['+str(theta [ max_acc_indices[0] ])+ ', ' + str(round( phi [ max_acc_indices[1] ], 2) ) + ']' )

        ax[2].set_title('Max. robustness: '               + str( round( number_robust[max_robust_indices]  , 2) ) \
                        +'\nBest accuracy with noise: ' + str( round( test_acc_noise[max_robust_indices] , 2) ) \
                        +'\nBest accuracy ideal: '      + str( round( test_acc_ideal[max_robust_indices] , 2) ) + '\n'\
                        +r'$[\theta, \phi]$ = '         + '[' + str(theta [ max_robust_indices[0] ]) \
                                                        + ', ' + str(phi [ max_robust_indices[1] ] ) + ']' )

        return



    if analytic:
        correct_classification_labels, number_robust = compute_analytic_misclassifed_condition(data_grid, ideal_params_vertical,\
                                                                                                encoding_choice, init_encoding_params,\
                                                                                                noise_values, grid_true_labels)
        plot_params = {'colors': ['blue', 'black'], 'alpha': 0.3}

        scatter(data_grid, correct_classification_labels, **plot_params)
        plt.show()

    if compare:
        plot_number_misclassified_amp_damp(ideal_params_vertical, num_shots, 500, qc, noise_values)
        plt.show()

if __name__ == "__main__":
    # main(train=False, encoding='denseangle_param', ideal=False, noise=False, analytic=True, compare=False)
    main(train=False, encoding='wavefunction_param', ideal=False, noise=False, analytic=True, compare=False)
