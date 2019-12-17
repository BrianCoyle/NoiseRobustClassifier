import matplotlib.pyplot as plt
import numpy as np

import nisqai
from pyquil import get_qc, Program
from pyquil.api import WavefunctionSimulator
from _dense_angle_encoding import DenseAngleEncoding
from _wavefunction_encoding import WaveFunctionEncoding
from _superdense_angle_encoding import SuperDenseAngleEncoding
from _generalised_wavefunction_encoding import GeneralisedWaveFunctionEncoding
from nisqai.encode import WaveFunctionEncoding
from nisqai.encode._feature_maps import FeatureMap
# from nisqai.encode._encoders import angle_simple_linear
from _encoders import angle_param_linear, angle_param_combination
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

from classifier_circuits import NoisyCircuits
from data_gen import *
# from plots import plot_correct_classifications

# from classifier_circuits import decoherence_noise_model, measurement_noise

make_wf = WavefunctionSimulator()


def classifier_params(classifier_qubits, values, n_layers=1):
    """Returns Parameters for the copying ansatz for ideal copying circuit structure.

    Args:
        qubits : list
            List of qubits in the ansatz.

        value : Union[float, int]
            Initial parameter value that appears in all gates.
    """
   
    values = list(np.array(values).reshape(len(classifier_qubits), n_layers, 3))
    params = { qubit: list(values[ii]) for ii, qubit in enumerate(classifier_qubits)}  
    # params = {qubits[0]: [values[0]], qubits[1]: [values[1], values[2]], qubits[2]: [values[3], values[4], values[5]], qubits[3]: [ values[6] ]
    # params = {  qubits[0]: [[values[0], values[1], values[2]],\
    #                         [values[6]],\
    #                         [],\
    #                         [values[9], values[10], values[11]],\
    #                         # [values[9], values[10], values[11]],\
    #                         ],\
    #             qubits[1]: [[values[3], values[4], values[5]],\
    #                         [values[7]],\
    #                         [values[8]],\
    #                         ]}
    # params = {  qubits[0]: [[values[0], values[1], values[2]], [values[6]]],\
    #         qubits[1]: [[values[3], values[4], values[5]],  [values[7]]]}

    print('Parameters are:', params)

    return Parameters(params)


def add_measurement(circuit, qubits):
    """
    Add measure instructions to and declare classical memory
    """
    ro = circuit.declare('ro', 'BIT', 1)

    circuit += MEASURE(qubits[0], ro[0])

    return circuit

def compute_label(results):
    """
    Counts the number of 0's in a list of results, if it is greater than
    then number of 1's, output label = 0, else output label = 1 
    """
    new_result = list(results)

    num_zeros = new_result.count([0])
    num_ones = new_result.count([1])
    if num_zeros >= num_ones: return 0
    elif num_zeros < num_ones: return 1

class ClassificationCircuit(BaseAnsatz):
    """Class for working with classifier circuit."""

    def __init__(self,  classifier_qubits, feature_vectors, noise=None, noise_params=None):
        """Initializes a ClassifierAnsatz.

        Args:
            num_qubits : int
                Number of qubits in the ansatz
            
            """
        # initialize the BaseAnsatz class
        super().__init__(len(classifier_qubits))
  
        # get parameters for the ansatz
        self.data = CData(feature_vectors)
        self.classifier_qubits = classifier_qubits
        self.noise_model = noise
        self.noise_params = noise_params

    def _encoding(self, encoding_choice, encoding_params=None):
        """Encodes feature vectors in circuits"""
        if len(self.classifier_qubits) > 1:
            feature_dict = { self.classifier_qubits[ii] : [ii+1, ii+2] for ii in range(len(self.classifier_qubits))[1:]} 
        elif len(self.classifier_qubits) == 1:
            feature_dict = {}
        else: raise ValueError("Classifier qubits must be a list of length 1, or >1")

        zero_dict = {self.classifier_qubits[0]: [0, 1]}
        feature_dict.update(zero_dict)

        # feature_dict_ttn = { self.classifier_qubits[ii] : [ii] for ii in range(len(self.classifier_qubits))}

        self.feature_map = FeatureMap(feature_dict)

        self.encoding_params = encoding_params
        #self.circuits = [BaseAnsatz(len(classifier_qubits)) for _ in range(self.data.num_samples)] #No encoding
        if encoding_choice.lower() == 'denseangle':
            self.circuits = DenseAngleEncoding(self.data, angle_param_linear, self.encoding_params, self.feature_map).circuits

        elif encoding_choice.lower() == 'denseangle_param':
            self.circuits = DenseAngleEncoding(self.data, angle_param_linear, self.encoding_params, self.feature_map).circuits

        elif encoding_choice.lower() == 'superdenseangle_param':
            self.circuits = SuperDenseAngleEncoding(self.data, angle_param_combination, self.encoding_params, self.feature_map).circuits

        elif encoding_choice.lower() == 'wavefunction':
            self.circuits = WaveFunctionEncoding(self.data).circuits

        elif encoding_choice.lower() == 'wavefunction_param':
            self.circuits = GeneralisedWaveFunctionEncoding(self.data, self.encoding_params).circuits
        else: raise ValueError("Encoding not defined")

        return self

    def _add_class_circuits(self, qc=None, num_shots=None):
        """
        Add the (potentially noisy) parameterized circuit to every encoded Program
        """
        for ii, enc_circuit in enumerate(self.circuits):
            enc_circuit.circuit = NoisyCircuits(enc_circuit.circuit, self.params, self.classifier_qubits,\
                                                noise=self.noise_model, noise_params=self.noise_params,\
                                                qc=qc, num_shots=num_shots).circuit
                
        return self

    def _predict(self, qc, num_shots):
        """
        Make predictions for each (noisy) circuit by measuring. 
        Args:
            num_shots : int
                number of measurements to make on QuantumComputer for each feature vector
            qc : QuantumComputer
                Quantum Computer to build classifier on.

        Returns:
            predicted_labels : list
                list of predictions for each feature vector
        """
        labels = []
        self.device_qubits = qc.qubits()

        for ii, enc_circuit in enumerate(self.circuits):
            
            # enc_circuit.circuit = add_measurement(enc_circuit.circuit, self.device_qubits)
            # enc_circuit.circuit.wrap_in_numshots_loop(num_shots)
            if self.noise_model is not None and self.noise_model.lower() == 'decoherence_symmetric_ro':
                enc_circuit.circuit.wrap_in_numshots_loop(num_shots)
                executable = qc.compile(enc_circuit.circuit, to_native_gates=False, optimize=False) # compile noisy circuit onto chip
                # print("Circuit is:", enc_circuit.circuit,'\nExec is:', executable )
            else:  
                enc_circuit.circuit = add_measurement(enc_circuit.circuit, self.device_qubits)
                enc_circuit.circuit.wrap_in_numshots_loop(num_shots)
                native_gates = qc.compiler.quil_to_native_quil(enc_circuit.circuit)
                executable = qc.compile(native_gates, to_native_gates=False, optimize=False)


 # If full decoherence model is used, gates must be compiled to native gates
                # native_gates = qc.compiler.quil_to_native_quil(enc_circuit.circuit)
                # enc_circuit.circuit = qc.compiler.quil_to_native_quil(enc_circuit.circuit)
                # noisy_circ = NoisyCircuits(enc_circuit.circuit, self.params, self.classifier_qubits, self.noise_model, self.noise_params)
                
            # print(executable)
            # enc_circuit.circuit = enc_circuit.circuit.nativ

            # print(native_gates)
            # print("The noise parameters for the chip, ", qc.name, "are: ", self.noise_params)
            # noisy_circ = NoisyCircuits(enc_circuit.circuit, self.params, self.classifier_qubits, self.noise_model, self.noise_params)
            # print('NOISY CIRCUIT WITHOUT ERRORS:', noisy_circ.circuit)
            # noisy_circ.nativize(qc) # Nativize circuit for chip
            # noisy_circ.decoherence_noise_model() # Add decoherence noise for native gates
            # print('NOISY CIRCUIT WITHOUT ERRORS:', noisy_circ.circuit)
            # print(executable)

            results = qc.run(executable)
            # print(results)
            label = compute_label(results) 
    
            labels.append(label)

        self.predicted_labels = np.array(labels)
        return self.predicted_labels
    
    def _compute_cost(self, true_labels):
        """
        Compute cost between true labels and predicted labels
        """
        self.cost = 0
        for i in range(len(true_labels)):
            self.cost += indicator(self.predicted_labels[i], true_labels[i])

        return self.cost

    def make_predictions(self, param_values, n_layers, encoding_choice, encoding_params, num_shots, qc):

        self.params = classifier_params(self.classifier_qubits, param_values, n_layers)

        self._encoding(encoding_choice, encoding_params)
        self._add_class_circuits(qc)
        predicted_labels = self._predict(qc, num_shots)

        return predicted_labels
    
    def build_classifier(self, param_values, n_layers, encoding_choice, encoding_params, num_shots, qc, true_labels):
       
        self.params = classifier_params(self.classifier_qubits, param_values, n_layers)
        self._encoding(encoding_choice, encoding_params)
        self._add_class_circuits(qc)
        predicted_labels = self._predict(qc, num_shots)
        # print('Data point', self.data[0], '\nCircuit', self.circuits[0])

        cost = self._compute_cost(true_labels) / len(true_labels)
        print('True labels are: ', true_labels)
        print('Predicted labels are:', predicted_labels)
        print('The cost is:', cost)
      
        return cost

# data_train, data_test, true_labels_train, true_labels_test = generate_data('iris', num_points=500, split=True)
# print(data_train)
# qc_name  = '2q-qvm'
# qc =get_qc(qc_name)
# circ = ClassificationCircuit(qc.qubits(), data_train)

# init_params = np.random.rand(2, 3)
# print(init_params)

# init_encoding_params = [np.pi, 2*np.pi]

# circ.build_classifier(init_params, 'denseangle_param', init_encoding_params, 100, qc, true_labels_train)


def build_classifier_encoding(encoding_params, encoding_choice, param_values, noise, noise_params, num_shots, qc, classifier_qubits, data, true_labels):
    """
    Same functionality as build_classifier, except encoding params are passed as the argument to the optimiser
    """
    circ = ClassificationCircuit(classifier_qubits, data, noise, noise_params)
    circ.params = classifier_params(classifier_qubits, param_values)

    circ._encoding(encoding_choice, encoding_params)

    circ._add_class_circuits()
    predicted_labels = circ._predict(num_shots, qc, num_shots)
    cost = circ._compute_cost(true_labels) / len(true_labels)


    print('The cost is:', cost)
    
    return cost

def train_classifier(qc, classifier_qubits,  num_shots, init_params, n_layers, encoding_choice, encoding_params, optimiser, data, true_labels):
    """
    Trains the single qubit classifer
    """

    params = list(init_params)
    def store(current_params):
        params.append(list(current_params))
    # no_bnd = (-np.inf, np.inf)
    # bounds = (no_bnd,no_bnd,no_bnd,no_bnd,no_bnd,no_bnd,(0.0, np.pi/4),(0.0, np.pi/4),(0.0, np.pi/4),no_bnd,no_bnd,no_bnd) # Don't allow parameter to reach 1 in generalized wavefunction encoding
    # result = minimize(ClassificationCircuit(qubits, data).build_classifier, init_params,  method=optimiser,\
    #                                                                     callback=store,\
    #                                                                     bounds=bounds,\
    #                                                                     args=(n_layers, encoding_choice, encoding_params, num_shots, qc, true_labels),\
    #                                                                     options={'eps':0.1})

    result = minimize(ClassificationCircuit(classifier_qubits, data).build_classifier, init_params,    method=optimiser,\
                                                                                            callback=store,\
                                                                                            args=(n_layers, encoding_choice, encoding_params, num_shots, qc, true_labels))
    return params, result


def train_classifier_encoding(qc, classifier_qubits, noise, noise_params, num_shots, init_params, encoding_choice, encoding_params, optimiser, data, true_labels):
    """
    Trains the single qubit classifer in the *encoding* unitary.
    """

    encoding_params = list(encoding_params)

    def store(current_params):
        encoding_params.append(list(current_params))
    if encoding_choice.lower() == 'wavefunction_param':
        bounds = ((0.0, 0.999),) # Don't allow parameter to reach 1 in generalized wavefunction encoding
        result = minimize(build_classifier_encoding, encoding_params,   method=optimiser,\
                                                                        callback=store,\
                                                                        bounds=bounds,\
                                                                        args=(encoding_choice, init_params, noise, noise_params, num_shots, qc, classifier_qubits,  data, true_labels),\
                                                                        options={'eps':0.01})
    else:
        result = minimize(build_classifier_encoding, encoding_params,   method=optimiser,\
                                                                        callback=store,\
                                                                        args=(encoding_choice, init_params, noise, noise_params, num_shots, qc, classifier_qubits, data, true_labels))                                     

    return encoding_params, result


def compute_number_misclassified(true_labels, noisy_predictions):
    return abs((true_labels-noisy_predictions)).sum()/len(true_labels)

def generate_noisy_classification(ideal_params,  n_layers, noise_type, noise_params, encoding_choice, encoding_params, qc, classifier_qubits, num_shots, data, true_labels):

    number_points = len(true_labels)

    circ =  ClassificationCircuit(classifier_qubits, data, noise=noise_type, noise_params=noise_params)
  
    noisy_predictions = circ.make_predictions(ideal_params, n_layers,  encoding_choice, encoding_params, num_shots, qc)

    number_correctly_classified = 1 - compute_number_misclassified(true_labels, noisy_predictions)

    return noisy_predictions, number_correctly_classified
