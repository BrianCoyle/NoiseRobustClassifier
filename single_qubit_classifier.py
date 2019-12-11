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

from noise_models import add_noisy_gate_to_circ, estimate_meas_noise
from classifier_circuits import NoisyCircuits
from data_gen import *
# from plots import plot_correct_classifications

# from classifier_circuits import decoherence_noise_model, measurement_noise

make_wf = WavefunctionSimulator()


def classifier_params(qubits, values):
    """Returns Parameters for the copying ansatz for ideal copying circuit structure.

    Args:
        qubits : list
            List of qubits in the ansatz.

        value : Union[float, int]
            Initial parameter value that appears in all gates.
    """

    params = {qubits[0]: [value for value in values]}  
    print(params)
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

    def __init__(self,  qubits, feature_vectors, noise=None, noise_params=None):
        """Initializes a ClassifierAnsatz.

        Args:
            num_qubits : int
                Number of qubits in the ansatz
            
            """
        # initialize the BaseAnsatz class
        super().__init__(len(qubits))
  
        # get parameters for the ansatz
        self.data = CData(feature_vectors)
        self.qubits = qubits
        self.noise_model = noise
        self.noise_params = noise_params

    def _add_params(self, param_values):
        self.params = classifier_params(self.qubits, param_values)
        return self

    def _encoding(self, encoding_choice, encoding_params=None):
        """Encodes feature vectors in circuits"""
        self.feature_map = FeatureMap({self.qubits[0]: [0, 1]})
        self.encoding_params = encoding_params

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


        return self

    def _add_class_circuits(self):
        """
        Add the (potentially noisy) parameterized circuit to every encoded Program
        """
        for ii, enc_circuit in enumerate(self.circuits):
            enc_circuit.circuit = NoisyCircuits(enc_circuit.circuit, self.params, self.qubits, noise=self.noise_model, noise_params=self.noise_params).circuit
                
        return self

    def _predict(self, num_shots, qc):
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
        for ii, enc_circuit in enumerate(self.circuits):
            enc_circuit.circuit = add_measurement(enc_circuit.circuit, self.qubits)
            enc_circuit.circuit.wrap_in_numshots_loop(num_shots)
            if self.noise_model is not None and self.noise_model.lower() == 'decoherence_symmetric_ro':
                # If full decoherence model is used, gates must be compiled to native gates
                native_gates = qc.compiler.quil_to_native_quil(enc_circuit.circuit)
                executable = NoisyCircuits(native_gates, self.params, self.qubits, self.noise_params).decoherence_noise_model() 
            else:  
    
                native_gates = qc.compiler.quil_to_native_quil(enc_circuit.circuit)
                executable = qc.compile(native_gates, to_native_gates=False, optimize=False)
               
            results = qc.run(enc_circuit.circuit)
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

    def make_predictions(self, param_values, encoding_choice, encoding_params, num_shots, qc):

        self.params = classifier_params(self.qubits, param_values)

        self._encoding(encoding_choice, encoding_params)
        self._add_class_circuits()
        predicted_labels = self._predict(num_shots, qc)

        return predicted_labels
    
    def build_classifier(self, param_values, encoding_choice, encoding_params, num_shots, qc, true_labels):
       
        self.params = classifier_params(self.qubits, param_values)

        self._encoding(encoding_choice, encoding_params)
        self._add_class_circuits()
        predicted_labels = self._predict(num_shots, qc)
        cost = self._compute_cost(true_labels) / len(true_labels)
        # print('True labels are: ', true_labels)
        # print('Predicted labels are:', predicted_labels)
        print('The cost is:', cost)
      
        return cost




def build_classifier_encoding(encoding_params, encoding_choice, param_values, noise, noise_params, num_shots, qc, data, true_labels):
    """
    Same functionality as build_classifier, except encoding params are passed as the argument to the optimiser
    """
    qubits = qc.qubits()
    circ = ClassificationCircuit(qubits, data, noise, noise_params)
    circ.params = classifier_params(qubits, param_values)

    circ._encoding(encoding_choice, encoding_params)

    circ._add_class_circuits()
    predicted_labels = circ._predict(num_shots, qc)
    cost = circ._compute_cost(true_labels) / len(true_labels)


    print('The cost is:', cost)
    
    return cost

def train_classifier(qc, num_shots, init_params, encoding_choice, encoding_params, optimiser, data, true_labels):
    """
    Trains the single qubit classifer
    """
    qubits = qc.qubits()

    params = list(init_params)

    def store(current_params):
        params.append(list(current_params))
    print(optimiser)
    result = minimize(ClassificationCircuit(qubits, data).build_classifier, init_params,    method=optimiser,\
                                                                                            callback=store,\
                                                                                            args=(encoding_choice, encoding_params, num_shots, qc, true_labels))
    return params, result


def train_classifier_encoding(qc, noise, noise_params, num_shots, init_params, encoding_choice, encoding_params, optimiser, data, true_labels):
    """
    Trains the single qubit classifer in the *encoding* unitary.
    """
    qubits = qc.qubits()

    encoding_params = list(encoding_params)

    def store(current_params):
        encoding_params.append(list(current_params))
    if encoding_choice.lower() == 'wavefunction_param':
        bounds = ((0.0, 0.999),) # Don't allow parameter to reach 1 in generalized wavefunction encoding
        result = minimize(build_classifier_encoding, encoding_params,   method=optimiser,\
                                                                        callback=store,\
                                                                        bounds=bounds,\
                                                                        args=(encoding_choice, init_params, noise, noise_params, num_shots, qc, data, true_labels),\
                                                                        options={'eps':0.01})
    else:
        result = minimize(build_classifier_encoding, encoding_params,   method=optimiser,\
                                                                        callback=store,\
                                                                        args=(encoding_choice, init_params, noise, noise_params, num_shots, qc, data, true_labels))                                     

    return encoding_params, result


def compute_number_misclassified(true_labels, noisy_predictions):
    return abs((true_labels-noisy_predictions)).sum()/len(true_labels)

def generate_noisy_classification(ideal_params, noise_type, noise_params, encoding_choice, encoding_params, qc, num_shots, data, true_labels):

    number_points = len(true_labels)
    qubits = qc.qubits()
    circ =  ClassificationCircuit(qubits, data, noise=noise_type, noise_params=noise_params)
  
    noisy_predictions = circ.make_predictions(ideal_params, encoding_choice, encoding_params, num_shots, qc)

    number_correctly_classified = 1 - compute_number_misclassified(true_labels, noisy_predictions)

    return noisy_predictions, number_correctly_classified
