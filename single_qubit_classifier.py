import matplotlib.pyplot as plt
import numpy as np

import nisqai
from pyquil import get_qc
from pyquil.api import WavefunctionSimulator
from nisqai.encode import DenseAngleEncoding, WaveFunctionEncoding
from nisqai.encode._feature_maps import FeatureMap
from nisqai.encode._encoders import angle_simple_linear
from nisqai.layer._base_ansatz import BaseAnsatz
from nisqai.layer._params import Parameters
from nisqai.data._cdata import CData
from nisqai.cost._classical_costs import indicator
from pyquil.gates import *
from scipy.optimize import minimize
from collections import Counter
import random
from sklearn.datasets import make_circles, make_moons

from noise_models import add_noisy_gate_to_circ
from classifier_circuits import ideal_circuit, dephased_before_meas_circuit, pauli_before_meas_circuit

make_wf = WavefunctionSimulator()

def grid_pts(nx=12, ny=12):
    return np.array([[x, y] for x in np.linspace(0, 1, nx) 
                     for y in np.linspace(0, 1, ny)])    



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
    # elif num_ones == num_zeros: return int(random.choice([0, 1]))

class ClassificationCircuit(BaseAnsatz):
    """Class for working with copying ansatze."""

    def __init__(self,  qubits, feature_vectors):
        """Initializes a CopyingAnsatz.

        Args:
            num_qubits : int
                Number of qubits in the ansatz
            
            """
        # initialize the BaseAnsatz class
        super().__init__(len(qubits))
  
        # get parameters for the ansatz
        self.data = CData(feature_vectors)
        self.qubits = qubits

    def _add_params(self, param_values):
        self.params = classifier_params(self.qubits, param_values)
        return self

    def _encoding(self, encoding_choice):
        """Encodes feature vectors in circuits"""

        if encoding_choice.lower() == 'denseangle':
            self.feature_map = FeatureMap({self.qubits[0]: [0, 1]})
       
            self.circuits = DenseAngleEncoding(self.data, angle_simple_linear, self.feature_map).circuits
        return self

    def _add_class_circuits(self, noise="None", noise_strength=0):
        for ii, enc_circuit in enumerate(self.circuits):
            if noise.lower() == "none":
                enc_circuit.circuit = ideal_circuit(enc_circuit.circuit, self.params, self.qubits)
            elif noise.lower() == "depolarizing_before_meas":
                enc_circuit.circuit = dephased_before_meas_circuit(enc_circuit.circuit, self.params, self.qubits, noise_strength)
            elif noise.lower() == "pauli_before_meas":
                enc_circuit.circuit = pauli_before_meas_circuit(enc_circuit.circuit, self.params, self.qubits, noise_strength)

        return self

    def _make_predictions(self, num_shots, qc):
        
        labels = []
        for ii, enc_circuit in enumerate(self.circuits):
            enc_circuit.circuit = ideal_circuit(enc_circuit.circuit, self.params, self.qubits)
            enc_circuit.circuit = add_measurement(enc_circuit.circuit, self.qubits)
            enc_circuit.circuit.wrap_in_numshots_loop(num_shots)
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

    def build_classifier(self, param_values, encoding_choice, num_shots, qc,true_labels):
        self.params = classifier_params(self.qubits, param_values)

        self._encoding(encoding_choice)
        self._add_class_circuits
        predicted_labels = self._make_predictions(num_shots, qc)
        cost = self._compute_cost(true_labels) / len(true_labels)
        print('True labels are: ', true_labels)
        print('Predicted labels are:', predicted_labels)
        print('The cost is:', cost)
      
        return cost




def train_classifier(qc, num_shots, init_params, encoding_choice, optimiser, data, true_labels):
    """
    Trains the single qubit classifer
    """
    qubits = qc.qubits()

    params = list(init_params)

    def store(current_params):
        params.append(list(current_params))

    result = minimize(ClassificationCircuit(qubits, data).build_classifier, init_params,    method=optimiser,\
                                                                                            callback=store,\
                                                                                            args=(encoding_choice, num_shots, qc, true_labels))
    return params, result


def plot_correct_classifications(true_labels, predicted_labels, data):
    """
    Plot the ideal, correctly classified data. Should be entirely blue in ideal case
    """
    # print(data)
    for ii in range(data.shape[0]):    
        # print(true_labels - predicted_labels)
        if indicator(true_labels[ii], predicted_labels[ii]) == 0: 
            #If label is correct for particular point
            plt.scatter(data[ii][0], data[ii][1],  marker="o", color="blue")
        else: 
            plt.scatter(data[ii][0], data[ii][1], marker="x", color="red")
    plt.show()
    return

def compute_number_misclassified(true_labels, noisy_predictions):
    return abs((true_labels-noisy_predictions)).sum()/len(true_labels)

def generate_noisy_classification(ideal_params, noise_type, noise_strength, qc, num_shots, data, true_labels):
    number_points = len(true_labels)
    circ =  ClassificationCircuit(qubits, data)
    circ._add_params(ideal_params)
    circ._encoding(encoding_choice)
    circ._add_class_circuits(noise=noise_type, noise_strength=noise_strength)

    noisy_predictions = circ._make_predictions(num_shots, qc)
   
    number_misclassified = compute_number_misclassified(true_labels, noisy_predictions)

    return noisy_predictions, number_misclassified

def generate_random_data_labels(n_x, n_y):

    data = grid_pts(n_x, n_y)
    true_labels = np.zeros(n_x*n_y, dtype=int)
    
    for ii in range(data.shape[0]):
        if data[ii][0] >= 0.5:
            true_labels[ii] = 1
    return data, true_labels

def plot_number_misclassified_shot_point(ideal_params, noise_type, noise_strength, qc):

    points_data_inc = []
    num_shots = 500
    for n_x in [2*i for i in range(2, 11)]:
        n_y = n_x
        data, true_labels = generate_random_data_labels(n_x, n_y)
        noisy_predictions, number_misclassified = generate_noisy_classification(ideal_params, noise_type, noise_strength, qc, num_shots, data, true_labels)
        points_data_inc.append([n_x*n_y, number_misclassified])
    
    points_data_inc_arr = np.array(points_data_inc)
    print(points_data_inc_arr[:,1])

    data, true_labels = generate_random_data_labels(10, 10)
    points_shots_inc = []

    for num_shots in [20*i for i in range(1, 20)]:
        
        noisy_predictions, number_misclassified = generate_noisy_classification(ideal_params, noise_type, noise_strength, qc, num_shots, data, true_labels)
        points_shots_inc.append([num_shots, number_misclassified])

    points_shot_inc_arr = np.array(points_shots_inc)

    fig, axs = plt.subplots(2)

    axs[0].plot(points_data_inc_arr[: , 0], points_data_inc_arr[: , 1]) 
    axs[1].plot(points_shot_inc_arr[: , 0], points_shot_inc_arr[: , 1])

    plt.show()


def plot_number_misclassified_noise_stren(ideal_params, num_shots, num_points, qc):

    points_noise_inc = []
    
    data, true_labels = generate_random_data_labels(num_points, num_points)
    noise_strengths = np.arange(0, 1, 0.05)

    noise_type = "Pauli_Before_Meas"

    points_noise_stren_inc = []
    for noise_str in noise_strengths:
        p = [1-noise_str, noise_str, 0, 0]
        noisy_predictions, number_misclassified = generate_noisy_classification(ideal_params, noise_type, p, qc, num_shots, data, true_labels)
        points_noise_stren_inc.append([noise_str, number_misclassified])
        print(points_noise_stren_inc)

    points_noise_stren_inc_array = np.array(points_noise_stren_inc)

    fig, axs = plt.subplots(1)

    axs.plot(points_noise_stren_inc_array[:, 0], points_noise_stren_inc_array[:, 1]) 

    plt.show()

# X, y_true = make_moons(n_samples=20, noise=0.0)


qc_name = '1q-qvm'
qc = get_qc(qc_name)
num_shots = 200
qubits = qc.qubits()
init_params = np.random.rand(3)
# init_params = [7.85082205, 0.01934754, 9.62729993]
# init_params = [7.85082205, 0.01934754, 9.0]

encoding_choice = 'denseangle'
optimiser = 'Powell' 
# data, true_labels = generate_random_data_labels(n_x, n_y)

# params, result = train_classifier(qc, num_shots, init_params, encoding_choice, optimiser, data, true_labels)
# print('The optimised parameters are:', result.x)
# print('These give a cost of:', ClassificationCircuit(qubits, data).build_classifier(result.x, encoding_choice, num_shots, qc, true_labels))

## Check ideal data encoding with learned parameters

ideal_params = [1.552728382792277, 2.1669057463233097, -0.013736721729997667]

# circ =  ClassificationCircuit(qubits, data)
# circ._add_params(ideal_params)
# circ._encoding(encoding_choice)
# circ._add_class_circuits()
# ideal_circuit_circ = circ.circuits[0]
# # print(ideal_circuit_circ)

# # ideal_wf = make_wf.wavefunction(ideal_circuit_circ.circuit)
# # print('Ideal WF is:', ideal_wf)
# ideal_predictions = circ._make_predictions(num_shots, qc)
# print(ideal_predictions, 'TRUE:\n',true_labels)
# plot_correct_classifications(true_labels, ideal_predictions, data)


noise = "Pauli_Before_Meas"
# noise = "Depolarizing_Before_Meas"

# p = 0.1
p=  [0.53, 0.20, 0.20, 0.07]
# plot_number_misclassified_shot_point(ideal_params, noise_type=noise, noise_strength=p, qc=qc)
plot_number_misclassified_noise_stren(ideal_params, num_shots=500, num_points=20, qc=qc)
# print('The fraction of misclassified points is:', abs((true_labels-noisy_predictions)).sum()/(n_x*n_y))
