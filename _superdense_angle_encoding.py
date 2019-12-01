
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from nisqai.layer._base_ansatz import BaseAnsatz
from nisqai.data._cdata import CData, LabeledCData

from numpy import array, cos, sin, exp, dot, identity, isclose
from pyquil import Program

# Changes from original:
# Extra argument in encoder
class SuperDenseAngleEncoding:
    """SuperDenseAngleEncoding class. Encode features into the angles of qubits via

    |\psi> = cos(\theta * x_1 + \phi * x_2) |0> +  sin(\theta *x_1 + \phi * x_2) |1>.


    With this encoding, n features require ceiling(n / 2) qubits.

    Args:
        data : nisqai.data.CData or nisqai.data.LabeledCData
            Data object to be encoded in the circuit.

        encoder : callable
            Function of the features...

        feature_map : callable or iterable
            Defines which features get encoded in which qubits.

            The first two features are mapped to the first qubit, the next two
            features are mapped to the second qubit, etc., unless a feature map
            is given, in which case features are mapped to qubit as specified by
            the feature map.
    """

    def __init__(self, data, encoder, encoding_params, feature_map):
        """Initialize a DenseAngleEncoding class."""
        # TODO: replace with better error checking
        assert isinstance(data, (CData, LabeledCData))
        self.data = data

        # determine the number of qubits from the input data
        num_qubits = self._compute_num_qubits()
        self.encoder = encoder
        self.encoding_params = encoding_params
        self.feature_map = feature_map
        # list to hold circuits for each data point, initialized to none
        self.circuits = [BaseAnsatz(num_qubits) for _ in range(self.data.num_samples)]

        # write all circuits
        for ind in range(len(self.circuits)):
            self._write_circuit(ind)

    def _compute_num_qubits(self):
        """Computes the number of qubits needed for the circuit
        from the input data.
        """
        return self.data.num_features // 2 + self.data.num_features % 2

    # TODO: make the angles parameters and store one circuit, instantiating the parameters
    def _write_circuit(self, feature_vector_index):
        """Writes the encoding circuit into self.circuit."""
        # grab the feature vector to create a circuit with
        feature_vector = self.data.data[feature_vector_index]

        # program to write
        prog = Program()

        # ===============================
        # collect features for each qubit
        # ===============================
        # example: for nearest_neighbor with linear encoding
        # qubit_features[0] = [feature_vector[0], feature_vector[1]]
        # qubit_features[1] = [feature_vector[2], feature_vector[3]]
        # etc.

        qubit_features = {}
        for ind in range(len(self.feature_map.map)):
            # temp list for all features
            features = []
            for x in self.feature_map.map[ind]:
                features.append(feature_vector[x])
            qubit_features[ind] = features

        # ==============================================================
        # use the encoder to get angles from the features for each qubit
        # ==============================================================

        qubit_angles = {}
        for (qubit_index, feature) in qubit_features.items():
            qubit_angles[qubit_index] = self.encoder(feature, self.encoding_params)

        # ============================
        # get matrices from the angles
        # ============================

        qubit_state_preps = {}
        for (qubit_index, angles) in qubit_angles.items():
            qubit_state_preps[qubit_index] = angles_to_matrix(angles)

        # ==================================
        # use each matrix to write a circuit
        # ==================================

        for (qubit_index, mat) in qubit_state_preps.items():
            # define the gate
            name = "S" + str(qubit_index)
            prog.defgate(name, mat)
            # write the gate into the circuit
            prog += (name, qubit_index)

        # write the program into the circuit of the ansatz
        self.circuits[feature_vector_index].circuit = prog

    def __getitem__(self, ind):
        """Returns the circuit for the data point indexed by ind."""
        assert isinstance(ind, int)
        return self.circuits[ind]


def angles_to_matrix(angles):
    """Converts a two element feature vector to a matrix
    preparing the feature vector from the ground state."""
    # grab the angles
    # TODO: check that theta and phi are within the correct range
    # TODO: allow for just one angle (odd number of features case)
    theta = angles[0]
    phi = angles[1]
    # theta and phi are the same in this case.

    # form the matrix
    mat = array([[cos(theta), -sin(phi)],
                 [sin(phi), cos(theta)]])

    # TODO: better error checking
    assert isclose(dot(mat, mat.conj().T), identity(mat.shape[0])).all()

    return mat
