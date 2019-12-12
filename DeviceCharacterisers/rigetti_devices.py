from pyquil import Program
# from pyquil.noise
# from pyquil import QuantumComputer

from pyquil.api import get_qc
from pyquil.gates import *

import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from file_operations_out import make_dir



class RigettiDevice:

    def __init__(self, qc_name, simulated=True):
        self.qc_name = qc_name

        if simulated:
            self.qc = get_qc(qc_name, as_qvm=True)
        else:
            if qc_name[-3:] == 'qvm':
                raise ValueError("Cannot get a real device for a simulated chip")
            else:
                self.qc = get_qc(qc_name)
        self.qubits = self.qc.qubits()

    def characterise_chip_measurement(self, trials):
        
        qubit_meas_noise = {}
        print(self.qubits)

        for qubit in self.qubits:
            probs_array = self._estimate_meas_probs(qubit, trials)
            qubit_meas_noise[qubit] = np.array([probs_array[0][0], probs_array[1][1]])

        self.meas_noise = qubit_meas_noise

    def meas_probs_to_file(self, trials,  n_runs):
        
        for run in range(n_runs):
            file_name = '%s/Run%i' % (self.qc_name, run)
            make_dir(file_name)
            self.characterise_chip_measurement(1024)
            for qubit in self.qubits:
                np.savetxt('%s/%s' %(file_name, qubit), device.meas_noise[qubit])

    def _estimate_meas_probs(self, q, trials, p0=None):
        """
        Estimate the readout assignment probabilities for a given qubit ``q``.
        The returned matrix is of the form::
                [[p00 p01]
                [p10 p11]]
        :param int q: The index of the qubit.
        :param int trials: The number of samples for each state preparation.
        :param Union[QVMConnection,QPUConnection] cxn: The quantum abstract machine to sample from.
        :param Program p0: A header program to prepend to the state preparation programs.
        :return: The assignment probability matrix
        :rtype: np.array

        NOTE: Version from Pyquil is deprecated
        """
        if p0 is None:  # pragma no coverage
            p0 = Program()
        ro = p0.declare("ro", 'BIT',2)
        print(q)
        pI = p0  + Program(I(q), MEASURE(q, ro[0]))
        pI.wrap_in_numshots_loop(trials)
        results_i = np.sum(self.qc.run(self.qc.compile(pI)))

        pX = p0 +  Program(X(q), MEASURE(q, ro[0]))
        pX.wrap_in_numshots_loop(trials)

        results_x = np.sum(self.qc.run(self.qc.compile(pX)))

        p00 = 1. - results_i / float(trials)
        p11 = results_x / float(trials)

        return np.array([[p00, 1 - p11],
                        [1 - p00, p11]])


n_runs = 10
chip_name = "Aspen-4-2Q-A"
device = RigettiDevice(chip_name, simulated=False)
# device.characterise_chip_measurement(1024)
device.meas_probs_to_file(1024, n_runs)
# print(device.chip_noise)


#     def estimate_meas_noise(self, num_shots, noisy_device=True, meas_probs=np.eye(2), noisy_programs=[]):
#         """
#         This function computes the measurement noise probabilities for 
#         a given device.
#         Args:
#             qc : QuantumComputer
#                 Quantum Computer object to characterise
#             num_shots : Union[int, list:int]
#             noisy_device : bool
#                 True/False value indicating qc is already noisy
#             meas_probs : np.array
#                 if noisy_device==False (i.e. qvm), add measurement noisy to qc according to 
#                 parameters 
#             noisy_programs : list
#                 if noisy_device==False (i.e. qvm), specify programs with noise added

#         Returns:
#             probs : np.ndarray of the measurment probabilities
#         """
#         qubits = qc.qubits()
#         meas_probs = np.zeros((len(qubits), 2, 2))

#         if noisy_device == True:
#             for ii, qubit in enumerate(qubits):
#                 meas_probs[ii] = estimate_assignment_probs(qubit, num_shots, qc, Program())

#         elif noisy_device == False:

#             assert len(noisy_programs) == len(qubits) # Must specify measurement noise on all qubits

#             for ii, qubit in enumerate(qubits):
#                 meas_probs[ii] = estimate_assignment_probs(qubit, num_shots, qc, noisy_programs[ii])

#         return meas_probs


# """
# Module for creating and verifying noisy gate and readout definitions.
# """
# from collections import namedtuple
# from typing import Sequence

# import numpy as np
# import sys

# from pyquil.gates import I, MEASURE, X
# from pyquil.quilbase import Pragma, Gate
# from pyquil.quilatom import MemoryReference, format_parameter

#     INFINITY = float("inf")
#     "Used for infinite coherence times."


#     _KrausModel = namedtuple("_KrausModel", ["gate", "params", "targets", "kraus_ops", "fidelity"])


#     class KrausModel(_KrausModel):
#         """
#         Encapsulate a single gate's noise model.
#         :ivar str gate: The name of the gate.
#         :ivar Sequence[float] params: Optional parameters for the gate.
#         :ivar Sequence[int] targets: The target qubit ids.
#         :ivar Sequence[np.array] kraus_ops: The Kraus operators (must be square complex numpy arrays).
#         :ivar float fidelity: The average gate fidelity associated with the Kraus map relative to the
#             ideal operation.
#         """

#         @staticmethod
#         def unpack_kraus_matrix(m):
#             """
#             Helper to optionally unpack a JSON compatible representation of a complex Kraus matrix.
#             :param Union[list,np.array] m: The representation of a Kraus operator. Either a complex
#                 square matrix (as numpy array or nested lists) or a JSON-able pair of real matrices
#                 (as nested lists) representing the element-wise real and imaginary part of m.
#             :return: A complex square numpy array representing the Kraus operator.
#             :rtype: np.array
#             """
#             m = np.asarray(m, dtype=complex)
#             if m.ndim == 3:
#                 m = m[0] + 1j * m[1]
#             if not m.ndim == 2:  # pragma no coverage
#                 raise ValueError("Need 2d array.")
#             if not m.shape[0] == m.shape[1]:  # pragma no coverage
#                 raise ValueError("Need square matrix.")
#             return m

#         def to_dict(self):
#             """
#             Create a dictionary representation of a KrausModel.
#             For example::
#                 {
#                     "gate": "RX",
#                     "params": np.pi,
#                     "targets": [0],
#                     "kraus_ops": [            # In this example single Kraus op = ideal RX(pi) gate
#                         [[[0,   0],           # element-wise real part of matrix
#                         [0,   0]],
#                         [[0, -1],           # element-wise imaginary part of matrix
#                         [-1, 0]]]
#                     ],
#                     "fidelity": 1.0
#                 }
#             :return: A JSON compatible dictionary representation.
#             :rtype: Dict[str,Any]
#             """
#             res = self._asdict()
#             res['kraus_ops'] = [[k.real.tolist(), k.imag.tolist()] for k in self.kraus_ops]
#             return res

#         @staticmethod
#         def from_dict(d):
#             """
#             Recreate a KrausModel from the dictionary representation.
#             :param dict d: The dictionary representing the KrausModel. See `to_dict` for an
#                 example.
#             :return: The deserialized KrausModel.
#             :rtype: KrausModel
#             """
#             kraus_ops = [KrausModel.unpack_kraus_matrix(k) for k in d['kraus_ops']]
#             return KrausModel(d['gate'], d['params'], d['targets'], kraus_ops, d['fidelity'])

#         def __eq__(self, other):
#             return isinstance(other, KrausModel) and self.to_dict() == other.to_dict()

#         def __neq__(self, other):
#             return not self.__eq__(other)


#     _NoiseModel = namedtuple("_NoiseModel", ["gates", "assignment_probs"])


#     class NoiseModel(_NoiseModel):


#         def to_dict(self):

#             return {
#                 "gates": [km.to_dict() for km in self.gates],
#                 "assignment_probs": {str(qid): a.tolist() for qid, a in self.assignment_probs.items()},
#             }

#         @staticmethod
#         def from_dict(d):
        
#             return NoiseModel(
#                 gates=[KrausModel.from_dict(t) for t in d["gates"]],
#                 assignment_probs={int(qid): np.array(a) for qid, a in d["assignment_probs"].items()},
#             )

#         def gates_by_name(self, name):
       
#             return [g for g in self.gates if g.gate == name]

#         def __eq__(self, other):
#             return isinstance(other, NoiseModel) and self.to_dict() == other.to_dict()

#         def __neq__(self, other):
#             return not self.__eq__(other)


#     def _check_kraus_ops(n, kraus_ops):

#         for k in kraus_ops:
#             if not np.shape(k) == (2 ** n, 2 ** n):
#                 raise ValueError(
#                     "Kraus operators for {0} qubits must have shape {1}x{1}: {2}".format(n, 2 ** n, k))

#         kdk_sum = sum(np.transpose(k).conjugate().dot(k) for k in kraus_ops)
#         if not np.allclose(kdk_sum, np.eye(2 ** n), atol=1e-3):
#             raise ValueError(
#                 "Kraus operator not correctly normalized: sum_j K_j^*K_j == {}".format(kdk_sum))


#     def _create_kraus_pragmas(name, qubit_indices, kraus_ops):
  
#         pragmas = [Pragma("ADD-KRAUS",
#                         [name] + list(qubit_indices),
#                         "({})".format(" ".join(map(format_parameter, np.ravel(k)))))
#                 for k in kraus_ops]
#         return pragmas


#     def append_kraus_to_gate(kraus_ops, gate_matrix):
#         """
#         Follow a gate ``gate_matrix`` by a Kraus map described by ``kraus_ops``.
#         :param list kraus_ops: The Kraus operators.
#         :param numpy.ndarray gate_matrix: The unitary gate.
#         :return: A list of transformed Kraus operators.
#         """
#         return [kj.dot(gate_matrix) for kj in kraus_ops]


#     def pauli_kraus_map(probabilities):

#         if len(probabilities) not in [4, 16]:
#             raise ValueError("Currently we only support one or two qubits, "
#                             "so the provided list of probabilities must have length 4 or 16.")
#         if not np.allclose(sum(probabilities), 1.0, atol=1e-3):
#             raise ValueError("Probabilities must sum to one.")

#         paulis = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]

#         if len(probabilities) == 4:
#             operators = paulis
#         else:
#             operators = np.kron(paulis, paulis)

#         return [coeff * op for coeff, op in zip(np.sqrt(probabilities), operators)]


#     def damping_kraus_map(p=0.10):

#         damping_op = np.sqrt(p) * np.array([[0, 1],
#                                             [0, 0]])

#         residual_kraus = np.diag([1, np.sqrt(1 - p)])
#         return [residual_kraus, damping_op]


#     def dephasing_kraus_map(p=0.10):

#         return [np.sqrt(1 - p) * np.eye(2), np.sqrt(p) * np.diag([1, -1])]


#     def tensor_kraus_maps(k1, k2):

#         return [np.kron(k1j, k2l) for k1j in k1 for k2l in k2]


#     def combine_kraus_maps(k1, k2):

#         return [np.dot(k1j, k2l) for k1j in k1 for k2l in k2]


#     def damping_after_dephasing(T1, T2, gate_time):

#         assert T1 >= 0
#         assert T2 >= 0

#         if T1 != INFINITY:
#             damping = damping_kraus_map(p=1 - np.exp(-float(gate_time) / float(T1)))
#         else:
#             damping = [np.eye(2)]

#         if T2 != INFINITY:
#             gamma_phi = float(gate_time) / float(T2)
#             if T1 != INFINITY:
#                 if T2 > 2 * T1:
#                     raise ValueError("T2 is upper bounded by 2 * T1")
#                 gamma_phi -= float(gate_time) / float(2 * T1)

#             dephasing = dephasing_kraus_map(p=.5 * (1 - np.exp(-2 * gamma_phi)))
#         else:
#             dephasing = [np.eye(2)]
#         return combine_kraus_maps(damping, dephasing)


#     # You can only apply gate-noise to non-parametrized gates or parametrized gates at fixed parameters.
#     NO_NOISE = ["RZ"]
#     ANGLE_TOLERANCE = 1e-10


#     class NoisyGateUndefined(Exception):
#         """Raise when user attempts to use noisy gate outside of currently supported set."""
#         pass


#     def get_noisy_gate(gate_name, params):
  
#         params = tuple(params)
#         if gate_name == "I":
#             assert params == ()
#             return np.eye(2), "NOISY-I"
#         if gate_name == "RX":
#             angle, = params
#             if np.isclose(angle, np.pi / 2, atol=ANGLE_TOLERANCE):
#                 return (np.array([[1, -1j],
#                                 [-1j, 1]]) / np.sqrt(2),
#                         "NOISY-RX-PLUS-90")
#             elif np.isclose(angle, -np.pi / 2, atol=ANGLE_TOLERANCE):
#                 return (np.array([[1, 1j],
#                                 [1j, 1]]) / np.sqrt(2),
#                         "NOISY-RX-MINUS-90")
#             elif np.isclose(angle, np.pi, atol=ANGLE_TOLERANCE):
#                 return (np.array([[0, -1j],
#                                 [-1j, 0]]),
#                         "NOISY-RX-PLUS-180")
#             elif np.isclose(angle, -np.pi, atol=ANGLE_TOLERANCE):
#                 return (np.array([[0, 1j],
#                                 [1j, 0]]),
#                         "NOISY-RX-MINUS-180")
#         elif gate_name == "CZ":
#             assert params == ()
#             return np.diag([1, 1, 1, -1]), "NOISY-CZ"
#         raise NoisyGateUndefined("Undefined gate and params: {}{}\n"
#                                 "Please restrict yourself to I, RX(+/-pi), RX(+/-pi/2), CZ"
#                                 .format(gate_name, params))


#     def _get_program_gates(prog):

#         return sorted({i for i in prog if isinstance(i, Gate)}, key=lambda g: g.out())


#     def _decoherence_noise_model(gates, T1=30e-6, T2=30e-6, gate_time_1q=50e-9,
#                                 gate_time_2q=150e-09, ro_fidelity=0.95):
    
#         all_qubits = set(sum(([t.index for t in g.qubits] for g in gates), []))
#         if isinstance(T1, dict):
#             all_qubits.update(T1.keys())
#         if isinstance(T2, dict):
#             all_qubits.update(T2.keys())
#         if isinstance(ro_fidelity, dict):
#             all_qubits.update(ro_fidelity.keys())

#         if not isinstance(T1, dict):
#             T1 = {q: T1 for q in all_qubits}

#         if not isinstance(T2, dict):
#             T2 = {q: T2 for q in all_qubits}

#         if not isinstance(ro_fidelity, dict):
#             ro_fidelity = {q: ro_fidelity for q in all_qubits}

#         noisy_identities_1q = {
#             q: damping_after_dephasing(T1.get(q, INFINITY), T2.get(q, INFINITY), gate_time_1q)
#             for q in all_qubits
#         }
#         noisy_identities_2q = {
#             q: damping_after_dephasing(T1.get(q, INFINITY), T2.get(q, INFINITY), gate_time_2q)
#             for q in all_qubits
#         }
#         kraus_maps = []
#         for g in gates:
#             targets = tuple(t.index for t in g.qubits)
#             key = (g.name, tuple(g.params))
#             if g.name in NO_NOISE:
#                 continue
#             matrix, _ = get_noisy_gate(g.name, g.params)

#             if len(targets) == 1:
#                 noisy_I = noisy_identities_1q[targets[0]]
#             else:
#                 if len(targets) != 2:
#                     raise ValueError("Noisy gates on more than 2Q not currently supported")

#                 # note this ordering of the tensor factors is necessary due to how the QVM orders
#                 # the wavefunction basis
#                 noisy_I = tensor_kraus_maps(noisy_identities_2q[targets[1]],
#                                             noisy_identities_2q[targets[0]])
#             kraus_maps.append(KrausModel(g.name, tuple(g.params), targets,
#                                         combine_kraus_maps(noisy_I, [matrix]),
#                                         # FIXME (Nik): compute actual avg gate fidelity for this simple
#                                         # noise model
#                                         1.0))
#         aprobs = {}
#         for q, f_ro in ro_fidelity.items():
#             aprobs[q] = np.array([[f_ro, 1. - f_ro],
#                                 [1. - f_ro, f_ro]])

#         return NoiseModel(kraus_maps, aprobs)


#     def decoherence_noise_with_asymmetric_ro(gates: Sequence[Gate], p00=0.975, p11=0.911):

#         noise_model = _decoherence_noise_model(gates)
#         aprobs = np.array([[p00, 1 - p00],
#                         [1 - p11, p11]])
#         aprobs = {q: aprobs for q in noise_model.assignment_probs.keys()}
#         return NoiseModel(noise_model.gates, aprobs)


#     def _noise_model_program_header(noise_model):
  
#         from pyquil.quil import Program
#         p = Program()
#         defgates = set()
#         for k in noise_model.gates:

#             # obtain ideal gate matrix and new, noisy name by looking it up in the NOISY_GATES dict
#             try:
#                 ideal_gate, new_name = get_noisy_gate(k.gate, tuple(k.params))

#                 # if ideal version of gate has not yet been DEFGATE'd, do this
#                 if new_name not in defgates:
#                     p.defgate(new_name, ideal_gate)
#                     defgates.add(new_name)
#             except NoisyGateUndefined:
#                 print("WARNING: Could not find ideal gate definition for gate {}".format(k.gate),
#                     file=sys.stderr)
#                 new_name = k.gate

#             # define noisy version of gate on specific targets
#             p.define_noisy_gate(new_name, k.targets, k.kraus_ops)

#         # define noisy readouts
#         for q, ap in noise_model.assignment_probs.items():
#             p.define_noisy_readout(q, p00=ap[0, 0], p11=ap[1, 1])
#         return p


#     def apply_noise_model(prog, noise_model):
     
#         new_prog = _noise_model_program_header(noise_model)
#         for i in prog:
#             if isinstance(i, Gate):
#                 try:
#                     _, new_name = get_noisy_gate(i.name, tuple(i.params))
#                     new_prog += Gate(new_name, [], i.qubits)
#                 except NoisyGateUndefined:
#                     new_prog += i
#             else:
#                 new_prog += i
#         return new_prog


#     def add_decoherence_noise(prog, T1=30e-6, T2=30e-6, gate_time_1q=50e-9, gate_time_2q=150e-09,
#                             ro_fidelity=0.95):
      
#         gates = _get_program_gates(prog)
#         noise_model = _decoherence_noise_model(
#             gates,
#             T1=T1,
#             T2=T2,
#             gate_time_1q=gate_time_1q,
#             gate_time_2q=gate_time_2q,
#             ro_fidelity=ro_fidelity
#         )
#         return apply_noise_model(prog, noise_model)


#     def _bitstring_probs_by_qubit(p):
      
#         p = np.asarray(p, order="C")
#         num_qubits = int(round(np.log2(p.size)))
#         return p.reshape((2,) * num_qubits)


#     def estimate_bitstring_probs(results):
      
#         nshots, nq = np.shape(results)
#         outcomes = np.array([int("".join(map(str, r)), 2) for r in results])
#         probs = np.histogram(outcomes, bins=np.arange(-.5, 2 ** nq, 1))[0] / float(nshots)
#         return _bitstring_probs_by_qubit(probs)


#     _CHARS = 'klmnopqrstuvwxyzabcdefgh0123456789'


#     def _apply_local_transforms(p, ts):
       
#         p_corrected = _bitstring_probs_by_qubit(p)
#         nq = p_corrected.ndim
#         for idx, trafo_idx in enumerate(ts):

#             # this contraction pattern looks like
#             # 'ij,abcd...jklm...->abcd...iklm...' so it properly applies a "local"
#             # transformation to a single tensor-index without changing the order of
#             # indices
#             einsum_pat = ('ij,' + _CHARS[:idx] + 'j' + _CHARS[idx:nq - 1]
#                         + '->' + _CHARS[:idx] + 'i' + _CHARS[idx:nq - 1])
#             p_corrected = np.einsum(einsum_pat, trafo_idx, p_corrected)

#         return p_corrected

#     def bitstring_probs_to_z_moments(p):

#         zmat = np.array([[1, 1],
#                         [1, -1]])
#         return _apply_local_transforms(p, (zmat for _ in range(p.ndim)))



