import numpy as np
from pyquil import Program
from pyquil.noise import append_kraus_to_gate, pauli_kraus_map, dephasing_kraus_map, damping_kraus_map

from pyquil.quilatom import quil_cos, quil_sin, MemoryReference
from pyquil.gates import *

def dephased_rx_gate(phi, p=.1):
    corrupted_rx = append_kraus_to_gate(dephasing_kraus_map(p),
    np.array([[np.cos(phi / 2), -1j * np.sin(phi / 2)], [-1j * np.sin(phi / 2), np.cos(phi / 2)]]))
    return corrupted_rx

def dephased_rz_gate(phi, p=.1):
    corrupted_rz = append_kraus_to_gate(dephasing_kraus_map(p),
    np.array([[quil_cos(phi / 2) - 1j * np.sin(phi / 2), 0], [0, np.cos(phi / 2) + 1j * np.sin(phi / 2)]]))
    return corrupted_rz

def dephased_ry_gate(phi, p=.1):
    corrupted_ry = append_kraus_to_gate(dephasing_kraus_map(p),
    np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]]))
    return corrupted_ry

def dephased_i_gate(p=.1):
    corrupted_iden = append_kraus_to_gate(dephasing_kraus_map(p),
    np.array([[1, 0], [0, 1]]))
    return corrupted_iden

def pauli_noise_i_gate(p_I=.7, p_x=0.1, p_y=0.1, p_z=0.1):
    corrupted_iden = append_kraus_to_gate(pauli_kraus_map([p_I, p_x, p_y, p_z]),
    np.array([[1, 0], [0, 1]]))
    return corrupted_iden

def pauli_noise_z_gate(p_I=.7, p_x=0.1, p_y=0.1, p_z=0.1):
    corrupted_Z = append_kraus_to_gate(pauli_kraus_map([p_I, p_x, p_y, p_z]),
    np.array([[1, 0], [0, -1]]))
    return corrupted_Z


def amp_damp_i_gate(p=.1):
    """
    Create amplitude damped Identity gate
    """
    corrupted_i = append_kraus_to_gate(damping_kraus_map(p), np.array([[1, 0], [0, 1]]))
    return corrupted_i

def add_noisy_gate_to_circ(circuit, corrupted_gate, qubit, gate_type):
    
    if gate_type.lower() == 'rz':
        circuit.define_noisy_gate("RZ", qubit, corrupted_gate)

    elif gate_type.lower() == 'rx':
        circuit.define_noisy_gate("RX", qubit, corrupted_gate)
    
    elif gate_type.lower() == 'ry':
        circuit.define_noisy_gate("RY", qubit, corrupted_gate)
    
    elif gate_type.lower() == 'iden':
        circuit.define_noisy_gate("I", qubit, corrupted_gate)
         
    return circuit

def estimate_assignment_probs(q, trials, qc, p0=None):
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
    from pyquil.quil import Program
    if p0 is None:  # pragma no coverage
        p0 = Program()
    ro = p0.declare("ro", 'BIT', 1)

    pI = p0  + Program(I(q), MEASURE(q, ro[0]))
    pI.wrap_in_numshots_loop(trials)
    results_i = np.sum(qc.run(qc.compile(pI)))

    pX = p0 +  Program(X(q), MEASURE(q, ro[0]))
    pX.wrap_in_numshots_loop(trials)

    results_x = np.sum(qc.run(qc.compile(pX)))

    p00 = 1. - results_i / float(trials)
    p11 = results_x / float(trials)

    return np.array([[p00, 1 - p11],
                     [1 - p00, p11]])


def estimate_meas_noise(qc, num_shots, noisy_device=True, meas_probs=np.eye(2), noisy_programs=[]):
    """
    This function computes the measurement noise probabilities for 
    a given device.
    Args:
        qc : QuantumComputer
            Quantum Computer object to characterise
        num_shots : Union[int, list:int]
        noisy_device : bool
            True/False value indicating qc is already noisy
        meas_probs : np.array
            if noisy_device==False (i.e. qvm), add measurement noisy to qc according to 
            parameters 
        noisy_programs : list
            if noisy_device==False (i.e. qvm), specify programs with noise added

    Returns:
        probs : np.ndarray of the measurment probabilities
    """
    qubits = qc.qubits()
    meas_probs = np.zeros((len(qubits), 2, 2))

    if noisy_device == True:
        for ii, qubit in enumerate(qubits):
            meas_probs[ii] = estimate_assignment_probs(qubit, num_shots, qc, Program())

    elif noisy_device == False:

        assert len(noisy_programs) == len(qubits) # Must specify measurement noise on all qubits

        for ii, qubit in enumerate(qubits):
            meas_probs[ii] = estimate_assignment_probs(qubit, num_shots, qc, noisy_programs[ii])

    return meas_probs

