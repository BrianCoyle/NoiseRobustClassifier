import numpy as np
from pyquil.gates import *
from noise_models import add_noisy_gate_to_circ
from noise_models import dephased_rx_gate, dephased_ry_gate, dephased_rz_gate, dephased_i_gate, pauli_noise_i_gate, pauli_noise_z_gate

def ideal_circuit(circuit, params, qubits):
    """
    Writes the ideal classification ansatz circuit.
    For now, just general single qubit unitary decomposed into RxRzRx
    with three parameters
    """
    
    circuit += RX(2*params.values[0][0], qubits[0])
    circuit += RZ(2*params.values[0][1], qubits[0])
    circuit += RX(2*params.values[0][2], qubits[0])

    return circuit  


def dephased_before_meas_circuit(circuit, params, qubits, p):
    """
    Writes the ideal classification ansatz circuit.
    For now, just general single qubit unitary decomposed into RxRzRx
    with three parameters
    """
    
    circuit += RX(2*params.values[0][0], qubits[0])
    circuit += RZ(2*params.values[0][1], qubits[0])
    circuit += RX(2*params.values[0][2], qubits[0])
    circuit += I(qubits[0])
    noisy_i = dephased_i_gate(p)

    circuit = add_noisy_gate_to_circ(circuit, noisy_i, [qubits[0]], 'iden')

    return circuit  


def pauli_before_meas_circuit(circuit, params, qubits, p):
    """
    Writes the ideal classification ansatz circuit.
    For now, just general single qubit unitary decomposed into RxRzRx
    with three parameters
    """
    [p_I, p_x, p_y, p_z] = p
    circuit += RX(2*params.values[0][0], qubits[0])
    circuit += RZ(2*params.values[0][1], qubits[0])
    circuit += RX(2*params.values[0][2], qubits[0])
    circuit += I(qubits[0])
    # circuit += Z(qubits[0])
    noisy_iden = pauli_noise_i_gate(p_I, p_x, p_y, p_z)

    circuit = add_noisy_gate_to_circ(circuit, noisy_iden, [qubits[0]], 'iden')

    return circuit  