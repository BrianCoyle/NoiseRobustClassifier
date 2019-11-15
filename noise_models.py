import numpy as np
from pyquil.noise import append_kraus_to_gate, pauli_kraus_map
from pyquil.quilatom import quil_cos, quil_sin

def dephasing_kraus_map(p=.1):
    """
    Generate the Kraus operators corresponding to a dephasing channel.

    :params float p: The one-step dephasing probability.
    :return: A list [k1, k2] of the Kraus operators that parametrize the map.
    :rtype: list
    """
    return [np.sqrt(1-p)*np.eye(2), np.sqrt(p)*np.diag([1, -1])]

# def dephased_rx_gate(phi, p=.1):
#     corrupted_rx = append_kraus_to_gate(dephasing_kraus_map(p),
#     np.array([[quil_cos(phi / 2), -1j * quil_sin(phi / 2)], [-1j * quil_sin(phi / 2), quil_cos(phi / 2)]]))
#     return corrupted_rx

# def dephased_rz_gate(phi, p=.1):
#     corrupted_gate = append_kraus_to_gate(dephasing_kraus_map(p),
#     np.array([[quil_cos(phi / 2) - 1j * quil_sin(phi / 2), 0], [0, quil_cos(phi / 2) + 1j * quil_sin(phi / 2)]]))
#     return corrupted_rz

# def dephased_ry_gate(phi, p=.1):
#     corrupted_gate = append_kraus_to_gate(dephasing_kraus_map(p),
#     np.array([[quil_cos(phi / 2), -quil_sin(phi / 2)], [quil_sin(phi / 2), quil_cos(phi / 2)]]))
#     return corrupted_ry

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