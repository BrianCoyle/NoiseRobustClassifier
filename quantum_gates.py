import numpy as np
from numpy import sin, cos
'''
    Define quantum gates used for fidelity analysis
'''

# Rotation matrices
def rz(param):
    return np.array([[np.cos(param / 2) - 1j * np.sin(param / 2), 0],
                    [0, np.cos(param / 2) + 1j * np.sin(param / 2)]] )

def ry(param):
    return np.array( [[cos(param / 2), -sin(param / 2)],
                    [sin(param / 2), cos(param / 2)]] )

def rx(param):
    return np.array( [[cos(param / 2), -1j * sin(param / 2)],
                    [-1j * sin(param / 2), cos(param / 2)]] )

# Pauli matrices
imat = np.identity(2)
xmat =  np.array([[0, 1], [1, 0]])
ymat = np.array([[0, -1j], [1j, 0]])
zmat = np.array([[1, 0], [0, -1]])

# Projectors
pi0 = (imat + zmat) / 2
pi1 = (imat - zmat) / 2

# Hadamard
hmat = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]])

# CNOT
cnotmat = np.array([[1, 0, 0, 0],\
                    [0, 1, 0, 0],\
                    [0, 0, 0, 1],\
                    [0, 0, 1, 0]])