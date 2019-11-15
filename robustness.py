"""Numerical implementations of noise robustness results."""

import matplotlib.pyplot as plt
import numpy as np


# Pauli matrices
imat = np.identity(2)
xmat = np.array([[0, 1], [1, 0]])
ymat = np.array([[0, -1j], [1j, 0]])
zmat = np.array([[1, 0], [0, -1]])

# Projectors
pi0 = (imat + zmat) / 2
pi1 = (imat - zmat) / 2


# Encodings
def state(f, g, x, y):
    rho = np.array([
            [abs(f(x, y))**2, f(x, y) * np.conj(g(x, y))],
            [np.conj(f(x, y)) * g(x, y), abs(g(x, y))**2]
            ])
    return rho / np.trace(rho)

    
def f(x, y):
    return x


def g(x, y):
    return y


def f2(x, y):
    return np.cos(np.pi * x)


def f3(x, y):
    return np.exp(2 * np.pi * 1j * y) * np.sin(np.pi * x)


# Noise channels
def pauli(rho, pi=0.51, px=0.0, py=0.49, pz=0.0):
    """Applies a single qubit pauli channel to the state rho."""
    assert np.isclose(sum((pi, px, py, pz)), 1.0)
    return (pi * rho + 
            px * xmat @ rho @ xmat +
            py * ymat @ rho @ ymat +
            pz * zmat @ rho @ zmat)


# Predictions
def noiseless(rho, unitary):
    """Returns the noiseless predictions."""
    rhotilde = unitary @ rho @ unitary.conj().T
    elt = rhotilde[0, 0]
    if elt >= 0.49999999:
        return 0, elt
    return 1, elt


def noisy(rho, channel, unitary):
    """Returns the noisy prediction with channel acting after the unitary."""
    rhotilde = unitary @ rho @ unitary.conj().T
    rhotilde = channel(rho)
    elt = rhotilde[0, 0]
    if elt >= 0.4999999999999:
        return 0, elt
    return 1, elt


# Simulations
def grid_pts(nx=12, ny=12):
    return np.array([[x, y] for x in np.linspace(-1, 1, nx) 
                     for y in np.linspace(-1, 1, ny)])    


def pauli_robustness():
    points = grid_pts(nx=20, ny=20)
    unitary = np.identity(2)
    
    ys = []
    yhats = []
    elts = []
    elthats = []
    for point in points:
        # Get the state
        rho = state(f, g, *point)
        
        # Noiseless and noisy predictions
        y, elt = noiseless(rho, unitary)
        
        yhat,elthat = noisy(rho, pauli, unitary)
        
        # Store them
        ys.append(y)
        yhats.append(yhat)

        elts.append(elt)
        elthats.append(elthat)

        if y == yhat:
            plt.scatter(*point, marker="o", color="green", s=50)
        else:
            plt.scatter(*point, marker="x", color="red", s=50)
    # print(yhats)
    for ii, y in enumerate(ys):
        if y != yhats[ii]:
            print(y, yhat)
            print(elts[ii], elthats[ii])
    plt.show()
    # print(ys)
    # print(yhats)

if __name__ == "__main__":
    pauli_robustness()
