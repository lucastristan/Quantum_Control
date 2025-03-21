import numpy as np
import matplotlib.pyplot as plt
from qutip import *  # QuTiP library for quantum mechanics simulations

# Parameters
omega = 1.0  # Frequency of the Hamiltonian
gamma = 0.1  # Measurement strength (reduced for smoother trajectories)
T = 10.0     # Total simulation time
N = 2000     # Increased number of time steps
dt = T / N   # Time step size

# Pauli matrices
sigma_x = sigmax()
sigma_y = sigmay()
sigma_z = sigmaz()

# Hamiltonian
H = 0.5 * omega * sigma_z

# Measurement operator
C = np.sqrt(gamma) * sigma_x

# Initial state (pure state |0>)
rho0 = ket2dm(basis(2, 0))  # Density matrix for |0><0|

# Stochastic master equation (SME) function
def sme(t, rho):
    # Deterministic part
    deterministic = -1j * (H * rho - rho * H) + gamma * (sigma_x * rho * sigma_x - 0.5 * (sigma_x**2 * rho + rho * sigma_x**2))
    # Stochastic part
    dw = np.random.normal(0, np.sqrt(dt))  # Wiener increment
    stochastic = np.sqrt(gamma) * (sigma_x * rho + rho * sigma_x - 2 * expect(sigma_x, rho) * rho) * dw
    return deterministic * dt + stochastic

# Simulation
times = np.linspace(0, T, N)
result = [rho0]  # List to store states
for t in times[1:]:
    rho = result[-1]  # Current state
    rho_new = rho + sme(t, rho)  # Evolve state using SME
    
    # Ensure the density matrix remains valid
    rho_new = 0.5 * (rho_new + rho_new.dag())  # Enforce Hermiticity
    rho_new = rho_new / rho_new.tr()  # Normalize to ensure trace = 1
    
    result.append(rho_new)  # Append new state

# Extract Bloch vectors
bloch_vectors = np.array([[expect(sigma_x, rho), expect(sigma_y, rho), expect(sigma_z, rho)] for rho in result])

# Reshape bloch_vectors to the correct format (3, N)
bloch_vectors = bloch_vectors.T  # Transpose to get shape (3, N)

# Plot on the Bloch sphere
b = Bloch()
b.point_color = ['r']  # Color of the trajectory
b.add_vectors([bloch_vectors[:, 0], bloch_vectors[:, -1]])  # Initial and final vectors
b.add_points(bloch_vectors)  # Add trajectory
b.show()  # Display the Bloch sphere
