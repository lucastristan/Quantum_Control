import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Use LaTeX fonts in plots with larger font sizes
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

# Define system parameters
m = 1.0  # Mass
omega0 = 1.0  # Natural frequency
gamma = 0.1  # Damping coefficient

# Compute numerical values of P
term1 = 1 + 1 / (m**2 * omega0**4)  # Common term

# Compute P_12
P_12 = m * omega0**2 * (-1 + np.sqrt(term1))

# Compute P_22
term2 = gamma**2 + 2 * omega0**2 * (-1 + np.sqrt(term1)) + 1
P_22 = -gamma + np.sqrt(term2)

# Compute P_11
P_11 = m**2 * omega0**2 * (-gamma + np.sqrt(term2 * term1))

# Assemble the P matrix
P = np.array([[P_11, P_12], [P_12, P_22]])

# Print the P matrix
print("P matrix:")
print(P)

# Define matrices
A = np.array([[0, 1/m], [-m * omega0**2, -gamma]])
B = np.array([[0], [1]])

# Compute closed-loop system matrix
K = np.dot(B.T, P)  # Optimal feedback gain
A_cl = A - np.dot(B, K)  # Closed-loop system matrix

# Define system dynamics
def closed_loop_dynamics(t, x):
    return A_cl @ x

def open_loop_dynamics(t, x):
    return A @ x

# Initial condition
x0 = np.array([1.0, 0.0])  # Initial position and velocity

# Time span
t_span = (0, 10)
t_eval = np.linspace(*t_span, 1000)

# Solve the closed-loop system
sol_cl = solve_ivp(closed_loop_dynamics, t_span, x0, t_eval=t_eval)

# Solve the open-loop system
sol_ol = solve_ivp(open_loop_dynamics, t_span, x0, t_eval=t_eval)

# Plot results
plt.figure(figsize=(12, 6))

# Plot closed-loop response
plt.plot(sol_cl.t, sol_cl.y[0, :], label=r"Closed-loop Position $x_1(t)$", color="blue", linewidth=2)
plt.plot(sol_cl.t, sol_cl.y[1, :], label=r"Closed-loop Velocity $x_2(t)$", linestyle="dashed", color="blue", linewidth=2)

# Plot open-loop response
plt.plot(sol_ol.t, sol_ol.y[0, :], label=r"Open-loop Position $x_1(t)$", color="red", linewidth=2)
plt.plot(sol_ol.t, sol_ol.y[1, :], label=r"Open-loop Velocity $x_2(t)$", linestyle="dashed", color="red", linewidth=2)

plt.xlabel(r"Time $t$ [s]", fontsize=18)
plt.ylabel(r"State variables $x(t)$", fontsize=18)
plt.legend()
plt.grid()
plt.show()