
import numpy as np

def array_lin(theta_deg, n_antennas):
    theta = np.pi * theta_deg / 180
    idx_antennas = np.linspace(0, n_antennas - 1, n_antennas) - n_antennas / 2 + 0.5
    phase = np.pi * np.outer(idx_antennas, np.cos(theta))
    return 1 / np.sqrt(n_antennas) * np.exp(1j * phase)