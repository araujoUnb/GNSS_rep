from DelayEstimation import simulation


DoA_deg = 10

# angle_diff_deg_vec = np.array([5, 10, 15])  
# delay_diff_vec     = np.linspace(0, 9, 10)/10
# C_N0_vec           = np.linspace(0, 60, 11)
# xi_vec             = 10**np.linspace(-6, 0, 7)
# epsilon_vec        = np.linspace(0, 9, 10)/10

angle_diff_deg = 5
delay_diff = 0.5
C_N0 = 48
xi = 1e-1
epsilon = 0.5
seed = 965

# for seed in range(900, 910):
#     results = simulation(seed, angle_diff_deg, delay_diff, C_N0, xi, epsilon)
#     print()

# print(results)

import numpy as np

lskrf = [0.00282, 0.00176, 0.00170, 0.00188, 0.00176, 0.00200, 0.00193, 0.00175, 0.00197, 0.00192]
bayesian = [11.86182, 9.99333, 10.45768, 10.45604, 10.99036, 11.02751, 10.11507, 11.46214, 12.07746, 10.77748]
bayesian_ref = [12.07906, 10.52772, 10.67987, 10.89114, 11.18328, 11.30544, 10.30382, 12.14307, 12.26952, 11.06637]

print(np.mean(lskrf))
print(np.mean(bayesian))
print(np.mean(bayesian_ref))