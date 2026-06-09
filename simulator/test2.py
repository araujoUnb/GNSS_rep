
import torch as th
import tensorly as tl
from DelayEstimation import gnssSimulSignal, black_box_LS
import numpy as np
import itertools as it

angle_diff_deg = 5
delay_diff = 0.5
C_N0 = 48
xi = 1e-2
epsilon = 0.5
seed = 965

th.manual_seed(seed)
opt_random_state = 1000 + seed

DoA_deg = 10
dataModel = gnssSimulSignal(C_N0)

Y, X, factors, Qw, CQ, CQw, Qw, OMEGA = dataModel.signalModel3(DoA_deg, angle_diff_deg, delay_diff, epsilon)

## pre-processing
PIm = th.fliplr(th.eye(dataModel.m, dtype=th.complex64))
Y0 = tl.unfold(Y, 0)
Y0conj = PIm @ th.conj(Y0)
Z = th.cat((Y0, Y0conj), 1)

l_s = 5  # number of sub-arrays
m_s = dataModel.m - l_s + 1  # resulting array size

#W = th.zeros([m_s,dataModel.k*dataModel.q*l_s],dtype = th.complex64)
E = th.zeros([m_s, 2 * dataModel.k * dataModel.q * l_s], dtype=th.complex64)

for vv in range(l_s - 1):
    #W[:,dataModel.k*dataModel.q*vv:dataModel.k*dataModel.q*(vv+1)] = Y0[vv:(m_s+vv),:]
    E[:, 2 * dataModel.k * dataModel.q * vv:2 * dataModel.k * dataModel.q * (vv + 1)] = Z[vv:(m_s + vv), :]

tau_real = dataModel.tau_vec[0]

Y2 = tl.unfold(Y, 2)

black_box_LSBayOpt = lambda tauLos, tauNLos: black_box_LS(Y2, dataModel, tauLos, tauNLos)

tau_los_limit = dataModel.Tc * 0.3

tau_values_los = np.linspace(-tau_los_limit, tau_los_limit, 101)
tau_values_nlos = np.linspace(-tau_los_limit, dataModel.Tc, 101)

x = np.zeros((tau_values_los.size, tau_values_nlos.size))
for idx1, tauLos in enumerate(tau_values_los):
    print(idx1)
    for idx2, tauNLos in enumerate(tau_values_nlos):
        x[idx1, idx2] = black_box_LSBayOpt(tauLos, tauNLos)

import shelve
with shelve.open(f'teste4.dat', 'n') as f:
    f['Tc'] = dataModel.Tc
    f['tauLos_vec'] = tau_values_los
    f['tauNLos_vec'] = tau_values_nlos
    f['x'] = x
