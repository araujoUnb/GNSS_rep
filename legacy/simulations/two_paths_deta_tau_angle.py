import numpy as np
import pandas as pd
from gnss_func.model import single_polarization
from gnss_func.estimators import singlePolModel_estimator

from os import path

MC = 100
#%%

nAntennas = 8
B = 1023e6
Tc = 1/B
T  = 1e-3
IDsat = 20
delta_phi = 60

number_of_epochs = 30
delayGranularity = 11

nColumns = 11
factor = np.linspace(0,1,nColumns)

Folder = 'deltaTau_angle/'
CN = 48


# Estimator parameter
theta_deg_space = np.linspace(35,75,100)

for ii in range(nColumns):
    print('Point:',ii)
    for jj in range(MC):

        pars = np.array([CN, ii, jj]).astype('str')
        file_results = Folder + pars[0] + '_' + pars[1] + '_' + pars[2] + '.pkl'

        bool = path.exists(file_results)

        if bool ==False:

            tau_los = np.random.rand(1).item() * Tc
            tau_nLos = tau_los + factor[ii]*Tc
            tau_vec = np.array([tau_los, tau_nLos])

            theta_los = np.random.rand(1).item()*360

            theta_deg_vec = np.array([theta_los, theta_los + delta_phi])

            gnss_model = single_polarization(nAntennas, B, T, Tc, delayGranularity, tau_vec, theta_deg_vec,
                                             number_of_epochs, IDsat, 'Qw')

            gnss_model.rx_signal(CN)

            estimator_model = singlePolModel_estimator(gnss_model, delayGranularity, theta_deg_space)
            rmse = estimator_model.rmse_angle(tau_los)

            results_dic = {'rmse':[rmse],
                           'tau_los':[tau_los],
                           'tau_nlos':[tau_nLos],
                           'tau_los_est':[estimator_model.tau_los_est],
                           'theta_los': [theta_los],
                           'theta_nlos': [theta_los + delta_phi],
                           }

            results_df = pd.DataFrame(data=results_dic)
            results_df.to_pickle(file_results)




