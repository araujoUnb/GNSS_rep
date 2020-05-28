import numpy as np
import tensorly as tl
from gnss_func.gnss_function import frequecy_domain_CA, correlator_bank_Q
from gnss_func.array import array_lin
import pandas as pd

import sys
sys.path.extend(['/Users/araujo/Documents/GitHub/GNSS_rep'])

from os import path


class single_polarization():

    def __init__(self, nAntennas, B, T, Tc, delayGranularity, tau_vec, theta_deg_vec, number_of_epochs, IDsat):

        self.tau_vec = tau_vec
        self.theta_deg_vec = theta_deg_vec
        self.number_of_epochs = number_of_epochs
        self.ID = IDsat
        self.bandwidth = B
        self.time_period = T
        self.chip_period = Tc
        self.nAntennas = nAntennas
        self.delayGranularity = delayGranularity


        self.CA_FFT = None
        self.CA_PSD = None

        self.Q = None
        self.C = None
        self.CQ = None

        self.B = None

        self.S = None
        
        
        self.create_output_correlator()

    def bank_delay(self):
        return np.linspace(-self.chip_period,self.chip_period,self.delayGranularity)
    def code_path(self):
        return '/Users/araujo/Documents/GitHub/GNSS_rep/CACODE/CA_FFT_ ' + str(self.ID) + '_' + str(self.bandwidth) + '.pkl'

    def number_of_paths(self):
        return self.tau_vec.size

    def create_output_correlator(self):
        BANK_delay = self.bank_delay()
        folder = self.code_path()
        bool = path.exists(folder)
        if bool == False:
            self.CA_FFT, self.CA_PSD = frequecy_domain_CA(self.bandwidth, self.time_period, self.ID)
        else:
            code_df = pd.read_pickle(folder)
            self.CA_FFT = code_df['CA_FFT']
            self.CA_PSD = code_df['CA_SPEC']

        self.Q, self.C, self.CQ = correlator_bank_Q(self.bandwidth, self.chip_period, self.time_period,
                                                    BANK_delay, self.tau_vec, self.CA_FFT)

    def channel_taps(self):
        n_epochs = self.number_of_epochs
        L = self.number_of_paths()

        self.B = 1 / np.sqrt(2) * (np.random.randn(n_epochs, L) + 1j * np.random.randn(n_epochs, L))

    def array(self):
        self.A = array_lin(self.theta_deg_vec, self.nAntennas)

    def create_signal(self):

        self.channel_taps()
        self.array()

        S0 = self.B @ tl.tenalg.khatri_rao([self.C, self.A]).T

        # Tx-tensor
        self.S = tl.tensor(S0.reshape(self.number_of_epochs, int(self.C.size / self.number_of_paths()), self.nAntennas))

    def rx_signal(self, SNR_dB):
        snr = 10 ** (SNR_dB / 10)
        self.create_signal()

        a, b, c = self.S.shape
        N = tl.tensor(1 / np.sqrt(2 * snr) * (np.random.randn(a, b, c) + 1j * np.random.randn(a, b, c)))
        self.rSignal = tl.tenalg.mode_dot(self.S, self.Q.T, mode=1) + tl.tenalg.mode_dot(N, self.Q.T, mode=1)
