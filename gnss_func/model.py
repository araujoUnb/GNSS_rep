import numpy as np
import tensorly as tl
from gnss_func.gnss_function import frequecy_domain_CA, correlator_bank_Q
from gnss_func.array import array_lin
from gnss_func.utils import normalise_columns
import pandas as pd
from sklearn.decomposition import TruncatedSVD
import sys

sys.path.extend(['/Users/araujo/Documents/GitHub/GNSS_rep'])

from os import path


class single_polarization():

    def __init__(self, nAntennas, B, T, Tc, delayGranularity, tau_vec, theta_deg_vec, number_of_epochs, IDsat,
                 correlatorType='Qw'):

        self.tau_vec = tau_vec
        self.theta_deg_vec = theta_deg_vec
        self.number_of_epochs = number_of_epochs
        self.ID = IDsat
        self.bandwidth = B
        self.time_period = T
        self.chip_period = Tc
        self.nAntennas = nAntennas
        self.delayGranularity = delayGranularity
        self.correlatorType = correlatorType

        self.CA_FFT = None
        self.CA_PSD = None
        self.tx_power = None
        self.Rnoise = None

        self.Q = None
        self.Qw = None
        self.C = None
        self.CQ = None
        self.Lnoise = None

        self.B = None

        self.S = None

        self.create_output_correlator()
        self.cov_matrix_noise_mode_2()

    def bank_delay(self):
        return np.linspace(-self.chip_period, self.chip_period, 2*self.delayGranularity)

    def code_path(self):
        return '/Users/araujo/Documents/GitHub/GNSS_rep/CACODE/CA_FFT_ ' + str(self.ID) + '_' + str(
            self.bandwidth) + '.pkl'

    def number_of_paths(self):
        return self.tau_vec.size

    def cov_matrix_noise_mode_2(self):
        if self.correlatorType == 'Q':
            self.Rnoise = np.conj(self.Q.T) @ self.Q
        elif self.correlatorType == 'Qw':
            self.Rnoise = np.conj(self.Qw.T) @ self.Qw

        self.Lnoise = np.linalg.cholesky(self.Rnoise)

    def calc_snr_pre(self, C_N_dB):
        return C_N_dB - 10 * np.log10(2 * self.bandwidth)

    def noise_var(self, C_N_dB):
        SNR_dB = self.calc_snr_pre(C_N_dB)
        snr = 10 ** (SNR_dB / 10)
        return self.tx_power / snr

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

        self.Q, self.C, self.CQ, self.tx_power = correlator_bank_Q(self.bandwidth, self.chip_period, self.time_period,
                                                                   BANK_delay, self.tau_vec, self.CA_FFT)

        if self.correlatorType == 'Qw':
            self.create_Qw()
            self.CQ = self.C.T @ self.Qw

    def create_Qw(self,n=7):
        svd = TruncatedSVD(n_components=n, n_iter=7, random_state=59, algorithm='arpack')
        svd.fit(self.Q)
        self.Qw = normalise_columns(svd.fit_transform(self.Q))

    def channel_taps(self):
        n_epochs = self.number_of_epochs
        L = self.number_of_paths()

        self.B = 1 / np.sqrt(2) * (np.random.randn(n_epochs, L) + 1j * np.random.randn(n_epochs, L))

    def array(self):
        self.A = array_lin(self.theta_deg_vec, self.nAntennas)

    def create_signal(self):

        self.channel_taps()
        self.array()

        S0 = self.B @ tl.tenalg.khatri_rao([self.CQ.T, self.A]).T

        # Tx-tensor
        self.S = tl.tensor(
            S0.reshape(self.number_of_epochs, int(self.CQ.size / self.number_of_paths()), self.nAntennas))

    def create_Noise(self, sigma, a, b, c):

        N2 = 1 / np.sqrt(2 * sigma) * (np.random.randn(b, a * c) + 1j * np.random.randn(b, a * c))
        N2 = self.Lnoise @ N2

        return tl.tensor(N2.reshape(a, b, c))

    def rx_signal(self, C_N_dB):

        sigma = self.noise_var(C_N_dB)
        self.create_signal()

        a, b, c = self.S.shape

        self.rSignal = self.S + self.create_Noise(sigma, a, b, c)
