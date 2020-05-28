import numpy as np
import tensorly as tl
from scipy.signal import find_peaks
from bayopt.gaussian import MM_BSL
from gnss_func.gnss_function import create_matrix_C
import matplotlib.pyplot as plt


class singlePolModel_estimator:

    def __init__(self, model, delay_granularity, theta_deg_space):
        self.model = model
        self.delay_granularity = delay_granularity
        self.theta_deg_space = theta_deg_space
        self.light_velocity = 299792458

        self.Cbasis = None
        self.delay_Basis = None

        self.tau_los_est = None
        self.beta_tau =None

        self.create_tau_space()
        self.create_delay_basis()

    def create_tau_space(self):
        self.tau_space = np.linspace(-1, 1, 2 * self.delay_granularity + 1) * self.model.chip_period

    def system_bandwidth(self):
        return self.model.bandwidth

    def system_ID(self):
        return self.model.ID

    def system_time_period(self):
        return self.model.time_period

    def system_chip_period(self):
        return self.model.chip_period

    def system_Qw(self):
        return self.model.Qw

    def system_Q(self):
        return self.model.Q

    def system_CA_FFT(self):
        return self.model.CA_FFT

    def create_delay_basis(self):
        self.Cbasis = create_matrix_C(self.system_bandwidth(), self.system_chip_period(), self.system_time_period(),
                                      self.tau_space, self.system_CA_FFT())
        if self.model.correlatorType == 'Qw':
            self.delay_Basis = self.Cbasis.T @ self.system_Qw()
        elif self.model.correlatorType == 'Q':
            self.delay_Basis = self.Cbasis.T @ self.system_Q()

    def sparse_delay_estimation(self, a=0, b=0, c=0, d=0):
        B = self.delay_Basis.T
        Y = tl.unfold(self.model.rSignal, mode=1)
        beta, sigma, error = MM_BSL(B, Y, a, b, c, d)

        return beta

    def rmse(self, tau_los):
        self.beta_tau = self.sparse_delay_estimation()
        idx_peaks = find_peaks(np.fft.fftshift(np.abs(self.beta_tau) ** 2))[0]
        self.tau_los_est = np.abs(self.tau_space[idx_peaks[0]])

        return self.light_velocity * np.abs(tau_los - self.tau_los_est)

    def plot_beta_tau(self):
        plt.plot(self.tau_space, np.fft.fftshift(np.abs(self.beta_tau) ** 2))
        plt.show()
