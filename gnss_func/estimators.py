import numpy as np
import tensorly as tl
from scipy.signal import find_peaks
from bayopt.gaussian import MM_BSL
from gnss_func.array import array_lin
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
        self.tau_est =None

        self.angle_basis = None
        self.beta_angle = None

        self.create_tau_space()
        self.create_delay_basis()

        self.create_angle_basis()

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

    def create_angle_basis(self):
        self.angle_basis = array_lin(self.theta_deg_space, self.model.nAntennas)


    def sparse_delay_estimation(self, a=0, b=0, c=0, d=0):
        B = self.delay_Basis.T
        Y = tl.unfold(self.model.rSignal, mode=1)
        beta, sigma, error = MM_BSL(B, Y, a, b, c, d)
        return beta

    def sparse_delay_angle_estimation(self, a=0, b=0, c=0, d=0):
        B = self.delay_Basis.T
        A = array_lin(self.sparse_angle_estimation(),self.model.nAntennas)
        Y_filtered = tl.tenalg.mode_dot(self.model.rSignal,np.linalg.pinv(A),mode=2)
        Y = tl.unfold(Y_filtered, mode=1)
        beta, sigma, error = MM_BSL(B, Y, a, b, c, d)
        return beta

    def sparse_angle_estimation(self, a=0, b=0, c=0, d=0):
        B = self.angle_basis
        Y = tl.unfold(self.model.rSignal, mode=2)
        R = Y @ np.conj(Y).T
        L, U = np.linalg.eig(R)
        Ru = U[:, 2::] @ np.conj(U[:, 2::]).T
        return 1 / (np.diag(np.conj(self.angle_basis).T @ Ru @ self.angle_basis))

    def theta_estimation(self):
        self.beta_angle = self.sparse_angle_estimation()
        idx_peaks = find_peaks(np.abs(self.beta_angle) ** 2)[0]
        idx_sort = np.flipud(np.argsort(np.abs(self.beta_angle)[idx_peaks]))
        return self.theta_deg_space[idx_peaks[idx_sort[0:2]]]

    def rmse_angle(self, tau_los):
        self.beta_tau = self.sparse_delay_angle_estimation()
        idx_peaks = find_peaks(np.fft.fftshift(np.sum(np.abs(self.beta_tau) ** 2, axis=1)))[0]
        idx_sort = np.flipud(np.argsort(np.fft.fftshift(np.sum(np.abs(self.beta_tau) ** 2, axis=1))[idx_peaks]))
        self.tau_est = np.abs(self.tau_space[idx_peaks[idx_sort]])
        self.tau_los_est = np.abs(self.tau_space[idx_peaks[idx_sort[0]]])

        return self.light_velocity * np.abs(tau_los - self.tau_los_est)

    def rmse(self, tau_los):
        self.beta_tau = self.sparse_delay_estimation()
        idx_peaks = find_peaks(np.fft.fftshift(np.sum(np.abs(self.beta_tau) ** 2, axis=1)))[0]
        idx_sort = np.flipud(np.argsort(np.fft.fftshift(np.sum(np.abs(self.beta_tau) ** 2, axis=1))[idx_peaks]))
        self.tau_est = np.abs(self.tau_space[idx_peaks[idx_sort]])
        self.tau_los_est = np.abs(self.tau_space[idx_peaks[idx_sort[0]]])

        return self.light_velocity * np.abs(tau_los - self.tau_los_est)

    def plot_beta_tau(self):
        self.beta_tau = self.sparse_delay_estimation()
        plt.plot(self.tau_space, np.fft.fftshift(np.sum(np.abs(self.beta_tau) ** 2,axis=1)))
        plt.show()

    def plot_beta_angle(self):
        self.beta_angle = self.sparse_angle_estimation()
        plt.plot(self.theta_deg_space, np.sum(np.abs(self.beta_angle) ** 2,axis=1))
        plt.show()
