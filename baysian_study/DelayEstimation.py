import torch as th
from numpy import pi, log10
from math import sqrt
import tensorly as tl
from tensorly.tenalg import khatri_rao
from scipy import interpolate
from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
import numpy as np
from scipy.optimize import minimize
import itertools as it

import warnings

warnings.filterwarnings("ignore")

tl.set_backend('pytorch')


def array_lin(theta_deg, n_antennas):
    theta = pi * th.tensor(theta_deg) / 180
    idx_antennas = th.linspace(0, n_antennas - 1, n_antennas) - n_antennas / 2 + 0.5
    phase = th.pi * th.outer(idx_antennas, th.sin(theta))
    return 1 / sqrt(n_antennas) * th.exp(1j * phase)


def array_lin_noisePhase(theta_deg, n_antennas, epsilon):
    theta = pi * th.tensor(theta_deg) / 180
    idx_antennas = th.linspace(0, n_antennas - 1, n_antennas) - n_antennas / 2 + 0.5
    phase_noise = (th.rand(n_antennas) - 0.5) * epsilon
    phase = th.pi * th.outer(idx_antennas, th.sin(theta))
    return 1 / sqrt(n_antennas) * th.exp(1j * phase) * th.exp(1j * th.pi * th.outer(phase_noise, th.sin(theta)))


def array_lin_noise(theta_deg, n_antennas, epsilon):
    return array_lin(
        theta_deg,
        n_antennas) + sqrt(epsilon / 2) * th.exp(1j * th.rand(array_lin(theta_deg, n_antennas).size()) * 2 * pi)


def shift(register, feedback, output):
    """GPS Shift Register

    :param list feedback: which positions to use as feedback (1 indexed)
    :param list output: which positions are output (1 indexed)
    :returns output of shift register:

    """

    # calculate output
    out = [register[i - 1] for i in output]
    if len(out) > 1:
        out = sum(out) % 2
    else:
        out = out[0]

    # modulo 2 add feedback
    fb = sum([register[i - 1] for i in feedback]) % 2

    # shift to the right
    for i in reversed(range(len(register[1:]))):
        register[i + 1] = register[i]

    # put feedback in position 1
    register[0] = fb

    return out


def PRN(sv):
    """Build the CA code (PRN) for a given satellite ID

    :param int sv: satellite code (1-32)
    :returns list: ca code for chosen satellite

    """

    SV = {
        1: [2, 6],
        2: [3, 7],
        3: [4, 8],
        4: [5, 9],
        5: [1, 9],
        6: [2, 10],
        7: [1, 8],
        8: [2, 9],
        9: [3, 10],
        10: [2, 3],
        11: [3, 4],
        12: [5, 6],
        13: [6, 7],
        14: [7, 8],
        15: [8, 9],
        16: [9, 10],
        17: [1, 4],
        18: [2, 5],
        19: [3, 6],
        20: [4, 7],
        21: [5, 8],
        22: [6, 9],
        23: [1, 3],
        24: [4, 6],
        25: [5, 7],
        26: [6, 8],
        27: [7, 9],
        28: [8, 10],
        29: [1, 6],
        30: [2, 7],
        31: [3, 8],
        32: [4, 9],
    }

    # init registers
    G1 = [1 for i in range(10)]
    G2 = [1 for i in range(10)]

    ca = []  # stuff output in here

    # create sequence
    for i in range(1023):
        g1 = shift(G1, [3, 10], [10])
        g2 = shift(G2, [2, 3, 6, 8, 9, 10], SV[sv])  # <- sat chosen here from table

        # modulo 2 add and append to the code
        ca.append((g1 + g2) % 2)

    # return C/A code!
    return -th.sign(th.tensor(ca) - 0.5)


def create_matrix_C(B, Tc, T, delay, CA_FFT):
    N = 2 * B * T  # number of samples per frame
    f0 = 2 * B / N  # basis frequency
    samples = f0 * (th.linspace(0, int(N) - 1, int(N)) - (N / 2 - 1 / 2))

    PULSE_FFT = th.fft.fftshift(sqrt(Tc * th.sinc(samples * Tc)**2))  # GPS C/A code rectangular pulse

    T_C = th.fft.fftshift(th.exp(-1j * 2 * pi * th.outer(samples, delay)), 0)

    Xc = th.outer(PULSE_FFT * CA_FFT, th.ones(delay.size()[0]))

    C = th.real(th.fft.ifft(T_C * Xc))
    C = th.fft.fftshift(C, 0)
    C = sqrt(N) * C / th.norm(C[:, 0])
    return C


def frequecy_domain_CA(B, T, SAT):
    # B  -> Bandwidth
    # T  -> frame period
    # SAT -> ID
    N = 2 * B * T  # number of samples per frame
    # f0 = 2 * B / N # basis frequency
    # samples = f0 * (np.linspace(0, int(N) - 1, int(N)) - (N / 2 - 1 / 2))

    satcode = PRN(SAT)
    ca_code_length = satcode.size()[0]

    number_of_sequeces = int(N / ca_code_length)
    CA_fft = number_of_sequeces * th.kron(th.ones(number_of_sequeces), th.fft.fft(satcode))

    # Autocorrelation function
    Ra_code = th.zeros(ca_code_length)
    for ii in range(ca_code_length):
        sat_shift = th.roll(satcode, -ii)
        Ra_code[ii] = sat_shift @ satcode / ca_code_length

    # Fourier transmform to obtain Power spetracl density of the sequence

    CA_SPEC = th.fft.fftshift(number_of_sequeces * th.kron(th.ones(number_of_sequeces), th.fft.fft(Ra_code)))

    return CA_fft, CA_SPEC


def gen_delayed_signal(B, T, Tc, CA_FFT, delay):

    N = 2 * B * T  # number of samples per frame
    f0 = 2 * B / N  # basis frequency
    samples = f0 * (th.linspace(0, int(N) - 1, int(N)) - (N / 2 - 1 / 2))

    PULSE_SPEC = th.fft.fftshift(Tc * th.sinc(samples * Tc)**2)  # GPS C/A code rectangular pulse
    PULSE_FFT = th.fft.fftshift(sqrt(Tc) * th.sinc(samples * Tc)**2)  # GPS C/A code rectangular pulse

    #SIGNAL_SPEC = PULSE_SPEC * CA_SPEC

    T_Q = th.fft.fftshift(th.exp(-1j * 2 * pi * th.kron(samples.reshape(samples.size()[0], 1), delay.T)), 0)

    # Back to time domain
    X = th.outer(PULSE_FFT * CA_FFT, th.ones(delay.size))

    Q = th.fft.ifft(T_Q * X)

    return th.sum(Q, axis=1)


def correlator_bank_Q(B, Tc, T, BANK_delay, delay, CA_FFT):
    N = 2 * B * T  # number of samples per frame
    f0 = 2 * B / N  # basis frequency
    samples = f0 * (th.linspace(0, int(N) - 1, int(N)) - (N / 2))

    PULSE_SPEC = th.fft.fftshift(Tc * th.sinc(samples * Tc)**2)  # GPS C/A code rectangular pulse
    PULSE_FFT = th.fft.fftshift(sqrt(Tc) * th.sinc(samples * Tc)**2)  # GPS C/A code rectangular pulse

    #SIGNAL_SPEC = PULSE_SPEC * CA_SPEC

    T_Q = th.exp(-1j * 2 * pi * th.outer(samples, BANK_delay))
    T_Q = th.fft.fftshift(T_Q, 0)

    # Back to time domain
    X = th.outer(PULSE_FFT * CA_FFT, th.ones(BANK_delay.size()[0]))
    Q = th.fft.ifft(T_Q * X, len(samples), 0)
    #Q = np.fft.fftshift(Q, 0)
    Q = sqrt(N) * Q / th.linalg.norm(Q[:, 0])

    T_C = th.exp(-1j * 2 * pi * th.outer(samples, delay))
    T_C = th.fft.fftshift(T_C, 0)

    Xc = th.outer(PULSE_FFT * CA_FFT, th.ones(delay.size()[0]))
    #tx_power = np.trapz(np.abs(Xc[:,0])**2,dx=T/N)/T

    C = th.fft.ifft(T_C * Xc, len(samples), 0)
    #C = np.fft.fftshift(C, 0)
    C = sqrt(N) * C / th.norm(C[:, 0])

    #plt.plot(np.fft.ifft(PULSE_FFT * CA_FFT),'b')
    #plt.plot(C[:,0],'r--')
    #plt.show()
    CQ = th.conj(Q).T @ C

    U, S, Vh = th.linalg.svd(Q, full_matrices=False)
    Qw = U
    OMEGA = th.diag(th.complex(S, th.zeros(S.size()[0]))) @ Vh

    CQw = th.conj(Qw).T @ C

    return CQ, Q, CQw, Qw, OMEGA


class gnssSimulSignal:

    def __init__(self, cn0, m=8, k=30, q=11) -> None:
        self.C_N0 = cn0  # carrier to noise density ratio C/N0 (dB-Hz)
        self.m = m  # no. of sensors in the array
        self.k = k  # seq. collection periods
        self.q = q  # number of correlators

        self.__setUpGNSSpar()
        self.__callfrequencyDomain()
        self.__setUpChannelpar()

    def __setUpGNSSpar(self):
        self.sat_no = 20  # satellite number
        self.Nd = 1023  # no. chips per PR seq.
        self.B = 1.023e6  # chip seq. bandwidth
        self.Tc = 1 / 1.023e6  # chip period
        self.fc = 1575.42e6  # carrier freq.
        self.T = 1e-3  #
        #self.q = 11             # number of correlators
        self.CB_delays = th.linspace(-self.Tc, self.Tc, self.q)

        res = 2 * self.Tc * self.fc + 1
        self.isv = th.linspace(min(self.CB_delays), max(self.CB_delays), int(res))

    def __callfrequencyDomain(self):
        self.CA_FFT, self.CA_PSD = frequecy_domain_CA(self.B, self.T, self.sat_no)

    def __setUpChannelpar(self):
        self.L = 1  # delays
        self.d = 1 + self.L  # model order (1 source + L delays)
        #self.m = 8              # no. of sensors in the array
        self.n = 2 * self.Nd  # of snapshots (per period)
        #self.k = 30             # seq. collection periods

        self.G = self.B * self.T  # processing gain (linear)
        self.SNR_dB = self.C_N0 - 10 * log10(2 * self.B) + 10 * log10(self.G)  # post-correlation SNR
        self.SMR_dB = 5  # Signal-to-Multipath Ratio

        P_LOS = 10**(self.SNR_dB / 10)
        gamma_LOS = sqrt(P_LOS)

        P_NLOS = P_LOS / (10**(self.SMR_dB / 10))
        gamma_NLOS = sqrt(P_NLOS) * th.ones(self.L)

        self.abs_gamma = th.tensor([gamma_LOS, gamma_NLOS])

    def signalModel(self, DoA_deg=10, angle_diff_deg=5, delay_diff=0.5):
        BANK_delay = self.CB_delays

        delay = th.zeros(self.d)

        phases = th.exp(1j * 2 * pi * (th.rand(2)))
        Gamma = th.outer(self.abs_gamma * phases, th.ones(self.k))

        phi = th.tensor([DoA_deg, DoA_deg + angle_diff_deg])

        A = array_lin(phi, self.m)

        # Generate delay

        tau0 = self.Tc * 0.3 / 0.5 * (th.rand(1) - 0.5)
        if self.L == 1:
            tauL = tau0 + self.Tc * delay_diff
        else:
            tauL = tau0 + self.Tc * delay_diff * th.ones(self.L)

        self.tau_vec = th.tensor([tau0, tauL])
        tau = self.tau_vec

        CQ, _, CQw, Qw, OMEGA = correlator_bank_Q(self.B, self.Tc, self.T, BANK_delay, tau, self.CA_FFT)

        X0 = A @ khatri_rao([Gamma.T, CQw]).T

        Z = tl.tensor(1 / sqrt(2) * th.randn(self.m, self.k, self.n) + 1j * th.randn(self.m, self.k, self.n),
                      dtype=th.complex64)
        Zf = tl.tenalg.mode_dot(Z, th.conj(Qw).T, 2)

        X = tl.tensor(X0.reshape(self.m, self.k, self.q).clone().detach(), dtype=th.complex64)

        factors = [A.clone().detach(), Gamma.T.clone().detach(), CQw.clone().detach()]

        Y = X + Zf

        return Y, X, factors, Qw, CQ, CQw, Qw, OMEGA

    def signalModel2(self, DoA_deg=10, angle_diff_deg=5, delay_diff=0.5, epsilon=0.5):
        BANK_delay = self.CB_delays

        delay = th.zeros(self.d)

        phases = th.exp(1j * 2 * pi * (th.rand(2)))
        Gamma = th.outer(self.abs_gamma * phases, th.ones(self.k))

        phi = th.tensor([DoA_deg, DoA_deg + angle_diff_deg])

        A = array_lin_noisePhase(phi, self.m, epsilon)

        tau0 = self.Tc * 0.3 / 0.5 * (th.rand(1) - 0.5)
        if self.L == 1:
            tauL = tau0 + self.Tc * delay_diff
        else:
            tauL = tau0 + self.Tc * delay_diff * th.ones(self.L)

        self.tau_vec = th.tensor([tau0, tauL])
        tau = self.tau_vec

        CQ, _, CQw, Qw, OMEGA = correlator_bank_Q(self.B, self.Tc, self.T, BANK_delay, tau, self.CA_FFT)

        X0 = A @ khatri_rao([Gamma.T, CQw]).T

        Z = tl.tensor(1 / sqrt(2) * th.randn(self.m, self.k, self.n) + 1j * th.randn(self.m, self.k, self.n),
                      dtype=th.complex64)
        Zf = tl.tenalg.mode_dot(Z, th.conj(Qw).T, 2)

        X = tl.tensor(X0.reshape(self.m, self.k, self.q).clone().detach(), dtype=th.complex64)

        factors = [A.clone().detach(), Gamma.T.clone().detach(), CQw.clone().detach()]

        Y = X + Zf

        return Y, X, factors, Qw, CQ, CQw, Qw, OMEGA

    def signalModel3(self, DoA_deg=10, angle_diff_deg=5, delay_diff=0.5, epsilon=0.5):
        BANK_delay = self.CB_delays

        delay = th.zeros(self.d)

        phases = th.exp(1j * 2 * pi * (th.rand(2)))
        Gamma = th.outer(self.abs_gamma * phases, th.ones(self.k))

        phi = th.tensor([DoA_deg, DoA_deg + angle_diff_deg])

        A = array_lin_noise(phi, self.m, epsilon)

        tau0 = self.Tc * 0.3 / 0.5 * (th.rand(1) - 0.5)
        if self.L == 1:
            tauL = tau0 + self.Tc * delay_diff
        else:
            tauL = tau0 + self.Tc * delay_diff * th.ones(self.L)

        self.tau_vec = th.tensor([tau0, tauL])
        tau = self.tau_vec

        CQ, _, CQw, Qw, OMEGA = correlator_bank_Q(self.B, self.Tc, self.T, BANK_delay, tau, self.CA_FFT)

        X0 = A @ khatri_rao([Gamma.T, CQw]).T

        Z = tl.tensor(1 / sqrt(2) * th.randn(self.m, self.k, self.n) + 1j * th.randn(self.m, self.k, self.n),
                      dtype=th.complex64)
        Zf = tl.tenalg.mode_dot(Z, th.conj(Qw).T, 2)

        X = tl.tensor(X0.reshape(self.m, self.k, self.q).clone().detach(), dtype=th.complex64)

        factors = [A.clone().detach(), Gamma.T.clone().detach(), CQw.clone().detach()]

        Y = X + Zf

        return Y, X, factors, Qw, CQ, CQw, Qw, OMEGA


def genCQw(B, Tc, T, BANK_delay, delay, CA_FFT):
    N = 2 * B * T  # number of samples per frame
    f0 = 2 * B / N  # basis frequency
    samples = f0 * (th.linspace(0, int(N) - 1, int(N)) - (N / 2))

    PULSE_SPEC = th.fft.fftshift(Tc * th.sinc(samples * Tc)**2)  # GPS C/A code rectangular pulse
    PULSE_FFT = th.fft.fftshift(sqrt(Tc) * th.sinc(samples * Tc)**2)  # GPS C/A code rectangular pulse

    #SIGNAL_SPEC = PULSE_SPEC * CA_SPEC

    T_Q = th.exp(-1j * 2 * pi * th.outer(samples, BANK_delay))
    T_Q = th.fft.fftshift(T_Q, 0)

    # Back to time domain
    X = th.outer(PULSE_FFT * CA_FFT, th.ones(BANK_delay.size()[0]))
    Q = th.fft.ifft(T_Q * X, len(samples), 0)
    #Q = np.fft.fftshift(Q, 0)
    Q = sqrt(N) * Q / th.linalg.norm(Q[:, 0])

    T_C = th.exp(-1j * 2 * pi * th.outer(samples, delay))
    T_C = th.fft.fftshift(T_C, 0)

    Xc = th.outer(PULSE_FFT * CA_FFT, th.ones(delay.size()[0]))
    #tx_power = np.trapz(np.abs(Xc[:,0])**2,dx=T/N)/T

    C = th.fft.ifft(T_C * Xc, len(samples), 0)
    #C = np.fft.fftshift(C, 0)
    C = sqrt(N) * C / th.norm(C[:, 0])

    #plt.plot(np.fft.ifft(PULSE_FFT * CA_FFT),'b')
    #plt.plot(C[:,0],'r--')
    #plt.show()
    CQ = np.conj(Q.numpy()).T @ C.numpy()

    U, S, Vh = np.linalg.svd(Q, full_matrices=False)
    Qw = U
    OMEGA = np.diag(S) @ Vh

    CQw = np.conj(Qw).T @ C.numpy()

    return CQw


def black_box_LS(Y2, dataModel, tauLos, tauNLos):
    if np.abs(tauLos - tauNLos) / dataModel.Tc < 1e-5 * dataModel.Tc:
        tauNLos += 1e-5 * dataModel.Tc

    if tauLos > tauNLos:
        return -1

    CQw = genCQw(dataModel.B, dataModel.Tc, dataModel.T, dataModel.CB_delays, th.tensor([tauLos, tauNLos]),
                 dataModel.CA_FFT)
    M = np.linalg.pinv(CQw) @ Y2.numpy()

    return -(np.linalg.norm(Y2.numpy() - CQw @ M) / np.linalg.norm(Y2.numpy()))**2


####################################################################################################################################


def interpolate_bank_filter(x, y, xfit):
    tck = interpolate.splrep(x, y, s=0)
    return th.tensor(interpolate.splev(xfit, tck, der=0))


########################################################## Ideal Case (Perfect Known of CQ) ########################################


def ideal_delay_estimation(CQ, dataModel):
    q_ideal = CQ[:, 0]
    yfit = interpolate_bank_filter(dataModel.CB_delays, abs(q_ideal), dataModel.isv)
    return dataModel.isv[th.argmax(yfit)]


########################################################## ideal case + LSKRF ######################################################


def known_spatial_phase_factors(Y, A0, A1, dataModel, OMEGA):
    q_hat = tl.tenalg.mode_dot(tl.tenalg.mode_dot(Y, th.linalg.pinv(A0), 0), th.linalg.pinv(A1), 1)[0, 0, :] @ OMEGA
    yfit = interpolate_bank_filter(dataModel.CB_delays, abs(q_hat), dataModel.isv)
    return dataModel.isv[th.argmax(yfit)], yfit


########################################################## ESPRIT + LSKRF############################################################


def extract_subspace(E, d):
    U, _, _ = th.svd(E)
    return U[:, 0:d]


def esprit(E, L, N):
    S = extract_subspace(E, L)
    Phi = th.linalg.pinv(
        S[0:N - 1, :]) @ S[1:N, :]  # the original array is divided into two subarrays [0,1,...,N-2] and [1,2,...,N-1]
    eigs, _ = th.linalg.eig(Phi)
    return th.arcsin(th.angle(eigs) / pi)


def lskrf(C, M, N):
    # A M x D
    # B N x D
    # C MN x D

    # C = krp(A,B)
    D = C.size()[1]
    A = th.zeros(M, D, dtype=th.complex64)
    B = th.zeros(N, D, dtype=th.complex64)

    for ii in range(D):

        c_d = C[:, ii]

        C_hat = c_d.reshape([M, N])

        U, S, Vh = th.svd(C_hat)

        B[:, ii] = sqrt(S[0]) * Vh[:, 0].conj()
        A[:, ii] = sqrt(S[0]) * U[:, 0]

    return A, B


def esprit_lskrf(Y, E, L, N, dataModel, OMEGA):

    DoAEst_deg = esprit(E, L=2, N=N) * 180 / pi
    A_SE = array_lin(DoAEst_deg, dataModel.m)

    C = th.linalg.pinv(A_SE) @ tl.unfold(Y, 0)
    Gamma, CQw = lskrf(C.T, dataModel.k, dataModel.q)
    idx = th.flipud(th.argsort(abs(th.diag(th.mm(Gamma.conj().T, Gamma)))))
    q_esprit_lskrf = CQw[:, idx[0]].T @ OMEGA
    yfit = interpolate_bank_filter(dataModel.CB_delays, abs(q_esprit_lskrf), dataModel.isv)
    return dataModel.isv[th.argmax(yfit)], yfit


#################################################### Simulation ######################################################


def simulation(seed, angle_diff_deg, delay_diff, C_N0, xi, epsilon):
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
    tau_HOSVD_ideal = ideal_delay_estimation(CQ, dataModel)
    tau_ag_est, y_ideal_pinv = known_spatial_phase_factors(Y, factors[0], factors[1], dataModel, OMEGA)
    tau_lskrf_est, y_lskrf = esprit_lskrf(Y, E, 2, m_s, dataModel, OMEGA)

    ## Calculate Bayesian Optimization

    tau_los_limit = dataModel.Tc * 0.3
    pbounds = {
        'tauLos': (-tau_los_limit, tau_los_limit),
        'tauNLos': (-tau_los_limit, dataModel.Tc)  # NLOS component may be into 2 Tc's
    }

    bounds_transformer = SequentialDomainReductionTransformer()

    Y2 = tl.unfold(Y, 2)

    black_box_LSBayOpt = lambda tauLos, tauNLos: black_box_LS(Y2, dataModel, tauLos, tauNLos)

    optimizer = BayesianOptimization(
        f=black_box_LSBayOpt,
        pbounds=pbounds,
        verbose=0,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        bounds_transformer=bounds_transformer,
        random_state=opt_random_state)

    randomPoints = 2
    optimizer.maximize(init_points=randomPoints, n_iter=60, acq="ei", xi=xi)

    tau_vec = np.array([optimizer.max['params']['tauLos'], optimizer.max['params']['tauNLos']])
    tau_bayOpt_est = np.min(tau_vec)

    ref_black_box_LSBayOpt = lambda tauVec: -black_box_LSBayOpt(tauVec[0], tauVec[1])

    res = minimize(ref_black_box_LSBayOpt, tau_vec, method='L-BFGS-B')
    tau_bayOpt_ref_est = np.min(res.x)

    return tau_real.numpy().item(), tau_HOSVD_ideal.numpy().item(), \
        tau_bayOpt_est, tau_bayOpt_ref_est, tau_lskrf_est.numpy().item(), tau_ag_est.numpy().item()


import argparse
import shelve

if __name__ == "__main__":
    DoA_deg = 10

    # angle_diff_deg_vec = np.array([5, 10, 15])
    # delay_diff_vec     = np.linspace(0, 9, 10)/10
    # C_N0_vec           = np.linspace(0, 60, 11)
    # xi_vec             = 10**np.linspace(-6, 0, 7)
    # epsilon_vec        = np.linspace(0, 9, 10)/10

    # parser = argparse.ArgumentParser(description='Delay Estimation')
    # parser.add_argument('-i', dest = 'idx', type = int, help = 'Index', required = True)
    # args = parser.parse_args()

    # idx = args.idx
    # dimensions = (len(angle_diff_deg_vec), len(delay_diff_vec), len(C_N0_vec), len(xi_vec), len(epsilon_vec))
    # angle_diff_deg_idx, delay_diff_idx, C_N0_idx, xi_idx, epsilon_idx = np.unravel_index(idx, dimensions)

    # angle_diff_deg = angle_diff_deg_vec[angle_diff_deg_idx]
    # delay_diff = delay_diff_vec[delay_diff_idx]
    # C_N0 = C_N0_vec[C_N0_idx]
    # xi = xi_vec[xi_idx]
    # epsilon = epsilon_vec[epsilon_idx]

    parser = argparse.ArgumentParser(description='Delay Estimation')
    parser.add_argument('-a', dest='angle_diff_deg', type=float, help='angle_diff_deg', required=True)
    parser.add_argument('-d', dest='delay_diff', type=float, help='delay_diff', required=True)
    parser.add_argument('-c', dest='C_N0', type=float, help='C_N0', required=True)
    parser.add_argument('-x', dest='xi', type=float, help='xi', required=True)
    parser.add_argument('-e', dest='epsilon', type=float, help='epsilon', required=True)
    parser.add_argument('-b', dest='batch', type=int, help='batch', required=True)
    args = parser.parse_args()

    angle_diff_deg = args.angle_diff_deg
    delay_diff = args.delay_diff
    C_N0 = args.C_N0
    xi = args.xi
    epsilon = args.epsilon
    batch = args.batch

    print(f'''Simulation: 
    angle_diff_deg = {angle_diff_deg}
    delay_diff = {delay_diff}
    C_N0 = {C_N0}
    xi = {xi}
    epsilon = {epsilon}
    batch = {batch}\n
    ''')

    mc = 1000

    init_seed = mc * batch

    tau_real = np.full(mc, np.nan)
    tau_HOSVD_ideal = np.full(mc, np.nan)
    tau_lskrf_est = np.full(mc, np.nan)
    tau_ag_est = np.full(mc, np.nan)
    tau_bayOpt_est = np.full(mc, np.nan)
    tau_bayOpt_ref_est = np.full(mc, np.nan)

    for seed in range(mc):
        results = simulation(init_seed + seed, angle_diff_deg, delay_diff, C_N0, xi, epsilon)

        tau_real[seed] = results[0]
        tau_HOSVD_ideal[seed] = results[1]
        tau_bayOpt_est[seed] = results[2]
        tau_bayOpt_ref_est[seed] = results[3]
        tau_lskrf_est[seed] = results[4]
        tau_ag_est[seed] = results[5]

        print(f'{seed}/{mc}', flush=True)

    with shelve.open(f'results/results_{angle_diff_deg}_{delay_diff}_{C_N0}_{xi}_{epsilon}_{batch}.dat', 'n') as f:
        f['angle_diff_deg'] = angle_diff_deg
        f['delay_diff'] = delay_diff
        f['C_N0'] = C_N0
        f['xi'] = xi
        f['epsilon'] = epsilon
        f['batch'] = batch

        f['tau_real'] = tau_real
        f['tau_HOSVD_ideal'] = tau_HOSVD_ideal
        f['tau_bayOpt_est'] = tau_bayOpt_est
        f['tau_bayOpt_ref_est'] = tau_bayOpt_ref_est
        f['tau_lskrf_est'] = tau_lskrf_est
        f['tau_ag_est'] = tau_ag_est
