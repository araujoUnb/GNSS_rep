import numpy as np
import pandas as pd

import sys

sys.path.extend(['/Users/araujo/Documents/GitHub/GNSS_rep'])


def ca_code(n):
    # returns the Gold code for GPS satellite ID n
    # n=1...32
    # the code is represented at levels : -1 for bit 0
    #                                     1 for bit 1

    # phase assignments
    phase = np.array([[2, 6], [3, 7], [4, 8], [5, 9], [1, 9], [2, 10], [1, 9], [2, 9],
                      [3, 10], [2, 3], [3, 4], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10],
                      [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [1, 3], [4, 6],
                      [5, 7], [6, 8], [7, 9], [8, 10], [1, 6], [2, 7], [3, 8], [4, 9]]).astype(int)

    g = np.zeros(1023)
    g0 = -1 * np.ones(10)
    g1 = g0

    # select taps for G2 delay
    s0 = phase[n, 0].item() - 1
    s1 = phase[n, 1].item() - 1

    for ii in range(1023):
        # Gold code
        g[ii] = g1[s0] * g1[s1] * g0[9]
        # generator 1 - shift reg1
        tmp = g0[0]
        g0[0] = g0[2] * g0[9]
        g0[1:10] = np.block([tmp, g0[1:9]])
        # generator 2 - shift reg2
        tmp = g1[0]
        g1[0] = g1[1] * g1[2] * g1[5] * g1[7] * g1[8] * g1[9]
        g1[1:10] = np.block([tmp, g1[1:9]])

    return g


def correlator_bank_Q(B, Tc, T, BANK_delay, delay, CA_FFT):
    N = 2 * B * T  # number of samples per frame
    f0 = 2 * B / N  # basis frequency
    samples = f0 * (np.linspace(0, int(N) - 1, int(N)) - (N / 2 - 1 / 2))

    #PULSE_SPEC = np.fft.fftshift(Tc * np.sinc(samples * Tc) ** 2)  # GPS C/A code rectangular pulse
    PULSE_FFT = np.fft.fftshift(np.sqrt(Tc) * np.sinc(samples * Tc) ** 2)  # GPS C/A code rectangular pulse

    T_Q = np.fft.fftshift(np.exp(-1j * 2 * np.pi * np.kron(samples.reshape(samples.size, 1), BANK_delay.T)), 0)

    # Back to time domain
    X = np.outer(PULSE_FFT * CA_FFT, np.ones(BANK_delay.size))
    Q = np.real(np.fft.ifft(T_Q * X))
    Q = np.fft.fftshift(Q, 0)
    Q = np.sqrt(N) * Q / np.linalg.norm(Q[:, 0])

    T_C = np.fft.fftshift(np.exp(-1j * 2 * np.pi * np.outer(samples, delay)), 0)

    Xc = np.outer(PULSE_FFT * CA_FFT, np.ones(delay.size))
    C = np.real(np.fft.ifft(T_C * Xc))
    C = np.fft.fftshift(C, 0)
    C = np.sqrt(N) * C / np.linalg.norm(C[:, 0])

    return Q, C, C.T @ Q,


def frequecy_domain_CA(B, T, SAT):
    # B  -> Bandwidth
    # T  -> frame period
    # SAT -> ID

    N = 2 * B * T  # number of samples per frame
    # f0 = 2 * B / N # basis frequency
    # samples = f0 * (np.linspace(0, int(N) - 1, int(N)) - (N / 2 - 1 / 2))

    satcode = ca_code(SAT)
    ca_code_length = satcode.size

    number_of_sequeces = int(N / ca_code_length)
    CA_fft = number_of_sequeces * np.kron(np.ones(number_of_sequeces), np.fft.fft(satcode))

    # Autocorrelation function
    Ra_code = np.zeros(satcode.size)
    for ii in range(ca_code_length):
        sat_shift = np.roll(satcode, ii)
        Ra_code[ii] = sat_shift @ satcode / ca_code_length

    # Fourier transmform to obtain Power spetracl density of the sequence

    CA_SPEC = np.fft.fftshift(number_of_sequeces * np.kron(np.ones(number_of_sequeces), np.fft.fft(Ra_code)))

    code = {'CA_FFT': CA_fft,
            'CA_SPEC': CA_SPEC}

    code_df = pd.DataFrame(data=code)
    Folder = '/Users/araujo/Documents/GitHub/GNSS_rep/CACODE/CA_FFT_' + str(SAT) + '_' + str(B) + '.pkl'
    code_df.to_pickle(Folder)

    return CA_fft, CA_SPEC
