import numpy as np
import pandas as pd

from os.path import exists

from zmq import ZAP_DOMAIN
import matplotlib.pyplot as plt 


#import sys

#sys.path.extend(['/Users/araujo/Documents/GitHub/GNSS_rep'])


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
    return -np.sign(np.array(ca) - 0.5)

def create_matrix_C(B, Tc, T, delay, CA_FFT):
    N = 2 * B * T  # number of samples per frame
    f0 = 2 * B / N  # basis frequency
    samples = f0 * (np.linspace(0, int(N) - 1, int(N)) - (N / 2 - 1 / 2))

    PULSE_FFT = np.fft.fftshift(np.sqrt(Tc * np.sinc(samples * Tc) ** 2))  # GPS C/A code rectangular pulse

    T_C = np.fft.fftshift(np.exp(-1j * 2 * np.pi * np.outer(samples, delay)), 0)

    Xc = np.outer(PULSE_FFT * CA_FFT, np.ones(delay.size))

    C = np.real(np.fft.ifft(T_C * Xc))
    C = np.fft.fftshift(C, 0)
    C = np.sqrt(N) * C / np.linalg.norm(C[:, 0])
    return C
    

def gen_delayed_signal(B,T,Tc,CA_FFT,delay):

    N = 2 * B * T  # number of samples per frame
    f0 = 2 * B / N  # basis frequency
    samples = f0 * (np.linspace(0, int(N) - 1, int(N)) - (N / 2 - 1 / 2))

    PULSE_SPEC = np.fft.fftshift(Tc * np.sinc(samples * Tc) ** 2)  # GPS C/A code rectangular pulse
    PULSE_FFT = np.fft.fftshift(np.sqrt(Tc) * np.sinc(samples * Tc) ** 2)  # GPS C/A code rectangular pulse

    #SIGNAL_SPEC = PULSE_SPEC * CA_SPEC

    T_Q = np.fft.fftshift(np.exp(-1j * 2 * np.pi * np.kron(samples.reshape(samples.size, 1), delay.T)), 0)

    # Back to time domain
    X = np.outer(PULSE_FFT * CA_FFT, np.ones(delay.size))

    Q = np.fft.ifft(T_Q * X)

    return np.sum(Q,axis=1)


def correlator_bank_Q(B, Tc, T, BANK_delay, delay, CA_FFT):
    N = 2 * B * T  # number of samples per frame
    f0 = 2 * B / N  # basis frequency
    samples = f0 * (np.linspace(0, int(N) - 1, int(N)) - (N / 2))

    PULSE_SPEC = np.fft.fftshift(Tc * np.sinc(samples * Tc) ** 2)  # GPS C/A code rectangular pulse
    PULSE_FFT = np.fft.fftshift(np.sqrt(Tc) * np.sinc(samples * Tc) ** 2)  # GPS C/A code rectangular pulse

    #SIGNAL_SPEC = PULSE_SPEC * CA_SPEC

    T_Q = np.exp(-1j * 2 * np.pi * np.outer(samples,BANK_delay))
    T_Q = np.fft.fftshift(T_Q, 0)

    # Back to time domain
    X = np.outer(PULSE_FFT * CA_FFT, np.ones(BANK_delay.size))
    Q = np.fft.ifft(T_Q * X,len(samples),0)
    #Q = np.fft.fftshift(Q, 0)
    Q = np.sqrt(N) * Q / np.linalg.norm(Q[:, 0])

   

    T_C = np.exp(-1j * 2 * np.pi * np.outer(samples,delay))
    T_C = np.fft.fftshift(T_C, 0)

    Xc = np.outer(PULSE_FFT * CA_FFT, np.ones(delay.size))
    #tx_power = np.trapz(np.abs(Xc[:,0])**2,dx=T/N)/T

    C = np.fft.ifft(T_C * Xc,len(samples),0)
    #C = np.fft.fftshift(C, 0)
    C = np.sqrt(N) * C / np.linalg.norm(C[:, 0])
   
    #plt.plot(np.fft.ifft(PULSE_FFT * CA_FFT),'b')
    #plt.plot(C[:,0],'r--')
    #plt.show()
    CQ = np.conj(Q).T @ C


    U,S,Vh = np.linalg.svd(Q,0)
    Qw = U
    OMEGA = np.diag(S)@Vh

    CQw = np.conj(Qw).T @ C

    return CQ,Q,CQw,Qw,OMEGA

    


def  frequecy_domain_CA(B, T, SAT):
    
    Folder = 'C:\\Users\\danie\\Git\\GNSS_Artigo\\CACODE' + str(SAT) + '_' + str(B) + '.pkl'

    if exists(Folder) :

        objects = pd.read_pickle(Folder)

        CA_fft = objects['CA_FFT']
        CA_SPEC = objects['CA_SPEC']

        return CA_fft, CA_SPEC
    else:

        # B  -> Bandwidth
        # T  -> frame period
        # SAT -> ID

        N = 2 * B * T  # number of samples per frame
        # f0 = 2 * B / N # basis frequency
        # samples = f0 * (np.linspace(0, int(N) - 1, int(N)) - (N / 2 - 1 / 2))

        satcode = PRN(SAT)
        ca_code_length = satcode.size

        number_of_sequeces = int(N / ca_code_length)
        CA_fft = number_of_sequeces * np.kron(np.ones(number_of_sequeces), np.fft.fft(satcode))

        # Autocorrelation function
        Ra_code = np.zeros(satcode.size)
        for ii in range(ca_code_length):
            sat_shift = np.roll(satcode, -ii)
            Ra_code[ii] = sat_shift @ satcode / ca_code_length

        # Fourier transmform to obtain Power spetracl density of the sequence

        CA_SPEC = np.fft.fftshift(number_of_sequeces * np.kron(np.ones(number_of_sequeces), np.fft.fft(Ra_code)))

        code = {'CA_FFT': CA_fft,
                'CA_SPEC': CA_SPEC}

        code_df = pd.DataFrame(data=code)
        Folder = 'C:\\Users\\danie\\Git\\GNSS_Artigo\\CACODE' + str(SAT) + '_' + str(B) + '.pkl'
        code_df.to_pickle(Folder)

    return CA_fft, CA_SPEC
