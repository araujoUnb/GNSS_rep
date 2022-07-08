
import numpy as np


def conv_circ( signal, ker ):
    '''
        signal: real 1D array
        ker: real 1D array
        signal and ker must have same shape
    '''
    return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(ker) ))

def lin2dBm(linPow):
    return 10*np.log10(linPow) + 30

def normalise_columns(A):

    G = np.diag(np.diag(np.conj(A).T @ A))
    return A @ np.linalg.inv(np.sqrt(G))


def krp(A,B):

    if np.shape(A)[1] == np.shape(A)[1]:

        nCols = np.shape(A)[1]

        C = np.array([]) 

        for ii in range(nCols):

            C = np.append(C,np.kron(A[:,ii],B[:,ii]))

    return C.reshape(nCols,np.shape(A)[0]*np.shape(B)[0]).T