import numpy as np
from scipy import interpolate



def interpolationMethod(CQ,isv,bank_delay):

    q_ideal = CQ[:,0]
    tck = interpolate.splrep(bank_delay, np.abs(q_ideal))
    F_ideal = interpolate.splev(isv, tck)
    idx_ideal = np.argmax(F_ideal)
    tau_ideal  = isv[idx_ideal]
    return tau_ideal

def wienerMethod(Y,Q,isv,B,T):

    Rx = Y @ np.conj(Y).T
    Rx_inv = np.linalg.pinv(Rx)

    N = 2 * B * T  # number of samples per frame
    f0 = 2 * B / N  # basis frequency
    samples = f0 * (np.linspace(0, int(N) - 1, int(N)) - (N / 2))

    TC =  np.exp(-1j * 2 * np.pi * np.outer(samples,isv))

    QT = np.conj(Q).T @ TC

    nRows = np.shape(Y)[0]
    
    W = np.zeros([nRows, len(isv)],dtype=complex)

    for ii in range(len(isv)):
        scalar  = np.conj(QT[:,ii]).T @ Rx_inv @ QT[:,ii]
        W[:,ii] = ( Rx_inv @ QT[:,ii] )/ scalar

    b = np.sum(np.abs(np.conj(W).T @ Y)**2,axis=1)
    idx = np.argmax(b)
    return isv[idx]


