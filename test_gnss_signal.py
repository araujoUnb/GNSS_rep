import numpy as np
from gnss_func.gnss_function import frequecy_domain_CA, correlator_bank_Q, create_matrix_C, frequecy_domain_CA, PRN
from gnss_func.array import array_lin
from gnss_func.utils import krp
from gnss_func.delayEst import interpolationMethod,  wienerMethod
from scipy.constants import speed_of_light
from scipy import interpolate
import matplotlib.pyplot as plt

class gnssSimulSignal:

    def setUp(self):

        
        self.setUpGNSSpar()
        self.callfrequencyDomain()
        self.setUpChannelpar()
        

    def setUpGNSSpar(self):
        self.sat_no = 20	 # satellite number
        self.Nd = 1023       # no. chips per PR seq.
        self.B = 1.023e6	 # chip seq. bandwidth
        self.Tc = 1/1.023e6  # chip period
        self.fc = 1575.42e6	 # carrier freq.
        self.T = 1e-3        # 
        self.q = 11             # number of correlators
        self.CB_delays = np.linspace(-self.Tc,self.Tc,self.q)

    def callfrequencyDomain(self):
        self.CA_FFT, self.CA_PSD = frequecy_domain_CA(self.B, self.T, self.sat_no)

    def setUpChannelpar(self):
        self.L = 1              # delays
        self.d = 1 + self.L     # model order (1 source + L delays)
        self.m = 8              # no. of sensors in the array
        self.n = 2*self.Nd      # of snapshots (per period)
        self.k = 30             # seq. collection periods

        self.C_N0 = 48           # carrier to noise density ratio C/N0 (dB-Hz)
        self.G = self.B*self.T   # processing gain (linear)
        self.SNR_dB = self.C_N0 - 10*np.log10(2*self.B) + 10*np.log10(self.G)	# post-correlation SNR
        self.SMR_dB = 5          # Signal-to-Multipath Ratio


        P_LOS = 10**(self.SNR_dB/10)
        gamma_LOS = np.sqrt(P_LOS)

        P_NLOS = P_LOS/(10**(self.SMR_dB/10))
        gamma_NLOS = np.sqrt(P_NLOS)*np.ones(self.L)

        self.abs_gamma = np.r_[gamma_LOS,gamma_NLOS]



    def testSignal(self):

        
        angle_diff = 60
        delay_diff = 0.1

        BANK_delay = np.linspace(-self.Tc,self.Tc,self.q)

        res = 2 * self.Tc * self.fc + 1
        isv = np.linspace(min(BANK_delay),max(BANK_delay),int(res))
        delay = np.zeros(self.d)    

       # _ ,_ ,_ ,Qw,OMEGA = correlator_bank_Q(self.B, self.Tc, self.T, BANK_delay, delay, self.CA_FFT)

        phases = np.exp(1j*2*np.pi*(np.random.rand(2)))
        Gamma = np.outer(self.abs_gamma*phases,np.ones(self.k))

        phi=np.array([10,10+angle_diff])

        A  = array_lin(phi, self.m )

        # Generate delay

        tau0=self.Tc*0.3/0.5*(np.random.rand()-0.5)
        if self.L == 1:
            tauL = tau0 + self.Tc*delay_diff
        else:
            tauL = tau0 + self.Tc*delay_diff*np.ones(self.L)

        tau  = np.r_[tau0,tauL]


        CQ,_,CQw,Qw,OMEGA  = correlator_bank_Q(self.B, self.Tc, self.T, BANK_delay, tau, self.CA_FFT)

        X2 = CQw @ krp(A,Gamma.T).T

        Z2 = 1/np.sqrt(2) * (np.random.randn(self.n,self.m*self.k) + 1j* np.random.randn(self.n,self.m*self.k))
        Zf = np.conj(Qw).T @ Z2

        Y = X2 + Zf

        # Direct correlation time-delay est

        tau_ideal       = interpolationMethod(CQ,isv,BANK_delay)
        e_ideal          = abs(tau0 - tau_ideal)**2
        dist_ideal       = np.sqrt(e_ideal)*speed_of_light

        print(dist_ideal)

        tau_wiener        = wienerMethod(Y,Qw,isv,self.B,self.T)
        e_wiener          = abs(tau0 - tau_wiener)**2
        dist_wiener       = np.sqrt(e_wiener)*speed_of_light

        print(dist_wiener)











if __name__ == "__main__" :

    simullGNSS = gnssSimulSignal()

    simullGNSS.setUp()
    simullGNSS.testSignal()
