import numpy as np
from gnss_func.gnss_function import frequecy_domain_CA, frequecy_domain_CA, gen_delayed_signal
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

        
        delay_diff = 0.1
        delay = np.zeros(self.d)    
        # Generate delay

        tau0 = 50.46*1e3/3e8   #self.Tc*0.3/0.5*(np.random.rand()-0.5)
        if self.L == 1:
            tauL = tau0 + self.Tc*delay_diff
        else:
            tauL = tau0 + self.Tc*delay_diff*np.ones(self.L)

        tau  = np.r_[tau0]

        r = gen_delayed_signal(self.B,self.T,self.Tc,self.CA_FFT,tau)      # s

        ref =  gen_delayed_signal(self.B,self.T,self.Tc,self.CA_FFT,np.array([0]))

        plt.plot(np.real(r))
        plt.plot(np.real(ref),'--')

        plt.show()








if __name__ == "__main__" :

    simullGNSS = gnssSimulSignal()

    simullGNSS.setUp()
    simullGNSS.testSignal()
