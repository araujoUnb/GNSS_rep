import numpy as np
import scipy.io
#mat = scipy.io.loadmat('file.mat')




def ca_code(n):
    # returns the Gold code for GPS satellite ID n
    # n=1...32
    #the code is represented at levels : -1 for bit 0
    #                                     1 for bit 1

    #phase assignments
    phase = np.array([[2,6],[3,7],[4,8],[5,9],[1,9],[2,10],[1,9],[2,9],
                      [3,10],[2,3],[3,4],[5,6],[6,7],[7,8],[8,9],[9,10],
                      [1,4],[2,5],[3,6],[4,7],[5,8],[6,9],[1,3],[4,6],
                      [5,7],[6,8],[7,9],[8,10],[1,6],[2,7],[3,8],[4,9]]).astype(int)

    g =  np.zeros(1023)
    g0 = -1*np.ones(10)
    g1 = g0

    #select taps for G2 delay
    s0 = phase[n,0].item() - 1
    s1 = phase[n,1].item() - 1
    tmp = 0

    for ii in range(1023):
        # Gold code
        g[ii] = g1[s0]*g1[s1]*g0[9]
        #generator 1 - shift reg1
        tmp = g0[0]
        g0[0] = g0[2] * g0[9]
        g0[1:10] = np.block([tmp,g0[1:9]])
        # generator 2 - shift reg2
        tmp = g1[0]
        g1[0] = g1[1]*g1[2]*g1[5]*g1[7]*g1[8]*g1[9]
        g1[1:10] = np.block([tmp,g1[1:9]])

    return g





