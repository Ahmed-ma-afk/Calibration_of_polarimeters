import numpy as np
import sys
sys.path.insert(1, 'AandW_pixel_calibration')
import Calibration_W as w
# We search ThetaP for polarizer and ThetaR for retarder


def ComputeEigenvaluesCmatrix(b0, b):
    b0inv = np.linalg.inv(b0)
    M_similar = b0inv@b  ## ?
    eigenvalues, eigenvectors = np.linalg.eig(M_similar)

    return eigenvalues


def Compute_t_Icp_Ic_Is(eigenvalues):
    "We assume that it respects the theoretical form"
    real_eigenvalues = []
    complex_eigenvalues = []
    for eigv in eigenvalues:
        if(np.imag(eigv)!=0):  ## ???? they are all complex in some cases
            complex_eigenvalues.append(eigv)
        else:
            real_eigenvalues.append(np.real(eigv))

    if(len(complex_eigenvalues) != 0):
        #real_eigenvalues = np.sort(np.array(real_eigenvalues))
        # print(eigenvalues)
        # print(real_eigenvalues)
        # print(np.real(eigenvalues))
        t = real_eigenvalues[0] + real_eigenvalues[1]
        Icp = np.abs((real_eigenvalues[1] - real_eigenvalues[0]))/t # We have to fix a sign for Icp
        Ic = np.real((complex_eigenvalues[0] + complex_eigenvalues[1]))/t 
        Is = np.abs((complex_eigenvalues[1] - complex_eigenvalues[0]))/t      # We have to fix a sign for Is
        return t, Icp, Ic, Is  # ok even >=1 because there is a different in the article
    
    else:
        print("On obtient quatre valeurs propres réelles")
        real_eigenvalues=np.sort(real_eigenvalues)
        t=real_eigenvalues[0]/2 ## for now
        return t,1,0,0

    

def ComputeMullerWithoutRotation(b0, b):
    eigenvalues = ComputeEigenvaluesCmatrix(b0 , b)
    t_Icp_Ic_Is = Compute_t_Icp_Ic_Is(eigenvalues)
    t = t_Icp_Ic_Is[0]
    Icp = t_Icp_Ic_Is[1]
    Ic = t_Icp_Ic_Is[2]
    Is = t_Icp_Ic_Is[3]

    Muller_WithoutRotation = (t/2)*np.array([[1, Icp, 0, 0],
                                             [Icp, 1, 0, 0],
                                             [0, 0, Ic, Is],
                                            [0, 0, -Is, Ic]])

    return Muller_WithoutRotation


def f_Rotation(a): return np.array([[1, 0, 0, 0],
                                    [0, np.cos(2*a), np.sin(2*a), 0],
                                    [0, -np.sin(2*a), np.cos(2*a), 0],
                                    [0, 0, 0, 1]])

def Find_real(thetaP,thetaR,M_0, M_1, M_2, M_3, B_0, B_1, B_2, B_3):
    m=100
    thetaP_x=np.linspace(thetaP-30,thetaP+30,m)
    thetaR_y=np.linspace(thetaR-30,thetaR+30,m)
    lamda_16_lamda_15=np.zeros((m,m))
    min =4
    couple=0
    for i in range(m):
        for j in range(m):
            M_2_thetaP=f_Rotation(thetaP_x[i]*np.pi/180)@M_2@f_Rotation(-thetaP_x[i]*np.pi/180)
            M_3_thetaR=f_Rotation(thetaR_y[j]*np.pi/180)@M_3@f_Rotation(-thetaR_y[j]*np.pi/180)
            lamda_16_lamda_15[i][j]= np.log(np.sqrt(np.abs(w.Calibration_W(M_0, M_1, M_2_thetaP,M_3_thetaR, B_0, B_1, B_2, B_3)[1])))
            if(min>lamda_16_lamda_15[i][j]):
                min=lamda_16_lamda_15[i][j]
                couple=[thetaP_x[i],thetaR_y[j]]

    return thetaP_x,thetaR_y,lamda_16_lamda_15,couple,np.sqrt(np.abs(min))

