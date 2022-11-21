import time
import sys
import numpy as np
sys.path.insert(1, 'AandW_pixel_calibration')
from Calibration_A import Calibration_A
from Calibration_W import Calibration_W

sys.path.insert(1, 'Preprocesssing')
from DataMatToNp import *
from MinimizeRapportEigenValues import *



def calibrationAandW():
    for i in range(n):
        for j in range(n):
            B0_pixel = b0[i, j, :]
            B1_pixel = b1[i, j, :]
            B2_pixel = b2[i, j, :]
            B3_pixel = b3[i, j, :]

            ##Calcul des matrices de Muller   ???????? a lot of problems
            # M_Pol0 = ComputeMullerWithoutRotation(B0_pixel , B1_pixel)
            # M_Pol90 = f_Rotation(thetaP*np.pi/180)@ComputeMullerWithoutRotation(B0_pixel , B2_pixel)@f_Rotation(-thetaP*np.pi/180)
            # M_Ret30 = f_Rotation(thetaR*np.pi/180)@ComputeMullerWithoutRotation(B0_pixel , B3_pixel)@f_Rotation(-thetaR*np.pi/180)

            # Make pixel calibration
            A_pixel = Calibration_A(
                M_Air, M_Pol0, M_Pol90, M_Ret30, B0_pixel, B1_pixel, B2_pixel, B3_pixel)
            W_pixel = Calibration_W(
                M_Air, M_Pol0, M_Pol90, M_Ret30, B0_pixel, B1_pixel, B2_pixel, B3_pixel)
            A[i, j] = A_pixel
            W[i, j] = W_pixel[0]
            # print('i ', i)
            # print('j ', j)

    return 


# t1 = time.perf_counter()
calibrationAandW()
# t2 = time.perf_counter()
# print("naive: ",t2 - t1)

A_m=np.einsum(
    'ijkl->kl', A)/(n**2)
W_m = np.einsum('ijkl->kl',W)/(n**2)

print(W_m)



