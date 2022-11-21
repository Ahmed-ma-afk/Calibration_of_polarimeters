import time
import sys
sys.path.insert(1, 'AandW_pixel_calibration')
from Calibration_A import Calibration_A
from Calibration_W import Calibration_W

sys.path.insert(1, 'Preprocesssing')
from DataMatToNp import *






def rec(i1, i2, j1, j2, n):
    if(n == 1):
        B0_pixel = b0[i1, j1]
        B1_pixel = b1[i1, j1]
        B2_pixel = b2[i1, j1]
        B3_pixel = b3[i1, j1]
        # Make pixel calibration
        A_pixel = Calibration_A(
                M_Air, M_Pol0, M_Pol90, M_Ret30, B0_pixel, B1_pixel, B2_pixel, B3_pixel)
        W_pixel = Calibration_W(
                M_Air, M_Pol0, M_Pol90, M_Ret30, B0_pixel, B1_pixel, B2_pixel, B3_pixel)
        A[i1, j1] = A_pixel
        W[i1, j1] = W_pixel
        return
    rec(i1, (i1+i2)//2, j1, (j1+j2)//2, n/2)
    rec(i1, (i1+i2)//2, ((j1+j2)//2)+1, j2, n/2)
    rec(((i1+i2)//2)+1, i2, j1, (j1+j2)//2, n/2)
    rec(((i1+i2)//2)+1, i2, ((j1+j2)//2)+1, j2, n/2)

t1 = time.perf_counter()
rec(0,n-1,0,n-1,n)
t2 = time.perf_counter()
print("rec: ",t2 - t1)

