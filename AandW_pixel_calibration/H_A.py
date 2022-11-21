import numpy as np


def H_A(M1, M2, B1, B2):
    invM1 = np.linalg.inv(M1)
    invB1 = np.linalg.inv(B1)
    B2_B1 = B2@invB1  # B2*(B1)^-1
    M2_M1 = M2@invM1  # M2*(M1)^-1
    zero = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            zero[i, j] = 1
            if(i == 0 and j == 0):
                a = (zero@M2_M1-B2_B1@zero).reshape((16, 1))
            else:
                a = np.concatenate(
                    (a, (zero@M2_M1-B2_B1@zero).reshape((16, 1))), axis=1)
            zero[i, j] = 0
    return a

# take around 0.4 on a mac pro processor


#I = np.identity(4)
#wI = np.ones((4, 4))
#print("#####_H_A_##########")
#print(H_A(I, wI, I, wI))
