import numpy as np

# une méthode naïve pour calculer H_W
# A*(Eij) Ci->j
# (Eij)*A Lj-i


def H_W(M1, M2, B1, B2):
    invM1 = np.linalg.inv(M1)
    invB1 = np.linalg.inv(B1)
    B1_B2 = invB1@B2  # (B1)^-1*B2
    M1_M2 = invM1@M2  # (M1)^-1*M2
    zero = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            zero[i, j] = 1
            if(i == 0 and j == 0):
                HW = (M1_M2@zero-zero@B1_B2).reshape((16, 1))
            else:
                HW = np.concatenate(
                    (HW, (M1_M2@zero-zero@B1_B2).reshape((16, 1))), axis=1)
            zero[i, j] = 0
    return HW


# take around 0.4 on a mac pro processor


#I = np.identity(4)
#wI = np.ones((4, 4))
#print("#####_H_W_##########")
#print(H_W(I, wI, I, wI))
