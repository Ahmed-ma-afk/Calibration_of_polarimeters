import numpy as np
from H_A import H_A
# it has to be optimized for performance


def K_A(M_0, M_1, M_2, M_3, B_0, B_1, B_2, B_3):
    KA = np.einsum('ij,jk', H_A(M_0, M_1, B_0, B_1).T, H_A(M_0, M_1, B_0, B_1))
    KA += np.einsum('ij,jk', H_A(M_0, M_2, B_0, B_2).T,
                    H_A(M_0, M_2, B_0, B_2))
    KA += np.einsum('ij,jk', H_A(M_0, M_3, B_0, B_3).T,
                    H_A(M_0, M_3, B_0, B_3))
    return KA
