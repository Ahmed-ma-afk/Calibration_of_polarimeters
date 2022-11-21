import numpy as np
from  H_W import H_W

def K_W(M_0,M_1,M_2,M_3,B_0,B_1,B_2,B_3):
    KW = np.einsum('ij,jk',H_W(M_0,M_1,B_0,B_1).T,H_W(M_0,M_1,B_0,B_1)) 
    KW += np.einsum('ij,jk',H_W(M_0,M_2,B_0,B_2).T,H_W(M_0,M_2,B_0,B_2))
    KW += np.einsum('ij,jk',H_W(M_0,M_3,B_0,B_3).T,H_W(M_0,M_3,B_0,B_3))
    return KW