import numpy as np
# Polariseur -> L'axe de transmission orienté à 0° (VERIF!!)

f_Polar = 0.5*np.array([[1, 1, 0, 0],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]])
# Retardateur avec retard d -> L'axe lent orienté à 0°


def f_Retarder(d): return np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, np.cos(d), np.sin(d)],
                                    [0, 0, -np.sin(d), np.cos(d)]])


# Rotation d'un angle t. Angle positif -> antihoraire en regardant le faisceau !

def f_Rotation(a): return np.array([[1, 0, 0, 0],
                                    [0, np.cos(2*a), -np.sin(2*a), 0],
                                    [0, np.sin(2*a), np.cos(2*a), 0],
                                    [0, 0, 0, 1]])


# Retardateur tournée
def f_Ret_tour(d, a): return f_Rotation(-a)@f_Retarder(d)@f_Rotation(a)

# Polariseur tournée


def f_Pol_tour(ap): return f_Rotation(-ap)@f_Polar@f_Rotation(ap)


# Vecteur de Stokes initial dépolarisé.
Si = np.array([[1],
               [0],
               [0],
               [0]])


# Définition du vecteur de Stokes, des PSA, et les conditionnements associés.
# ---------------------------------------------------------------------------

# Vecteur de Stokes final
def f_So(dr, ar): return f_Ret_tour(dr, ar)@f_Polar@Si

# Vecteur de Stokes normalisé


def f_So1(dr, ar): return np.array(
    [1, 0, 0, 0])@f_So(dr, ar)  # Constante de normalisation


def f_SoN(dr, ar): return f_So(dr, ar)/f_So1(dr, ar)

# Matrice W normalisée et non-normalisée


def f_PSG(p): return np.concatenate((f_So(p[0], p[1]), f_So(
    p[0], p[2]), f_So(p[0], p[3]), f_So(p[0], p[4])), axis=1)


def f_PSGN(p): return np.concatenate((f_SoN(p[0], p[1]), f_SoN(
    p[0], p[2]), f_SoN(p[0], p[3]), f_SoN(p[0], p[4])), axis=1)

# Configurations d'Analyse


def f_ASo(dr, ar): return (np.array(
    [1, 0, 0, 0])@f_Polar@f_Ret_tour(dr, ar)).reshape(1, 4)
# Cette définition correspond à une configuration physiquement
# réalisable si au moment d'implémenter réellement le PSA, le signe des
# angles des élements du PSA sont lus suivant le même critère que pour
# ceux du PSG. Critère: En regardant le faisceau, le mouvement
# antihoraire est positif !

# Configurations d'Analyse normalisées


def f_ASo1(dr, ar): return f_ASo(dr, ar)@(np.array([1, 0, 0, 0]))


def f_ASoN(dr, ar): return f_ASo(dr, ar)/f_ASo1(dr, ar)

# Matrice A normalisée et non-normalisée


def f_PSA(p): return np.concatenate((f_ASo(p[0], p[1]), f_ASo(
    p[0], p[2]), f_ASo(p[0], p[3]), f_ASo(p[0], p[4])), axis=0)


def f_PSAN(p): return np.concatenate((f_ASoN(p[0], p[1]), f_ASoN(
    p[0], p[2]), f_ASoN(p[0], p[3]), f_ASoN(p[0], p[4])), axis=0)


# Conditionnement des matrices du PSG du PSA

def f_Condition_PSG(p): return 1/np.linalg.cond(f_PSG(p))
def f_Condition_PSGN(p): return 1/np.linalg.cond(f_PSGN(p))


def f_Condition_PSA(p): return 1/np.linalg.cond(f_PSA(p))
def f_Condition_PSAN(p): return 1/np.linalg.cond(f_PSAN(p))


# Matrice B et conditionnement
# Matrice non normalisée

# matrice de mmooler est l'identité
def f_MB(pA, pW): return f_PSA(pA)@f_PSG(pW)

# Matrice B normalisée


def f_MBN(pA, pW): return f_PSAN(pA)@f_PSGN(pW)

# Conditionnement de la matrice B


def f_Condition_MB(pA, pW): return 1/np.linalg.cond(f_MB(pA, pW))


def f_Condition_MBN(pA, pW): return 1/np.linalg.cond(f_MBN(pA, pW))

# Matrices B des élements de Calibration non-normal


def f_MBCalPol(pA, pW, ap): return f_PSA(pA)@f_Pol_tour(ap)@f_PSG(pW)
def f_MBCalRet(pA, pW, dr, ar): return f_PSA(pA)@f_Ret_tour(dr, ar)@f_PSG(pW)

# Matrices B des élements de Calibration normalisées


def f_MBCalPolN(pA, pW, ap): return f_PSAN(pA)@f_Pol_tour(ap)@f_PSGN(pW)


def f_MBCalRetN(pA, pW, dr, ar): return f_PSAN(
    pA)@f_Ret_tour(dr, ar)@f_PSGN(pW)


# ================ Simulate the polarimeter ==================================
#
# Params pour calcul matrices
# Params typiques pour le PSG
offset = 0
Ret_PSG = 90
Azi_PSG = np.array([Ret_PSG, -51.7+offset, -15.1+offset, 15.1 +
                   offset, 51.7+offset])*(np.pi/180)

# Params typiques pour le PSA
offset = 0
Ret_PSA = 90
Azi_PSA = np.array([Ret_PSA, -51.7+offset, -15.1+offset, 15.1 +
                   offset, 51.7+offset])*(np.pi/180)


# Matrices
# Matrices du PSG, PSA  et B normalisées
PSGN = f_PSGN(Azi_PSG)  # Matrice W du PSG.
Cdt_PSGN = f_Condition_PSGN(Azi_PSG)
PSAN = f_PSAN(Azi_PSA)  # Matrix A du PSA.
Cdt_PSAN = f_Condition_PSAN(Azi_PSA)

MBN = f_MBN(Azi_PSA, Azi_PSG)  # Matrice B0 (Air) B_Air
Cdt_MBN = f_Condition_MBN(Azi_PSA, Azi_PSG)

MBN_CalPol0 = f_MBCalPolN(Azi_PSA,  Azi_PSG, 0)  # Matrice B_Pol0
MBN_CalPol90 = f_MBCalPolN(Azi_PSA,  Azi_PSG, 90*np.pi/180)  # Matrice B_Pol90
MBN_CalRet30 = f_MBCalRetN(Azi_PSA,  Azi_PSG, 90 *
                           np.pi/180, 30*np.pi/180)  # Matrice B_Ret30

M_Ret30 = f_Ret_tour(90*np.pi/180, 30*np.pi/180)
M_Pol90 = f_Pol_tour(90*np.pi/180)

M_Air = np.identity(4)  # M_Air


# print("###########PSGN############")
# print(PSGN)
# print("###########Cdt_PSGN############")
# print(Cdt_PSGN)
# print("###########PSAN############")
# print(PSAN)
# print("###########Cdt_PSAN############")
# print(Cdt_PSAN)
# print("###########MBN############")
# print(MBN)
# print("###########Cdt_MBN############")
# print(Cdt_MBN)
# print("###########MBN_CalPol0############")
# print(MBN_CalPol0)
# print("###########MBN_CalPol90############")
# print(MBN_CalPol90)
# print("###########MBN_CalRet30############")
# print(MBN_CalRet30)
