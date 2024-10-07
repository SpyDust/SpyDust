import numpy as np
from scipy.special import erf
from spdust import SpDust_data_dir
from utils.util import cgsconst
from numba import njit
import os


mp = cgsconst.mp
q = cgsconst.q
pi = np.pi

class grainparams:
    # Stores some grain geometrical properties
    a2 = 6e-8          # radius under which grains are disklike
    d = 3.35e-8        # disk thickness (graphite interlayer separation)
    rho = 2.24         # carbon density in g/cm^3
    epsilon = 0.01     # charge centroid displacement

rho = grainparams.rho
d = grainparams.d
a2 = grainparams.a2
epsilon = grainparams.epsilon


# Number of carbon atoms in a grain
@njit
def N_C(a):
    return int(np.floor(4 * pi / 3 * a**3 * rho / (12 * mp)) + 1)

# Number of hydrogen atoms in a grain
@njit
def N_H(a):
    Nc = N_C(a)
    if Nc < 25:
        result = np.floor(0.5 * Nc + 0.5)
    elif Nc < 100:
        result = np.floor(2.5 * np.sqrt(Nc) + 0.5)
    else:
        result = np.floor(0.25 * Nc + 0.5)
    return int(result)

# Largest moment of inertia
@njit
def Inertia(a):
    Mass = (12 * N_C(a) + N_H(a)) * mp
    Isphere = 0.4 * Mass * a**2

    if a <= a2:
        return 5 / 3 * a / d * Isphere   # DL98b (A8)
    else:
        return Isphere

# Surface-equivalent radius
@njit
def asurf(a):
    if a <= a2:
        b = np.sqrt(4 / 3 * a**3 / d)
        return np.sqrt(b**2 / 2 + b * d / 2)
    else:
        return a

# Cylindrical excitation equivalent radius
@njit
def acx(a):
    if a <= a2:
        R = np.sqrt(4 / 3 * a**3 / d)
        return (3 / 8)**0.25 * R
    else:
        return a

# Root mean square dipole moment
@njit
def rms_dipole(a, Z2, beta):
    muZ = epsilon * np.sqrt(Z2) * q * acx(a)
    N_at = N_C(a) + N_H(a)
    return np.sqrt(N_at * beta**2 + muZ**2)




# Define the path to the size distribution file
size_dist_file = os.path.join(SpDust_data_dir, 'sizedists_table1.out')

class size_dist_arrays:
    bc1e5_tab, alpha_tab, beta_tab, at_tab, ac_tab, C_tab = \
    np.loadtxt(size_dist_file, usecols=(1, 3, 4, 5, 6, 7), unpack=True, comments=';')

class size_params():
    bc, alpha_g, beta_g, at_g, ac_g, C_g = None, None, None, None, None, None

    def __call__(self, line):
        self.bc = size_dist_arrays.bc1e5_tab[line] * 1e-5
        self.alpha_g = size_dist_arrays.alpha_tab[line]
        self.beta_g = size_dist_arrays.beta_tab[line]
        self.at_g = size_dist_arrays.at_tab[line] * 1e-4
        self.ac_g = size_dist_arrays.ac_tab[line] * 1e-4
        self.C_g = size_dist_arrays.C_tab[line]
        pass

# Grain size distribution
def size_dist(a, line):
    '''
    Grain size distribution, using Weingartner & Draine, 2001a prescription. 
    The line of their table 1 is given by the user in param_file.
    '''
    size_dist_params = size_params()
    size_dist_params(line)

    # Unpack size distribution parameters
    bc = size_dist_params.bc
    alpha_g = size_dist_params.alpha_g
    beta_g = size_dist_params.beta_g
    at_g = size_dist_params.at_g
    ac_g = size_dist_params.ac_g
    C_g = size_dist_params.C_g

    rho = grainparams.rho
    # Lognormal populations parameters
    mc = 12 * mp
    bci = np.array([0.75, 0.25]) * bc
    a0i = np.array([3.5, 30.]) * 1e-8
    sigma = 0.4
    amin = 3.5e-8

    Bi = 3 / (2 * np.pi)**1.5 * np.exp(-4.5 * sigma**2) / (
        rho * a0i**3 * sigma) * bci * mc / (1 + erf(3 * sigma / np.sqrt(2) + np.log(a0i / amin) / (sigma * np.sqrt(2))))
    D_a = np.sum(Bi / a * np.exp(-0.5 * (np.log(a / a0i) / sigma)**2))

    if beta_g >= 0:
        F_a = 1 + beta_g * a / at_g
    else:
        F_a = 1 / (1 - beta_g * a / at_g)

    cutoff = 1
    if a > at_g:
        cutoff = np.exp(-((a - at_g) / ac_g)**3)

    return D_a + C_g / a * (a / at_g)**alpha_g * F_a * cutoff
