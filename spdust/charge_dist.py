# This is the python version of the charge_dist.pro file of spdust2.0.
# 
# It first computes the photoemission rate according to Weingartner & Draine,
# 2001b, ApJ 134, 263. All equation numbers refer to that paper.
# Then compute collisional charging rates according to Draine & Sutin, 1987, ApJ, 320, 803.                              
# Finally, gets the grain charge distribution function for given grain size      
# and environmental conditions.  


import numpy as np
from spdust import SpDust_data_dir  
from utils.util import cgsconst, makelogtab, DX_over_X, biinterp_func, coord_grid
from spdust.grain_properties import asurf, N_C
from scipy.interpolate import interp1d    
import os      
from numba import njit, jit


class paramjpe:
    # Stores the parameters used in the JPE calculation
    W = 4.4  # Work function in eV

W = paramjpe.W

class Qabstabs:
    # Stores the Qabs arrays
    Nrad = None
    Nwav = None
    a_tab = None
    Qabs_hnu_tab = None
    Q_abs_ion = None
    Q_abs_neu = None

class refr_indices:
    hnu_tab = None
    la_tab = None

class jpe_arrays:
    # Stores the Jpeisrf arrays
    a_values = None
    Jpe_pos_isrf = None
    Jpe_neg_isrf = None

# Constants
c, q, k, mp, me, h, debye, eV = cgsconst.c, cgsconst.q, cgsconst.k, cgsconst.mp, cgsconst.me, cgsconst.h, cgsconst.debye, cgsconst.eV
W = paramjpe.W
pi = np.pi

# Equation (2), in eV
@njit
def IPv(Z, a):
    return W + ((Z + 0.5) * q**2 / a + (Z + 2) * q**2 / a * (0.3e-8) / a) / eV


# Equation (7), in eV
@njit
def E_min(Z, a):
    if Z < -1:
        return -(Z + 1) * q**2 / a / (1 + (27e-8 / a)**0.75) / eV
    else:
        return 0.0

# Equation (6), in eV
@njit
def hnu_pet(Z, a):
    if Z >= -1:
        return max(0.0, IPv(Z, a))
    else:
        return max(0.0, IPv(Z, a) + E_min(Z, a))
    
# Equation (9), in eV
@njit
def theta(hnu_tab_aux, Z, a):
 
    hnu_pet_result = hnu_pet(Z, a)

    if Z >= 0:
        return hnu_tab_aux - hnu_pet_result + (Z + 1.0) * q**2 / a / eV
    else:
        return hnu_tab_aux - hnu_pet_result
    
# Equation (11) and preceding paragraph
@njit
def y2(hnu_tab_aux, Z, a):
    Nnu = len(hnu_tab_aux)
    y2_tab = np.zeros(Nnu)

    if Z >= 0:
        E_low = -(Z + 1.0) * q**2 / a / eV
        E_high = hnu_tab_aux - hnu_pet(Z, a)
        ind = np.where(E_high > max([0.0, E_low]))
        count = len(ind[0])
        if count != 0:
            E_high = E_high[ind]
            y2_tab[ind] = E_high**2 * (E_high - 3.0 * E_low) / (E_high - E_low)**3

    else:
        ind = np.where(hnu_tab_aux > hnu_pet(Z, a))
        count = len(ind[0])
        if count != 0:
            y2_tab[ind] = 1.0

    return y2_tab

#@jit(nopython=False)
def l_a():
    """
    Stores la(hnu) needed for equation (15) of WD01b.    
    Uses tabulated values for Im(m_para) and
    Im(m_perp), taken from Draine website and given for a = 100A, T = 20K.
    We assume the changes due to grain size and temperature are negligible.
    """

    # Read data from files
    fileperp = SpDust_data_dir + 'perpendicular.out'
    filepara = SpDust_data_dir + 'parallel.out'

    lamb_perp, Imm_perp = np.loadtxt(fileperp, usecols=(0, 4), unpack=True, comments=';')
    lamb_para, Imm_para = np.loadtxt(filepara, usecols=(0, 4), unpack=True, comments=';')
   
    # There are more wavelength values given for m_para to account for line effects,
    # so we interpolate Im(m_perp) at those values.

    # Interpolate Im(m_perp) at additional wavelength points
    #Imm_perp_interp = np.exp(np.interp(np.log(lamb_para), np.log(lamb_perp), np.log(Imm_perp)))
    
    interp_function = interp1d(np.log(lamb_perp), np.log(Imm_perp), kind='cubic', fill_value="extrapolate")
    Imm_perp_interp = np.exp(interp_function(np.log(lamb_para)))

    # Convert wavelength to cm and calculate energy
    lamb_tab = lamb_para * 1e-4  # in cm
    hnu_tab_aux = h * c / (lamb_tab * eV)

    # Calculate la_tab using WD01 (15)
    la_tab_aux = 3 * lamb_tab / (4 * pi) / (2 * Imm_perp_interp + Imm_para)

    print('l_a computed')

    return hnu_tab_aux, la_tab_aux

hnu_tab, la_tab = l_a()
refr_indices.hnu_tab = hnu_tab
refr_indices.la_tab = la_tab 
 
# Takes care of small beta case in eq (13)
@njit
def fy1(x):
    """
    Takes care of small beta case as described in equation (13).
    
    Parameters:
    - x: NumPy array of values for which the function is computed.
    
    Returns:
    - NumPy array with the computed result for each value of `x`.
    """

    xmin = 1e-5  # Threshold for small x values
    result = np.zeros_like(x)  # Initialize result array with the same shape as x

    # Boolean indexing for small and rest values
    ind_small = x < xmin
    rest = x >= xmin

    # Apply the formula for small x
    if np.any(ind_small):
        result[ind_small] = x[ind_small] / 3.0

    # Apply the formula for the rest of the values
    if np.any(rest):
        xrest = x[rest]
        result[rest] = (xrest**2 - 2.0 * xrest + 2.0 - 2.0 * np.exp(-xrest)) / xrest**2

    return result

# Equation (13) in WD01b
@njit
def y1(a):
    l_e = 1e-7  # electron escape length in cm

    beta = a / la_tab
    alpha = a / la_tab + a / l_e

    return fy1(alpha) / fy1(beta)

# Equation (16) in WD01b
@njit
def y0(theta_val):
    return 9e-3 * (theta_val / W)**5 / (1 + 3.7e-2 * (theta_val / W)**5)

# Equation (12) for hnu values in hnu_tab
@njit
def Y(Z, a):
    N_hnu = len(hnu_tab)
    Y_result = np.zeros(N_hnu)

    theta_values = theta(hnu_tab, Z, a)
    y0_values = y0(theta_values)
    y1_values = y1(a)

    y0y1_values = y0_values * y1_values
    #ind = np.where(y0y1_values > 1.0)
    #if len(ind[0]) > 0:
    #    y0y1_values[ind] = 1.0
    ind = y0y1_values > 1.0
    if np.any(ind):
        y0y1_values[ind] = 1.0


    Y_result = y2(hnu_tab, Z, a) * y0y1_values

    return Y_result

# Planck function for nu in Hz 
@njit
def Planck_B(nu, T):
    return 2.0 * h * nu**3 / c**2 / (np.exp(h * nu / (k * T)) - 1.0)

@njit
def nu_uisrf(hnu_tab_aux):
    """
    ; ---> Average interstellar radiation field spectrum nu*u_nu
    ; (Mathis, Mezger & Panagia 1983) see eq (31) of WD01b
    ; IMPORTANT note: the ambiant radiation field is always supposed to
    ; be a multiple of u_ISRF. Of course you may change that but you will
    ; need to re-run some calculations for the Infrared damping and
    ; excitation coefficients (see readme file). 
    """

    # Initialize array
    uisrf_tab = np.zeros(hnu_tab_aux.shape, dtype=np.float64)

    # Conditions for different ranges
    #ind1 = np.where((hnu_tab_aux < 13.6) & (hnu_tab_aux > 11.2))
    ind1 = (hnu_tab_aux < 13.6) & (hnu_tab_aux > 11.2)
    if np.any(ind1):
        uisrf_tab[ind1] = 3.328e-9 * hnu_tab_aux[ind1]**(-4.4172)

    #ind2 = np.where((hnu_tab_aux < 11.2) & (hnu_tab_aux > 9.26))
    ind2 = (hnu_tab_aux < 11.2) & (hnu_tab_aux > 9.26)
    if np.any(ind2):
        uisrf_tab[ind2] = 8.463e-13 / hnu_tab_aux[ind2]

    #ind3 = np.where((hnu_tab_aux < 9.26) & (hnu_tab_aux > 5.04))
    ind3 = (hnu_tab_aux < 9.26) & (hnu_tab_aux > 5.04)
    if np.any(ind3):
        uisrf_tab[ind3] = 2.055e-14 * hnu_tab_aux[ind3]**0.6678

    #ind4 = np.where(hnu_tab_aux < 5.04)
    ind4 = (hnu_tab_aux < 5.04)
    if np.any(ind4):
        nu = hnu_tab_aux[ind4] * eV / h
        uisrf_tab[ind4] = 4 * pi * nu / c * (
            1e-14 * Planck_B(nu, 7500) + 1.65e-13 * Planck_B(nu, 4000) + 4e-13 * Planck_B(nu, 3000)
        )

    return uisrf_tab

def readPAH():

    # File paths
    PAHion = SpDust_data_dir + 'PAHion_read.out'
    PAHneu = SpDust_data_dir + 'PAHneu_read.out'

    # Common variables
    Nrad = 30
    Nwav = 1201

    # Read PAH ion and neutral data
    C1aux, C2_ion = np.loadtxt(PAHion, usecols=(0, 2), unpack=True, comments=';')
    C2_neu = np.loadtxt(PAHneu, usecols=(2,), unpack=True, comments=';')

    # Initialize arrays
    a_tab = np.zeros(Nrad)
    lambda_tab = np.zeros(Nwav)

    Q_abs_neu = np.zeros((Nrad, Nwav))
    Q_abs_ion = np.zeros((Nrad, Nwav))

    # Populate absorption efficiency arrays
    for i in range(Nrad):
        a_tab[i] = C1aux[(Nwav+1)*i] * 1e-4
        for j in range(Nwav):
            lambda_tab[j] = C1aux[(Nwav + 1) * i + j + 1]
            Q_abs_neu[i, j] = C2_neu[(Nwav + 1) * i + j + 1]
            Q_abs_ion[i, j] = C2_ion[(Nwav + 1) * i + j + 1]

    # Calculate wavelength array
    Qabs_hnu_tab = h * c / (lambda_tab * 1e-4) / eV

    print('readPAH computed')

    Qabstabs.Nrad = Nrad
    Qabstabs.Nwav = Nwav
    Qabstabs.a_tab = a_tab
    Qabstabs.Qabs_hnu_tab = Qabs_hnu_tab
    Qabstabs.Q_abs_ion = Q_abs_ion
    Qabstabs.Q_abs_neu = Q_abs_neu
    return 

readPAH()

Qabs_interp_func_neu = biinterp_func(Qabstabs.Q_abs_neu, np.log(Qabstabs.a_tab), np.log(Qabstabs.Qabs_hnu_tab))
Qabs_interp_func_ion = biinterp_func(Qabstabs.Q_abs_ion, np.log(Qabstabs.a_tab), np.log(Qabstabs.Qabs_hnu_tab)) 

def Qabs(a, Z, hnu_tab_aux):
    Qabs_hnu_tab = Qabstabs.Qabs_hnu_tab
    #a_tab = Qabstabs.a_tab
    #Q_abs_ion = Qabstabs.Q_abs_ion
    #Q_abs_neu = Qabstabs.Q_abs_neu

    # Minimum frequency for which Q_abs is tabulated
    hnu_min = np.min(Qabs_hnu_tab)

    N_nu = len(hnu_tab_aux)
    Qtab = np.zeros(N_nu)

    # Find indices where hnu_tab > hnu_min
    ind = hnu_tab_aux > hnu_min
    pts = coord_grid(np.log(a), np.log(hnu_tab_aux[ind]))
    if np.any(ind):
        if Z == 0:
            Qtab[ind] = Qabs_interp_func_neu(pts)[0]
        else:
            Qtab[ind] = Qabs_interp_func_ion(pts)[0]
            #log_biinterplog(Q_abs_ion, a_tab, Qabs_hnu_tab, a, hnu_tab_aux[ind])

    # For long wavelengths where hnu_tab <= hnu_min, assume Qabs ∝ ν^2
    ind = hnu_tab_aux <= hnu_min
    pts = coord_grid(np.log(a), np.log(hnu_min))
    if np.any(ind):
        if Z == 0:
            Qtab[ind] = (hnu_tab_aux[ind] / hnu_min)**2 * Qabs_interp_func_neu(pts)[0]
        else:
            Qtab[ind] = (hnu_tab_aux[ind] / hnu_min)**2 * Qabs_interp_func_ion(pts)[0]
            #Qtab[ind] = (hnu_tab_aux[ind] / hnu_min)**2 * log_biinterplog(Q_abs_ion, a_tab, Qabs_hnu_tab, a, hnu_min)[0]

    return Qtab

# Equation (4) in eV for Z < 0
@njit
def EA(Z, a):
    return W + ((Z - 0.5) * q**2 / a - q**2 / a * 4e-8 / (a + 7e-8)) / eV

# Equation (18) in eV for Z < 0
@njit
def hnu_pdt(Z, a):
    return max(0, EA(Z + 1, a) + E_min(Z, a))  

# Equation (20) in cm^2, for Z < 0
@njit
def sigma_pdt(hnu_tab_aux, Z, a):
    DeltaE = 3.0  # in eV
    x = (hnu_tab_aux - hnu_pdt(Z, a)) / DeltaE
    sigma = np.zeros_like(hnu_tab_aux)

    if Z >= 0:
        return sigma

    ind = x > 0
    if np.any(ind):
        x = x[ind]
        sigma[ind] = 1.2e-17 * np.abs(Z) * x / (1 + x**2 / 3)**2

    return sigma

# First term in eq (25) for the standard interstellar radiation field (31)


def first_term(Z, a):
    hnu_min = max([1e-5, hnu_pet(Z, a)])
    hnu_max = 13.6

    # Parameter
    Nnu = 500

    if hnu_min > hnu_max:
        return 0.0

    hnu = makelogtab(hnu_min, hnu_max, Nnu)
    Dnu_over_nu = DX_over_X(hnu_min, hnu_max, Nnu)

    Ytab = np.interp(np.log(hnu), np.log(hnu_tab), Y(Z, a))

    Qtab = Qabs(a, Z, hnu)
    utab = nu_uisrf(hnu)

    return c * pi * a**2 * Dnu_over_nu * np.sum(Ytab * Qtab * utab / (hnu * eV))

# Second term in eq (25) for the standard interstellar radiation field (31), for Z < 0
@njit
def second_term(Z, a):

    hnu_min = max([1e-5, hnu_pdt(Z, a)])
    hnu_max = 13.6

    # Parameter
    Nnu = 500

    if Z >= 0 or hnu_pdt(Z, a) > hnu_max:
        return 0.0

    hnu = makelogtab(hnu_min, hnu_max, Nnu)
    Dnu_over_nu = DX_over_X(hnu_min, hnu_max, Nnu)

    utab = nu_uisrf(hnu)
    sigmatab = sigma_pdt(hnu, Z, a)

    return c * np.sum(sigmatab * utab / (hnu * eV)) * Dnu_over_nu

# Equation (21)
def Jpe(Z, a):
    return first_term(Z, a) + second_term(Z, a)

# Equation (22)
@njit
def Zmax(a):
    aA = a / 1e-8  # a in A

    hnu_max = 13.6  # You may need to change this value
    result= np.floor(((hnu_max - W)/ 14.4 * aA + 0.5 - 0.3 / aA) / (1 + 0.3 / aA))
    return int(result)

# Equations (23), (24)
@njit
def Zmin(a):
    aA = a / 1e-8  # a in A
    U_ait = -(3.9 + 0.12 * aA + 2 / aA)
    result = np.floor(U_ait / 14.4 * aA) + 1
    return int(result)

# Function to compute and store Jpeisrf
def JPEisrf_calc(filename='jpeisrf_data.npz'):
    """
    ; ---> Computes and stores the photoemission rate Jpe(a, Z), for an
    ; array of grain radii and charges, for tha averge interstellar
    ; radiation field.

    If previously computed results exist, they will be loaded.
    """
    # Check if the saved file exists
    filename = os.path.join(SpDust_data_dir, filename)
    if os.path.exists(filename):
        # Load saved data
        data = np.load(filename)
        a_values = data['a_values']
        Jpe_pos_isrf = data['Jpe_pos_isrf']
        Jpe_neg_isrf = data['Jpe_neg_isrf']
        print('Loaded previously computed Jpeisrf arrays from file.')
    else:
        a_min = 3.5e-8
        a_max = 1e-6
        Na = 30
        # Initialize arrays
        a_values = makelogtab(a_min, a_max, Na)
        Z_min = Zmin(a_max)
        Z_max = Zmax(a_max)
        Jpe_neg_isrf = np.zeros((Na, abs(Z_min) + 1))
        Jpe_pos_isrf = np.zeros((Na, Z_max + 1))

        # Loop to compute Jpeisrf
        for i in range(Na):
            a = a_values[i]
            for j in range(Zmax(a) + 1):
                Z = j
                Jpe_pos_isrf[i, j] = Jpe(Z, a)
                if j <= abs(Zmin(a)):
                    Z = -j
                    Jpe_neg_isrf[i, j] = Jpe(Z, a)

        print('Jpeisrf computed')

        # Save computed arrays to file for future use
        np.savez(filename, a_values=a_values, Jpe_pos_isrf=Jpe_pos_isrf, Jpe_neg_isrf=Jpe_neg_isrf)
        print(f'Jpeisrf arrays saved to {filename}.')

    jpe_arrays.a_values = a_values
    jpe_arrays.Jpe_pos_isrf = Jpe_pos_isrf
    jpe_arrays.Jpe_neg_isrf = Jpe_neg_isrf

    return a_values, Jpe_pos_isrf, Jpe_neg_isrf

# Call the function
a_values, Jpe_pos_isrf, Jpe_neg_isrf = JPEisrf_calc()


# Function to interpolate Jpe arrays
@jit
def JPEisrf(a):
    #a_values, Jpe_pos_isrf, Jpe_neg_isrf = jpe_arrays.a_values, jpe_arrays.Jpe_pos_isrf, jpe_arrays.Jpe_neg_isrf

    Jpepos = np.zeros(len(Jpe_pos_isrf[0, :]))
    Jpeneg = np.zeros(len(Jpe_neg_isrf[0, :]))

    if a <= np.min(a_values):
        Jpepos = Jpe_pos_isrf[0, :]
        Jpeneg = Jpe_neg_isrf[0, :]
    else:
        if a >= np.max(a_values):
            N_a = len(a_values)
            Jpepos = Jpe_pos_isrf[N_a - 1, :]
            Jpeneg = Jpe_neg_isrf[N_a - 1, :]
        else:
            #ia = np.max(np.where(a_values <= a))
                # Use np.where with index extraction
            indices = np.where(a_values <= a)[0]
            if indices.size > 0:
                ia = indices[-1]  # Take the last index where the condition holds
            else:
                ia = 0  # Handle case where no values match the condition
            alpha = np.log(a / a_values[ia]) / np.log(a_values[ia + 1] / a_values[ia])
            Jpepos = (1.0 - alpha) * Jpe_pos_isrf[ia, :] + alpha * Jpe_pos_isrf[ia + 1, :]
            Jpeneg = (1.0 - alpha) * Jpe_neg_isrf[ia, :] + alpha * Jpe_neg_isrf[ia + 1, :]

    Zmin_val = Zmin(a)
    Zmax_val = Zmax(a)

    Jpepos = Jpepos[:Zmax_val + 1]
    Jpeneg = Jpeneg[:-Zmin_val + 1]

    return Jpepos, Jpeneg

# Function to calculate Jtilde
@njit
def Jtilde(tau, nu):

    if nu == 0:
        return 1.0 + np.sqrt(pi / (2.0 * tau))  # Equation (3.3)

    if nu < 0:
        return (1.0 - nu / tau) * (1.0 + np.sqrt(2.0 / (tau - 2.0 * nu)))  # Equation (3.4)

    if nu > 0:
        ksi = 1.0 + 1.0 / np.sqrt(3.0 * nu)
        theta = nu / ksi - 0.5 / (ksi**2 * (ksi**2 - 1.0))  # Equation (2.4a)

        if theta / tau < 700.0:  # Avoid floating-point underflow
            return (1.0 + (4.0 * tau + 3.0 * nu)**(-0.5))**2 * np.exp(-theta / tau)  # Equation (3.5)
        else:
            return 0.0
        
# Function to calculate the sticking coefficient for electrons
# Equations refer to WD01b
# Note that s_i = 1 for ions

@njit
def se(Z, a):
    l_e = 1e-7  # "electron escape length" (10 A) see below (28)
    Nc = N_C(a)
    Z_min = Zmin(a)  

    if Z == 0 or (Z < 0 and Z > Z_min):
        return 0.5 * (1.0 - np.exp(-a / l_e)) / (1.0 + np.exp(20.0 - Nc))  # (28),(29)

    if Z < 0 and Z <= Z_min:
        return 0.0  # (29)

    if Z > 0:
        return 0.5 * (1.0 - np.exp(-a / l_e))  # (30)

# Function to calculate J_ion
@njit
def J_ion_aux(nh, T, xh, xc, Z, a):
    asurf_val = asurf(a)  
    tau = asurf_val * k * T / q**2
    nu = Z 
    return nh * np.sqrt(8.0 * k * T / (pi * mp)) * pi * asurf_val**2 * (xh + xc / np.sqrt(12.0)) * Jtilde(tau, nu)

def J_ion(env, Z, a):
    nh = env['nh']
    T = env['T']
    xh = env['xh']
    xc = env['xC']
    return J_ion_aux(nh, T, xh, xc, Z, a)

# Function to calculate J_electron
@njit
def J_electron_aux(nh, T, xh, xc, Z, a):
    asurf_val = asurf(a)  
    tau = asurf_val * k * T / q**2
    nu = -Z 
    return nh * (xh + xc) * se(Z, a) * np.sqrt(8.0 * k * T / (np.pi * me)) * pi * asurf_val**2 * Jtilde(tau, nu)

def J_electron(env, Z, a):
    nh = env['nh']
    T = env['T']
    xh = env['xh']
    xc = env['xC']
    return J_electron_aux(nh, T, xh, xc, Z, a) 

# Function to compute charge distribution
@jit
def charge_dist_aux(Chi, nh, T, xh, xc, a):
    Z_min = Zmin(a) 
    Z_max = Zmax(a)  

    fneg = np.zeros(abs(Z_min) + 1)
    fpos = np.zeros(Z_max + 1)

    Ji_neg = np.zeros(abs(Z_min) + 1)
    Je_neg = np.zeros(abs(Z_min) + 1)
    Ji_pos = np.zeros(Z_max + 1)
    Je_pos = np.zeros(Z_max + 1)
    Jpepos_val, Jpeneg_val = JPEisrf(a)
    Jpe_ISRF = {'Jpepos': Jpepos_val, 'Jpeneg': Jpeneg_val}

    for j in range(abs(Z_min) + 1):
        Z = -j
        Ji_neg[j] = J_ion_aux(nh, T, xh, xc, Z, a)
        Je_neg[j] = J_electron_aux(nh, T, xh, xc, Z, a)
    Jpe_neg = Chi * Jpe_ISRF['Jpeneg']

    for Z in range(Z_max + 1):
        Ji_pos[Z] = J_ion_aux(nh, T, xh, xc, Z, a)
        Je_pos[Z] = J_electron_aux(nh, T, xh, xc, Z, a)
    Jpe_pos = Chi * Jpe_ISRF['Jpepos']

    # Compute charge distribution function using DL98b (4)
    fpos[0] = 1.0
    fneg[0] = 1.0

    Z = -1
    while Z >= Z_min:
        fneg[-Z] = fneg[-(Z + 1)] * Je_neg[-(Z + 1)] / (Ji_neg[-Z] + Jpe_neg[-Z])
        Z -= 1
        fneg /= np.sum(fneg)

    Z = 1
    while Z <= Z_max:
        fpos[Z] = fpos[Z - 1] * (Ji_pos[Z - 1] + Jpe_pos[Z - 1]) / Je_pos[Z]
        Z += 1
        fpos /= np.sum(fpos)

    if fneg[0] > 0:
        fneg = fpos[0] / fneg[0] * fneg
    else:
        fpos = np.zeros(1)

    norm = np.sum(fneg) + np.sum(fpos) - fpos[0]

    fneg /= norm
    fpos /= norm

    Zneg = -np.arange(abs(Z_min) + 1)
    Zpos = np.arange(Z_max + 1)

    # Keep only the values for which f_Z > fmin
    ind_neg = np.where((fneg > 1e-10) & (Zneg != 0))
    if len(ind_neg[0]) > 0:
        fneg = fneg[ind_neg]
        Zneg = Zneg[ind_neg]

    ind_pos = np.where((fpos > 1e-10) | (Zpos == 0))
    fpos = fpos[ind_pos]
    Zpos = Zpos[ind_pos]

    # Combine arrays
    fZ = np.zeros((2, len(Zpos) + len(Zneg)))

    fZ_array = fpos
    Z_array = Zpos

    if len(Zneg) > 0:
        Z_array = np.concatenate((Z_array, Zneg))
        fZ_array = np.concatenate((fZ_array, fneg))

    # Renormalize
    fZ_array /= np.sum(fZ_array)

    fZ[0, :] = Z_array
    fZ[1, :] = fZ_array

    return fZ

def charge_dist(env, a):
    Chi = env['Chi']
    nh = env['nh']
    T = env['T']
    xh = env['xh']
    xc = env['xC']

    return charge_dist_aux(Chi, nh, T, xh, xc, a)
