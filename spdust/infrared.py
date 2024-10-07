# Normalized rotational damping and excitation rates through infrared emission.  
# Refers to Section 7 of Ali-Haimoud, Hirata & Dickinson, 2009, MNRAS, 395, 1055  
# and Section 6 of Silsbee, Ali-Haimoud & Hirata, 2010.      
# Use Draine & Li, 2001, ApJ 551, 807 (DL01) "thermal continuous" approximation 
# to compute the infrared spectrum.

import numpy as np
from spdust import SpDust_data_dir
from utils.util import maketab, DX_over_X, cgsconst, makelogtab
from spdust.grain_properties import N_C, N_H, Inertia, acx, grainparams
from spdust.charge_dist import Qabs, nu_uisrf, Qabstabs
from numba import jit, njit
import os
from utils.mpiutil import *


class IR_arrays:
    a_min = None
    a_max = None
    Na =  None
    a_tab = None
    chi_min = None
    chi_max = None
    Nchi = None
    chi_tab = None
    FIR_integral_charged = None
    GIR_integral_charged = None
    FIR_integral_neutral = None
    GIR_integral_neutral = None
    Tev_tab = None


Theta_op = 863.0  # Debye temperature for out-of-plane C-C modes
Theta_ip = 2504.0  # Debye temperature for in-plane C-C modes
k = cgsconst.k
mp = cgsconst.mp
h = cgsconst.h
c = cgsconst.c
eV = cgsconst.eV
pi = np.pi

Energy_tab_max = np.max(Qabstabs.Qabs_hnu_tab) # Maximum energy (eV) in the Qabs_hnu_tab
print(f"Maximum energy in the Qabs_hnu_tab is {Energy_tab_max:.2f} eV.")


@njit
def f2(x):
    """
    Compute f2(x) as per DL01 Eq. (10). x can be an array.

    Parameters:
    - x: Input array.

    Returns:
    - Array of f2 values corresponding to the input x.
    """

    Nx = np.size(x)  # Number of elements in x
    Ny = 500     # Number of integration points for y
    y = maketab(0, 1, Ny)  # Create the y array from 0 to 1 with Ny points
    Dy = 1.0 / Ny  # Integration step size

    # Initialize the integrand array (Ny x Nx)
    integrand = np.zeros((Ny, Nx))

    # Create the y/x matrix (outer product of y and 1/x)
    y_over_x = np.outer(y, 1.0 / x)

    # Find the indices where y_over_x < 500
    # ind = y_over_x < 500

    # Compute the integrand at the valid indices
    # if np.any(ind):
    #    integrand[ind] = 1.0 / (np.exp(y_over_x[ind]) - 1.0)

    # Use loops for njit compatibility
    for i in range(Ny):
        for j in range(Nx):
            if y_over_x[i, j] < 500:
                integrand[i, j] = 1.0 / (np.exp(y_over_x[i, j]) - 1.0)

    # Multiply by y^2
    integrand *= (y[:, np.newaxis]**2) 

    # Perform numerical integration: sum over the first axis (y-axis)
    result = 2.0 * Dy * np.sum(integrand, axis=0)

    return result

@njit
def Energy_modes(a):
    Nc = N_C(a)
    Nm = (Nc - 2) * np.array([1, 2])  # Number of out-of-plane and in-plane modes
    beta2_tab = np.zeros(2)

    if Nc <= 54:
        beta2_tab = np.zeros(2)
    elif 54 < Nc <= 102:
        beta2_tab = (Nc - 54) / 52 / (2 * Nm - 1)
    else:
        beta2_tab = ((Nc - 2) / 52 * (102 / Nc)**(2 / 3) - 1) / (2 * Nm - 1)

    # Out-of-plane modes
    beta2 = beta2_tab[0]
    Nm = Nc - 2
    NewNm = Nm
    Eop_tab = np.zeros(NewNm + 1)
    delta_tab = 0.5 + np.zeros(NewNm + 1)
    delta_tab[2] = 1
    delta_tab[3] = 1
    Eop_tab[1:] = k * Theta_op * np.sqrt((1 - beta2) / Nm * (1 + np.arange(NewNm) - delta_tab[1:]) + beta2)

    # In-plane modes
    beta2 = beta2_tab[1]
    Nm = 2 * (Nc - 2)
    NewNm = Nm
    Eip_tab = np.zeros(NewNm + 1)
    delta_tab = 0.5 + np.zeros(NewNm + 1)
    delta_tab[2] = 1
    delta_tab[3] = 1
    Eip_tab[1:] = k * Theta_ip * np.sqrt((1 - beta2) / Nm * (1 + np.arange(NewNm) - delta_tab[1:]) + beta2)

    # C-H stretching modes
    lambda_inverse = np.array([886.0, 1161.0, 3030.0])
    CH_modes = h * c * lambda_inverse

    return Eop_tab[1:], Eip_tab[1:], CH_modes
    #return {'op': Eop_tab[1:], 'ip': Eip_tab[1:], 'CH': CH_modes}

@njit
def EPAH(a, T, Energy_modes_op, Energy_modes_ip, Energy_modes_CH):
    """
    Compute the average energy E_PAH(a, T) in the thermal approximation.

    Parameters:
    - a: Grain size.
    - T: Array of temperatures. Must be ARRAY-like.
    - Energy_modes: Energy modes calculated based on grain size.

    Returns:
    - Array of average energies E_PAH for the given temperatures.
    """

    cutoff = 6e4  # Cutoff for Nc to switch between exact and continuous limit
    NT = np.size(T)  # Number of temperature values

    # Number of carbon and hydrogen atoms
    Nc = N_C(a)  # Assuming N_C(a) is defined
    NH = N_H(a)  # Assuming N_H(a) is defined

    # Energy of C-H bending and stretching modes
    E_Hmodes = np.zeros((np.size(Energy_modes_CH), NT))  # Initialize the energy array
    E_over_kT = Energy_modes_CH[:, None] / (k * T)  # Broadcasting to divide CH modes by k*T

    # Avoid floating point overflow for large E_over_kT values
    # mask = E_over_kT < 600
    # E_Hmodes[mask] = 1.0 / (np.exp(E_over_kT[mask]) - 1.0)

    # Use loops for njit compatibility
    for i in range(E_over_kT.shape[0]):
        for j in range(E_over_kT.shape[1]):
            if E_over_kT[i, j] < 600:
                E_Hmodes[i, j] = 1.0 / (np.exp(E_over_kT[i, j]) - 1.0)

    # Multiply by CH modes to get total energy in these modes
    E_Hmodes = (Energy_modes_CH[:, None] * np.ones(NT)) * E_Hmodes
    E_Hmodes = NH * np.sum(E_Hmodes, axis=0)

    if Nc > cutoff:  # Continuous limit for large Nc, equation (33)
        return (Nc - 2) * k * (Theta_op * f2(T / Theta_op) + 2 * Theta_ip * f2(T / Theta_ip)) + E_Hmodes
    else:  # Exact mode spectrum for small Nc
        # Out-of-plane modes
        Eop_tab =  Energy_modes_op
        Eop_temp = np.zeros((np.size(Eop_tab), NT))
        E_over_kT = Eop_tab[:, None] / (k * T)

        #mask = E_over_kT < 600
        #Eop_temp[mask] = 1.0 / (np.exp(E_over_kT[mask]) - 1.0)

        # Use loops for njit compatibility
        for i in range(E_over_kT.shape[0]):
            for j in range(E_over_kT.shape[1]):
                if E_over_kT[i, j] < 600:
                    Eop_temp[i, j] = 1.0 / (np.exp(E_over_kT[i, j]) - 1.0)

        Eop_temp = (Eop_tab[:, None] * np.ones(NT)) * Eop_temp
        Eop_bar = np.sum(Eop_temp, axis=0)

        # In-plane modes
        Eip_tab =  Energy_modes_ip
        Eip_temp = np.zeros((np.size(Eip_tab), NT))
        E_over_kT = Eip_tab[:, None] / (k * T)

        #mask = E_over_kT < 600
        #Eip_temp[mask] = 1.0 / (np.exp(E_over_kT[mask]) - 1.0)

        # Use loops for njit compatibility
        for i in range(E_over_kT.shape[0]):
            for j in range(E_over_kT.shape[1]):
                if E_over_kT[i, j] < 600:
                    Eip_temp[i, j] = 1.0 / (np.exp(E_over_kT[i, j]) - 1.0)

        Eip_temp = (Eip_tab[:, None] * np.ones(NT)) * Eip_temp
        Eip_bar = np.sum(Eip_temp, axis=0)

        return Eop_bar + Eip_bar + E_Hmodes

@jit
def Temp(a, Energy, Energy_modes_op, Energy_modes_ip, Energy_modes_CH):
    Tmin = 1.0
    Tmax = 1.0e4
    NT = 100
    T_tab = makelogtab(Tmin, Tmax, NT)
    E_tab = EPAH(a, T_tab, Energy_modes_op, Energy_modes_ip, Energy_modes_CH)

    #if Energy > E_tab[-1] or Energy < E_tab[0]:
    #    print(f'Energy {Energy / eV:.2f} eV is out of range in function Temp. The range is {E_tab[0] / eV:.2f} to {E_tab[-1] / eV:.2f} eV.')
    #    return None

    Temperature = np.exp(np.interp(np.log(Energy), np.log(E_tab), np.log(T_tab)))

    # Handle the energy modes (op, ip, CH)
    # modes = np.concatenate([ Energy_modes_op,  Energy_modes_ip, Energy_modes_CH])
    op_len = np.size(Energy_modes_op)
    ip_len = np.size(Energy_modes_ip)
    ch_len = np.size(Energy_modes_CH)
    modes = np.empty(op_len + ip_len + ch_len, dtype=np.float64)
    modes[:op_len] = Energy_modes_op
    modes[op_len:op_len+ip_len] = Energy_modes_ip
    modes[op_len+ip_len:] = Energy_modes_CH


    hbar_omega1 = np.min(modes)
    modes[np.argmin(modes)] = 2 * np.max(modes)

    for j in range(2, 21):
        hbar_omega20 = np.min(modes)
        modes[np.argmin(modes)] = 2 * np.max(modes)

    # ind = np.where(Energy <= hbar_omega20)
    # if len(ind[0]) > 0:
    #     Temperature[ind] = hbar_omega1 / (k * np.log(2.0))

    # Use loops for njit compatibility
    for i in range(len(Energy)):
        if Energy[i] <= hbar_omega20:
            Temperature[i] = hbar_omega1 / (k * np.log(2.0))

    return Temperature

@njit
def Energy_bins(a, Energy_modes_op, Energy_modes_ip, Energy_modes_CH, M, Energy_max):
    """
    Returns an array E_tab containing the central, minimum, and maximum values
    of M energy bins. It returns an array of shape (3, M+1) with the center, 
    minimum, and maximum of the energy bins.
    
    Parameters:
    - a: Grain size.
    - Energy_modes: Energy modes for the grain size.
    - M: Number of energy bins.
    - Energy_max: Maximum energy for the bins.
    
    Returns:
    - E_tab: A 2D NumPy array with the center, minimum, and maximum of energy bins.
    """

    jmax = 10  # First 10 energy bins are calculated as in DL01

    #assert M > 11, f'Must have M > 1. M = {M}.'

    # Collect modes 
    # modes = np.concatenate([ Energy_modes_op[:jmax+1],  Energy_modes_ip[:jmax+1], Energy_modes_CH])
    op_len = np.size(Energy_modes_op[:jmax+1])
    ip_len = np.size(Energy_modes_ip[:jmax+1])
    ch_len = np.size(Energy_modes_CH)
    # Preallocate the array for modes
    modes = np.empty(op_len + ip_len + ch_len, dtype=np.float64)
    # Fill the preallocated array
    modes[:op_len] = Energy_modes_op[:jmax+1]
    modes[op_len:op_len+ip_len] = Energy_modes_ip[:jmax+1]
    modes[op_len+ip_len:] = Energy_modes_CH


    hbar_omega = np.zeros(2 * jmax)

    # Fill hbar_omega with the minimum of the modes array
    for j in range(1, 2 * jmax):
        index = np.argmin(modes)
        hbar_omega[j] = modes[index]
        modes[index] = 2 * np.max(modes)

    # Initialize the E_tab array: (3, M+1) for center, min, and max values
    E_tab = np.zeros((3, M + 1))

    # Calculate the first few energy bins according to DL01
    E_tab[1, 1] = 1.5 * hbar_omega[1] - 0.5 * hbar_omega[2]
    for j in range(1, 3):
        E_tab[0, j] = hbar_omega[j]
        E_tab[1, j + 1] = 0.5 * (hbar_omega[j] + hbar_omega[j + 1])
        E_tab[2, j] = E_tab[1, j + 1]

    for j in range(3, jmax + 1):
        E_tab[2, j] = 0.5 * (hbar_omega[2 * j - 2] + hbar_omega[2 * j - 1])
        E_tab[0, j] = 0.5 * (hbar_omega[2 * j - 3] + hbar_omega[2 * j - 2])
        E_tab[1, j + 1] = E_tab[2, j]

    # Calculate the next bins (up to jmax) that are linearly spaced
    DeltaE = E_tab[2, jmax] - E_tab[1, jmax]

    # Define K_tab for linear spacing
    K_tab = 12 + np.arange(M - 11)

    # Calculate E_M_max for the highest energy bin
    E_M_max = np.exp((M - K_tab + 1) * np.log(E_tab[2, jmax] + (K_tab - 10) * DeltaE) - 
                     (M - K_tab) * np.log(E_tab[2, jmax] + (K_tab - 11) * DeltaE))

    # Check if the highest energy bin is below Energy_max
    if E_M_max[0] < Energy_max:
        #print(f'{M} energy bins are not enough, using {M+50} bins instead (in function Energy_bins).')
        return Energy_bins(a, Energy_modes_op, Energy_modes_ip, Energy_modes_CH, M + 50, Energy_max)
    else:
        indices = np.where(E_M_max >= Energy_max)[0]  # Get indices where E_M_max >= Energy_max
        index = indices[-1]  # Find the largest index where E_M_max >= Energy_max


    BigK = K_tab[index]

    # Fill in the linearly spaced bins
    E_tab[2, jmax + 1:BigK + 1] = E_tab[2, jmax] + (np.arange(BigK + 1) - jmax)[jmax + 1:BigK + 1] * DeltaE
    E_tab[1, jmax + 2:BigK + 1] = E_tab[2, jmax + 1:BigK]
    E_tab[0, jmax + 1:BigK + 1] = 0.5 * (E_tab[1, jmax + 1:BigK + 1] + E_tab[2, jmax + 1:BigK + 1])

    # Fill in the last bins with logarithmic spacing if BigK < M
    if BigK < M:
        E_tab[2, BigK + 1:M + 1] = E_tab[2, BigK] * np.exp((np.arange(M + 1) - BigK)[BigK + 1:M + 1] * 
                                                           np.log(E_tab[2, BigK] / E_tab[2, BigK - 1]))
        E_tab[1, BigK + 1:M + 1] = E_tab[2, BigK:M]
        E_tab[0, BigK + 1:M + 1] = np.sqrt(E_tab[2, BigK + 1:M + 1] * E_tab[1, BigK + 1:M + 1])

    return E_tab


def Btilde_ISRF_func(a, Z, Energy_modes_op, Energy_modes_ip, Energy_modes_CH, M, Energy_max):
    """
    Computes the matrix Bji for the average interstellar radiation field (ISRF),
    for a grain of radius `a` and charge `Z`, using the thermal continuous approximation.

    Parameters:
    - a: Grain radius.
    - Z: Grain charge.
    - Energy_modes: Dictionary containing energy modes.
    - M: Initial number of elements in energy bins.
    - Energy_max: Maximum energy.

    Returns:
    - Btilde matrix (upward and downward transition rates).
    """

    # Compute the energy bins
    Energies = Energy_bins(a, Energy_modes_op, Energy_modes_ip, Energy_modes_CH, M, Energy_max)
    E_tab = Energies[0, :]  # Energy bins
    M = np.size(E_tab) - 1  # Update M based on energy bin calculation

    E_max = Energies[2, :]
    E_min = Energies[1, :]

    # Temperatures for the thermal approximation
    # T_tab = np.concatenate(([0], Temp(a, E_tab[1:],  Energy_modes_op, Energy_modes_ip, Energy_modes_CH)))
    aux_temp = Temp(a, E_tab[1:],  Energy_modes_op, Energy_modes_ip, Energy_modes_CH)
    T_tab = np.zeros(np.size(aux_temp)+1, dtype=np.float64)
    T_tab[1:] = aux_temp

    NEarr = 500  # Number of elements in the integral over energies
    Energy_min = h * c / 1e2  # Corresponds to a wavelength of 100 cm

    # Initialize transition rates arrays
    T_upward = np.zeros((M + 1, M + 1))  # Upward transition rates
    T_downward = np.zeros(M + 1)         # Downward transition rates
    T_downward_td = np.zeros((M + 1, M + 1))  # Thermal discrete downward rates

    # Pre-allocate arrays for the Cabs and nu_uisrf values
    Cabs = np.zeros(NEarr)
    E_array = np.zeros(NEarr)
    DE_over_E = np.zeros(NEarr)
    u_arr = np.zeros(NEarr)
    GuL = np.zeros(NEarr)


    # --- Upward Transitions ---
    for u in range(1, M + 1):
        Eu = E_tab[u]
        Emax_u = E_max[u]
        Emin_u = E_min[u]
        DeltaEu = Emax_u - Emin_u
        Tu = T_tab[u]

        # Case L = 0 (excitations from the ground state)
        W1 = Emin_u
        W4 = Emax_u
        E_array = makelogtab(W1, W4, NEarr)
        DE_over_E = DX_over_X(W1, W4, NEarr)
        Cabs = pi * a**2 * Qabs(a, Z, E_array / eV)
        T_upward[u, 0] = c / Eu * np.sum(nu_uisrf(E_array / eV) * Cabs) * DE_over_E

        T_downward_td[0, u] = 8 * pi / (h**3 * c**2) / Eu * (E_max[1] - E_min[1]) / DeltaEu * \
                              np.sum(E_array**4 / (np.exp(E_array / (k * Tu)) - 1) * Cabs) * DE_over_E

        # Case L > 0
        for L in range(1, u):
            EL = E_tab[L]
            DeltaEL = E_max[L] - E_min[L]
            W1 = Emin_u - E_max[L]
            W2 = min([Emin_u - E_min[L], Emax_u - E_max[L]])
            W3 = max([Emin_u - E_min[L], Emax_u - E_max[L]])
            W4 = Emax_u - E_min[L]

            if max([W1, Energy_min]) < W4:
                E_array = makelogtab(max([W1, Energy_min]), W4, NEarr)
                DE_over_E = DX_over_X(max([W1, Energy_min]), W4, NEarr)
                GuL = np.zeros(NEarr)

                ind1 = np.where((E_array > W1) & (E_array < W2))
                if len(ind1[0]) > 0:
                    GuL[ind1] = (E_array[ind1] - W1) / (DeltaEu * DeltaEL)

                ind2 = np.where((E_array > W2) & (E_array < W3))
                if len(ind2[0]) > 0:
                    GuL[ind2] = min([DeltaEu, DeltaEL]) / (DeltaEu * DeltaEL)

                ind3 = np.where((E_array > W3) & (E_array < W4))
                if len(ind3[0]) > 0:
                    GuL[ind3] = (W4 - E_array[ind3]) / (DeltaEu * DeltaEL)

                u_arr = nu_uisrf(E_array / eV)
                Cabs = pi * a**2 * Qabs(a, Z, E_array / eV)
                T_upward[u, L] = c * DeltaEu / (Eu - EL) * np.sum(GuL * Cabs * u_arr) * DE_over_E

                T_downward_td[L, u] = (8 * pi / (h**3 * c**2)) * DeltaEL / (Eu - EL) * \
                                      np.sum(GuL * Cabs * E_array**4 / (np.exp(E_array / (k * Tu)) - 1)) * DE_over_E

    # Case u = M, L > 0: include upward transitions above the highest level
    EM = E_tab[M]
    for Lind in range(1, M):
        EL = E_tab[Lind]
        W1 = E_min[M] - E_max[Lind]
        Wc = E_min[M] - E_min[Lind]
        integral_1 = 0

        if max([W1, Energy_min]) < Wc:
            E_array = makelogtab(max([W1, Energy_min]), Wc, NEarr)
            DE_over_E = DX_over_X(max([W1, Energy_min]), Wc, NEarr)
            u_arr = nu_uisrf(E_array / eV)
            Cabs = pi * a**2 * Qabs(a, Z, E_array / eV)
            integral_1 = np.sum((E_array - W1) / (Wc - W1) * Cabs * u_arr) * DE_over_E

        integral_2 = 0
        if Wc < 13.6 * eV:
            E_array = makelogtab(Wc, 13.6 * eV, NEarr)
            DE_over_E = DX_over_X(Wc, 13.6 * eV, NEarr)
            u_arr = nu_uisrf(E_array / eV)
            Cabs = pi * a**2 * Qabs(a, Z, E_array / eV)
            integral_2 = np.sum(Cabs * u_arr) * DE_over_E

        T_upward[M, Lind] = c / (EM - EL) * (integral_1 + integral_2)

    # Case u = M, L = 0
    Wc = E_min[M]
    integral_2 = 0
    if Wc < 13.6 * eV:
        E_array = makelogtab(Wc, 13.6 * eV, NEarr)
        DE_over_E = DX_over_X(Wc, 13.6 * eV, NEarr)
        u_arr = nu_uisrf(E_array / eV)
        Cabs = pi * a**2 * Qabs(a, Z, E_array / eV)
        integral_2 = np.sum(Cabs * u_arr) * DE_over_E

    T_upward[M, 0] = c / EM * integral_2

    # Intrabin contributions to upward transitions
    for u in range(2, M + 1):
        DeltaEu_1 = E_max[u - 1] - E_min[u - 1]
        if DeltaEu_1 > Energy_min:
            E_array = makelogtab(Energy_min, DeltaEu_1, NEarr)
            DE_over_E = DX_over_X(Energy_min, DeltaEu_1, NEarr)
            Cabs = pi * a**2 * Qabs(a, Z, E_array / eV)
            T_upward[u, u - 1] += c / (E_tab[u] - E_tab[u - 1]) * \
                                  np.sum((1 - E_array / DeltaEu_1) * Cabs * nu_uisrf(E_array / eV)) * DE_over_E

    # --- Downward Transitions ---
    for u in range(1, M + 1):
        T_downward[u] = 1 / (E_tab[u] - E_tab[u - 1]) * np.sum((E_tab[u] - E_tab[:u]) * T_downward_td[:u, u])

        DeltaEu = E_max[u] - E_min[u]
        if Energy_min < DeltaEu:
            E_array = makelogtab(Energy_min, DeltaEu, NEarr)
            DE_over_E = DX_over_X(Energy_min, DeltaEu, NEarr)
            Cabs = pi * a**2 * Qabs(a, Z, E_array / eV)
            T_downward[u] += 8 * pi / (h**3 * c**2) / (E_tab[u] - E_tab[u - 1]) * \
                             np.sum((1 - E_array / DeltaEu) * E_array**4 * Cabs / (np.exp(E_array / (k * T_tab[u])) - 1)) * DE_over_E

    # Define Btilde_{j,i}
    Tinv = np.flip(T_upward, axis=0)
    Binv = np.cumsum(Tinv, axis=0)
    Btab = np.flip(Binv, axis=0)
    T_downward = np.outer(T_downward, np.ones(M + 1))
    T_downward[0, :] = np.ones(M + 1)

    Btilde = Btab / T_downward

    return Btilde, M


def distribution(a, Z, Chi,  Energy_modes_op, Energy_modes_ip, Energy_modes_CH, M, Energy_max):
    """
    Returns the distribution function for the energy levels.

    Parameters:
    - a: Grain radius.
    - Z: Grain charge.
    - Chi: ISRF intensity scaling factor.
    -  Energy_modes_op, Energy_modes_ip, Energy_modes_CH: Energy modes calculated based on grain size.
    - M: Initial number of elements in energy bins.
    - Energy_max: Maximum energy.

    Returns:
    - Dictionary containing Energy_bins and P_tab (the distribution).
    """


    # Iterate while X_tab[M] is greater than a small threshold (1e-14)

    # Scale the initial Energy_max
    Energy_max = Energy_max/3

    # Initialize the distribution (X_tab) with ones
    X_tab = np.ones(M + 1)

    while X_tab[M] > 1e-14:    
        print(f"E_M = {Energy_max / eV:.2f} eV is not high enough in 'distribution'. Using {3 * Energy_max / eV:.2f} eV instead")
        Energy_max *= 3
        aux1, M = Btilde_ISRF_func(a, Z,  Energy_modes_op, Energy_modes_ip, Energy_modes_CH, M, Energy_max) 
        print("M = ", M)
        Btilde = Chi * aux1 
        X_tab = np.ones(M + 1)  # Reset X_tab since M might have changed
        
        # Loop through energy levels and compute the distribution
        for j in range(1, M + 1):
            X_tab[j] = np.sum(Btilde[j, 0:j] * X_tab[0:j])  # Reform array and compute sum
            X_tab[0:j + 1] =  X_tab[0:j + 1] / np.sum(X_tab[0:j + 1])  # Normalize the distribution
    
    '''
    Energy_max = Energy_tab_max
    X_tab = np.ones(M + 1)  # Initialize the distribution with ones
    while X_tab[M] > 1e-14: 
        M+=50
        aux1, M = Btilde_ISRF_func(a, Z,  Energy_modes_op, Energy_modes_ip, Energy_modes_CH, M, Energy_max) 
        X_tab = np.ones(M + 1)
        Btilde = Chi * aux1 
        for j in range(1, M + 1):
            X_tab[j] = np.sum(Btilde[j, 0:j] * X_tab[0:j])  # Reform array and compute sum
            X_tab[0:j + 1] =  X_tab[0:j + 1] / np.sum(X_tab[0:j + 1])  # Normalize the distribution
    
    
    aux1, M = Btilde_ISRF_func(a, Z,  Energy_modes_op, Energy_modes_ip, Energy_modes_CH, M, Energy_max) 
    X_tab = np.ones(M + 1)
    Btilde = Chi * aux1 
    for j in range(1, M + 1):
        X_tab[j] = np.sum(Btilde[j, 0:j] * X_tab[0:j])  # Reform array and compute sum
        X_tab[0:j + 1] =  X_tab[0:j + 1] / np.sum(X_tab[0:j + 1])  # Normalize the distribution
    '''
    print("M = ", M)
    print("x_tab[M]= ", X_tab[M])

    # Get the energy bins
    Energy_bins_eval = Energy_bins(a,  Energy_modes_op, Energy_modes_ip, Energy_modes_CH, M, Energy_max)  

    # Return the energy bins and the computed distribution (P_tab)
    return {'Energy_bins': Energy_bins_eval, 'P_tab': X_tab}, Energy_max, M

def IRemission(a, Z, nu_tab,  Energy_modes_op, Energy_modes_ip, Energy_modes_CH, distribution):
    """
    Computes the infrared emission spectrum for a grain of radius `a` and charge `Z`.

    Parameters:
    - a: Grain radius.
    - Z: Grain charge.
    - Chi: ISRF intensity scaling factor.
    - nu_tab: Array of frequency values.
    -  Energy_modes_op, Energy_modes_ip, Energy_modes_CH: Energy modes calculated based on grain size.
    - distribution: Distribution function calculated for energy levels.

    Returns:
    - Array of infrared emission values (F_nu) corresponding to the given frequencies (nu_tab).
    """

    # Extract the distribution data
    P_tab = distribution['P_tab']
    Mval = np.size(P_tab) - 1
    Energy_bins_val = distribution['Energy_bins']
    E_tab = Energy_bins_val[0, 1:Mval+1].reshape(Mval)  # Reforming as needed for Python
    T_tab = Temp(a, E_tab,  Energy_modes_op, Energy_modes_ip, Energy_modes_CH)  

    N_nu = np.size(nu_tab)
    F_nu = np.zeros(N_nu)  # Initialize the emission array

    # Broadcasting arrays for matrix multiplication
    new_nu_tab = nu_tab[:, None] * np.ones((1, Mval))
    new_E_tab = np.ones((N_nu, 1)) * E_tab[None, :]
    new_P_tab = np.ones((N_nu, 1)) * P_tab[1:Mval+1][None, :]
    new_T_tab = np.ones((N_nu, 1)) * T_tab[None, :]

    # Handle cases to avoid floating point overflow
    ind = np.where((new_E_tab < h * new_nu_tab) | (h * new_nu_tab / (k * new_T_tab) > 700))

    if ind[0].size > 0:
        new_P_tab[ind] = 0  # Set P_tab values to zero where overflow occurs
        new_nu_tab[ind] = 700 * k * new_T_tab[ind] / h  # Adjust nu_tab to avoid overflow
        F_nu = np.sum(new_P_tab / (np.exp(h * new_nu_tab / (k * new_T_tab)) - 1), axis=1)

    # Compute the final emission spectrum
    F_nu = (2 * h * nu_tab**3 / c**2) * pi * a**2 * Qabs(a, Z, h * nu_tab / eV) * F_nu

    return F_nu


def FGIR_integrals(a, Z, Chi,  Energy_modes_op, Energy_modes_ip, Energy_modes_CH, Mval, Energy_max):
    """
    Computes the integrals \int F_nu/nu^2 dnu and \int F_nu/nu dnu.

    Parameters:
    - a: Grain radius.
    - Z: Grain charge.
    - Chi: ISRF intensity scaling factor.
    -  Energy_modes_op, Energy_modes_ip, Energy_modes_CH: Energy modes calculated based on grain size.
    - M: Initial number of elements in energy bins.
    - Energy_max: Maximum energy.

    Returns:
    - A dictionary containing the FIR and GIR integrals.
    """

    # Compute the distribution
    dist, Energy_max, Mval = distribution(a, Z, Chi,  Energy_modes_op, Energy_modes_ip, Energy_modes_CH, Mval, Energy_max)

    # Update M and Energy_max based on the distribution function
    lambda_max = 1e-1  # 0.1 cm
    nu_min = c / lambda_max
    nu_max = Energy_max / h
    N_lambda = 1000  # Number of frequency points

    # Generate the logarithmic frequency table and differential factor
    nu_tab = makelogtab(nu_min, nu_max, N_lambda)
    Dnu_over_nu = DX_over_X(nu_min, nu_max, N_lambda)  
    # Compute the infrared emission spectrum
    F_nu = IRemission(a, Z, nu_tab,  Energy_modes_op, Energy_modes_ip, Energy_modes_CH, dist)

    # Perform the integrals
    FIR_integral = np.sum(F_nu / nu_tab) * Dnu_over_nu
    GIR_integral = np.sum(F_nu) * Dnu_over_nu

    return {'FIR_integral': FIR_integral, 'GIR_integral': GIR_integral}, Energy_max, Mval


# Function to set up the infrared arrays
def set_up_IR_arrays():
    """
    Sets up the infrared arrays for the calculations.
    
    Returns:
    - Dictionary containing the parameters and the arrays.
    """

    # Grain size range
    a_min = 3.5e-8  # cm
    a_max = 1e-6    # cm
    Na = 30
    a_tab = makelogtab(a_min, a_max, Na)

    # Scaling factor (chi) range
    chi_0 = 1e-5
    chi_N = 1e10
    Nchi = 30

    # Calculating chi_min and chi_max as done in IDL
    chi_min = np.exp(np.log(chi_0) - 1.0 / (2 * Nchi) * np.log(chi_N / chi_0))
    chi_max = np.exp(np.log(chi_N) - 1.0 / (2 * Nchi) * np.log(chi_N / chi_0))

    # Generating the chi_tab array
    chi_tab = makelogtab(chi_min, chi_max, Nchi)

    IR_arrays.a_min = a_min
    IR_arrays.a_max = a_max
    IR_arrays.Na = Na
    IR_arrays.a_tab = a_tab
    IR_arrays.chi_min = chi_min
    IR_arrays.chi_max = chi_max
    IR_arrays.Nchi = Nchi
    IR_arrays.chi_tab = chi_tab
    pass

set_up_IR_arrays()

def compute_FGIR_integrals():
    """
    Computes and stores FIR and GIR integrals for charged and neutral grains
    over a range of grain sizes (a) and ISRF scaling factors (Chi).
    """

    Na = IR_arrays.Na
    Nchi = IR_arrays.Nchi
    a_tab = IR_arrays.a_tab
    chi_tab = IR_arrays.chi_tab

    # If the FIR and GIR integrals are already stored in the data directory, 
    # load them and return

    bool1 = os.path.exists(SpDust_data_dir + f'FIR_integral_charged_{Na}a_{Nchi}chi.txt')
    bool2 = os.path.exists(SpDust_data_dir + f'GIR_integral_charged_{Na}a_{Nchi}chi.txt')
    bool3 = os.path.exists(SpDust_data_dir + f'FIR_integral_neutral_{Na}a_{Nchi}chi.txt')
    bool4 = os.path.exists(SpDust_data_dir + f'GIR_integral_neutral_{Na}a_{Nchi}chi.txt')

    if bool1 and bool2 and bool3 and bool4:
        FIR_integral_charged = np.loadtxt(SpDust_data_dir + f'FIR_integral_charged_{Na}a_{Nchi}chi.txt')
        GIR_integral_charged = np.loadtxt(SpDust_data_dir + f'GIR_integral_charged_{Na}a_{Nchi}chi.txt')
        FIR_integral_neutral = np.loadtxt(SpDust_data_dir + f'FIR_integral_neutral_{Na}a_{Nchi}chi.txt')
        GIR_integral_neutral = np.loadtxt(SpDust_data_dir + f'GIR_integral_neutral_{Na}a_{Nchi}chi.txt')

        IR_arrays.FIR_integral_charged = FIR_integral_charged
        IR_arrays.GIR_integral_charged = GIR_integral_charged
        IR_arrays.FIR_integral_neutral = FIR_integral_neutral
        IR_arrays.GIR_integral_neutral = GIR_integral_neutral

        # Print shape of the arrays
        print(f'FIR_integral_charged shape: {FIR_integral_charged.shape}')
        # Print Na and Nchi
        print(f'Na: {Na}, Nchi: {Nchi}')
        return

    # Loop over grain sizes (a_tab) and Chi values (chi_tab)

    # Define a function to do the loop for a given grain size
    def loop(ia):
        a = a_tab[ia]
        Energy_modes_op, Energy_modes_ip, Energy_modes_CH = Energy_modes(a)
        Mval = 100
        Energy_max = 13.6 * eV  # Maximum energy in erg

        result = np.zeros((Nchi, 4))

        for ichi in range(Nchi):
            Chi = chi_tab[ichi]

            # Compute integrals for charged grains (Z = 1)
            FGIR_integral_charged, Energy_max, Mval = FGIR_integrals(a, 1, Chi,  Energy_modes_op, Energy_modes_ip, Energy_modes_CH, Mval, Energy_max)
            result[ichi, 0] = FGIR_integral_charged['FIR_integral']
            result[ichi, 1] = FGIR_integral_charged['GIR_integral']

            # Compute integrals for neutral grains (Z = 0)
            FGIR_integral_neutral, Energy_max, Mval = FGIR_integrals(a, 0, Chi,  Energy_modes_op, Energy_modes_ip, Energy_modes_CH, Mval, Energy_max)
            result[ichi, 2] = FGIR_integral_neutral['FIR_integral']
            result[ichi, 3] = FGIR_integral_neutral['GIR_integral']

        return result
    
    # Parallelize the loop
    result = np.array(parallel_map(loop, np.arange(Na)) ) # shape: (Na, Nchi, 4)
        
    """
    # Initialize arrays for FIR and GIR integrals (charged and neutral)
    FIR_integral_charged = np.zeros((Na, Nchi))
    GIR_integral_charged = np.zeros((Na, Nchi))
    FIR_integral_neutral = np.zeros((Na, Nchi))
    GIR_integral_neutral = np.zeros((Na, Nchi))

    for ia in range(Na):
        a = a_tab[ia]
        Energy_modes_op, Energy_modes_ip, Energy_modes_CH = Energy_modes(a)  
        Mval = 100
        Energy_max = 13.6 * eV  # Maximum energy in erg

        for ichi in range(Nchi):
            Chi = chi_tab[ichi]

            # Compute integrals for charged grains (Z = 1)
            FGIR_integral_charged, Energy_max, Mval = FGIR_integrals(a, 1, Chi,  Energy_modes_op, Energy_modes_ip, Energy_modes_CH, Mval, Energy_max)
            FIR_integral_charged[ia, ichi] = FGIR_integral_charged['FIR_integral']
            GIR_integral_charged[ia, ichi] = FGIR_integral_charged['GIR_integral']

            # Compute integrals for neutral grains (Z = 0)
            FGIR_integral_neutral, Energy_max, Mval = FGIR_integrals(a, 0, Chi,  Energy_modes_op, Energy_modes_ip, Energy_modes_CH, Mval, Energy_max)
            FIR_integral_neutral[ia, ichi] = FGIR_integral_neutral['FIR_integral']
            GIR_integral_neutral[ia, ichi] = FGIR_integral_neutral['GIR_integral']

    IR_arrays.FIR_integral_charged = FIR_integral_charged
    IR_arrays.GIR_integral_charged = GIR_integral_charged
    IR_arrays.FIR_integral_neutral = FIR_integral_neutral
    IR_arrays.GIR_integral_neutral = GIR_integral_neutral
    """

    if rank0:
        IR_arrays.FIR_integral_charged = result[:, :, 0]
        IR_arrays.GIR_integral_charged = result[:, :, 1]
        IR_arrays.FIR_integral_neutral = result[:, :, 2]
        IR_arrays.GIR_integral_neutral = result[:, :, 3]

        # Save FIR and GIR integrals for charged grains
        np.savetxt(SpDust_data_dir + f'FIR_integral_charged_{Na}a_{Nchi}chi.txt', IR_arrays.FIR_integral_charged)
        np.savetxt(SpDust_data_dir + f'GIR_integral_charged_{Na}a_{Nchi}chi.txt', IR_arrays.GIR_integral_charged)

        # Save FIR and GIR integrals for neutral grains
        np.savetxt(SpDust_data_dir + f'FIR_integral_neutral_{Na}a_{Nchi}chi.txt', IR_arrays.FIR_integral_neutral)
        np.savetxt(SpDust_data_dir + f'GIR_integral_neutral_{Na}a_{Nchi}chi.txt', IR_arrays.GIR_integral_neutral)

    return

"""
def precompile_njit_functions():
    a=3e-7
    op, ip, ch = Energy_modes(a)
    aux = Energy_bins(a, op, ip, ch, 100, 13.6*eV)
    aux = Temp(a,np.array([50]), op, ip, ch)
    pass

precompile_njit_functions()
"""

compute_FGIR_integrals()

def FGIR_integrals_interpol(a, Z, Chi):
    """
    Interpolates or extrapolates the FIR and GIR integrals for a given grain size `a`, charge `Z`, and ISRF scaling factor `Chi`.

    Parameters:
    - a: Grain size.
    - Z: Grain charge.
    - Chi: ISRF intensity scaling factor.
    
    Returns:
    - A dictionary containing the interpolated/extrapolated FIR and GIR integrals.
    """

    FIR_integral_charged, GIR_integral_charged, FIR_integral_neutral, GIR_integral_neutral, \
    a_min, a_max, Na, chi_min, chi_max, Nchi \
    = IR_arrays.FIR_integral_charged, IR_arrays.GIR_integral_charged, IR_arrays.FIR_integral_neutral, IR_arrays.GIR_integral_neutral, \
        IR_arrays.a_min, IR_arrays.a_max, IR_arrays.Na, IR_arrays.chi_min, IR_arrays.chi_max, IR_arrays.Nchi

    # Select the appropriate FIR and GIR integral tables based on charge
    if Z != 0:
        FIR_integral_tab = FIR_integral_charged
        GIR_integral_tab = GIR_integral_charged
    else:
        FIR_integral_tab = FIR_integral_neutral
        GIR_integral_tab = GIR_integral_neutral

    # Tabulation for grain size (a_tab) and chi (chi_tab)
    a_tab = IR_arrays.a_tab
    chi_tab = IR_arrays.chi_tab

    # Finding the indices and coefficients alpha and beta for a and chi s.t. a = alpha *a_i + (1-alpha)*a_i+1 
    if a <= min(a_tab):
        ia = 0
        alpha = 1.0
    elif a >= max(a_tab):
        ia = Na - 2
        alpha = 0.0
    else:
        ia = np.where(a_tab <= a)[0][-1]
        alpha = 1.0 - Na * np.log(a / a_tab[ia]) / np.log(a_max / a_min)

    # --- Case of low radiation field: FIR, GIR are linear in chi ---
    if Chi <= np.min(chi_tab):
        FIR_integral_0 = np.exp(alpha * np.log(FIR_integral_tab[ia, 0]) + (1.0 - alpha) * np.log(FIR_integral_tab[ia+1, 0]))
        FIR_integral = (Chi / chi_tab[0]) * FIR_integral_0

        GIR_integral_0 = np.exp(alpha * np.log(GIR_integral_tab[ia, 0]) + (1.0 - alpha) * np.log(GIR_integral_tab[ia+1, 0]))
        GIR_integral = (Chi / chi_tab[0]) * GIR_integral_0

        return {'FIR_integral': FIR_integral, 'GIR_integral': GIR_integral}
    
    # --- Case of high radiation field: Approximate FIR, GIR by a power-law ---
    if Chi >= np.max(chi_tab):
        print(f'You are using chi= {Chi}! The code is written for chi < 1E5. It assumes a power-law in chi for higher values.')

        FIR_integral_last = np.exp(alpha * np.log(FIR_integral_tab[ia, Nchi-1]) 
                                   + (1.0 - alpha) * np.log(FIR_integral_tab[ia+1, Nchi-1]))
        FIR_integral_before_last = np.exp(alpha * np.log(FIR_integral_tab[ia, Nchi-2]) 
                                          + (1.0 - alpha) * np.log(FIR_integral_tab[ia+1, Nchi-2]))
        
        FIR_index = np.log(FIR_integral_last / FIR_integral_before_last) / np.log(chi_tab[Nchi-1] / chi_tab[Nchi-2])
        FIR_integral = (Chi / chi_tab[Nchi-1]) ** FIR_index * FIR_integral_last

        GIR_integral_last = np.exp(alpha * np.log(GIR_integral_tab[ia, Nchi-1]) 
                                   + (1.0 - alpha) * np.log(GIR_integral_tab[ia+1, Nchi-1]))
        GIR_integral_before_last = np.exp(alpha * np.log(GIR_integral_tab[ia, Nchi-2]) 
                                          + (1.0 - alpha) * np.log(GIR_integral_tab[ia+1, Nchi-2]))
        
        GIR_index = np.log(GIR_integral_last / GIR_integral_before_last) / np.log(chi_tab[Nchi-1] / chi_tab[Nchi-2])
        GIR_integral = (Chi / chi_tab[Nchi-1]) ** GIR_index * GIR_integral_last

        return {'FIR_integral': FIR_integral, 'GIR_integral': GIR_integral}

    # Case for intermediate Chi values (interpolation)
    ichi = np.where(chi_tab <= Chi)[0][-1]
    beta = 1.0 - Nchi * np.log(Chi / chi_tab[ichi]) / np.log(chi_max / chi_min)

    FIR_integral = np.exp(
        alpha * (beta * np.log(FIR_integral_tab[ia, ichi]) 
                 + (1 - beta) * np.log(FIR_integral_tab[ia, ichi + 1])) 
                 + (1 - alpha) * (beta * np.log(FIR_integral_tab[ia + 1, ichi]) 
                                  + (1 - beta) * np.log(FIR_integral_tab[ia + 1, ichi + 1]))
    )

    GIR_integral = np.exp(
        alpha * (beta * np.log(GIR_integral_tab[ia, ichi]) 
                 + (1 - beta) * np.log(GIR_integral_tab[ia, ichi + 1])) 
                 + (1 - alpha) * (beta * np.log(GIR_integral_tab[ia + 1, ichi]) 
                                  + (1 - beta) * np.log(GIR_integral_tab[ia + 1, ichi + 1]))
    )

    return {'FIR_integral': FIR_integral, 'GIR_integral': GIR_integral}

def FGIR(env, a, Zg):
    """
    Computes FIR and GIR for a grain with radius `a`, charge `Zg`, and environment `env`.

    Parameters:
    - env: Environment object containing Chi, T (temperature), and nh (number density of hydrogen).
    - a: Grain radius.
    - Zg: Grain charge.

    Returns:
    - A dictionary containing FIR and GIR values.
    """

    # Extract environment parameters
    Chi = env['Chi']
    Tval = env['T']
    nh = env['nh']

    # Compute various physical quantities
    Inertia_val = Inertia(a) 
    acx_val = acx(a)  
    tau_H = 1.0 / (nh * mp * np.sqrt(2.0 * k * Tval / (pi * mp)) 
                   * 4.0 * pi * acx_val**4 / (3.0 * Inertia_val))

    # Interpolate FIR and GIR integrals
    FGIR_integrals_aux = FGIR_integrals_interpol(a, Zg, Chi)  
    IntF = FGIR_integrals_aux['FIR_integral']
    IntG = FGIR_integrals_aux['GIR_integral']

    # Compute FIR and GIR values
    FIR = 2.0 * tau_H / (pi * Inertia_val) * IntF
    GIR = h * tau_H / (3.0 * pi * Inertia_val * k * Tval) * IntG  # Corrected version

    # Adjust for disklike grains if a < a2
    if a < grainparams.a2:
        FIR *= 5.0 / 3.0

    return {'FIR': FIR, 'GIR': GIR}

def FGIR_averaged(env, a, fZ):
    """
    Computes the averaged FIR and GIR values for a grain with radius `a`, using the charge distribution `fZ` and environment `env`.

    Parameters:
    - env: Environment object containing Chi, T (temperature), and nh (number density of hydrogen).
    - a: Grain radius.
    - fZ: Charge distribution array.

    Returns:
    - A dictionary containing the averaged FIR and GIR values.
    """

    # Extract the first element of fZ (f0 corresponds to the neutral grain contribution)
    f0 = fZ[1, 0]

    # Compute FIR and GIR for neutral and charged grains
    FGIR_neutral = FGIR(env, a, 0)  # For neutral grain (Z = 0)
    FGIR_charged = FGIR(env, a, 1)  # For charged grain (Z = 1)

    # Average the FIR and GIR values
    FIR = f0 * FGIR_neutral['FIR'] + (1.0 - f0) * FGIR_charged['FIR']
    GIR = f0 * FGIR_neutral['GIR'] + (1.0 - f0) * FGIR_charged['GIR']

    return {'FIR': FIR, 'GIR': GIR}