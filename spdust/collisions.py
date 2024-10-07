from spdust import SpDust_data_dir
from utils.util import cgsconst, makelogtab, DX_over_X
from spdust.grain_properties import grainparams, N_C, acx
from spdust.charge_dist import Qabs, nu_uisrf
from spdust.infrared import Temp, Energy_modes, IR_arrays

import numpy as np
from scipy.special import erf
from numba import jit
import os


h = cgsconst.h
c = cgsconst.c
q = cgsconst.q
k = cgsconst.k
eV = cgsconst.eV
mp = cgsconst.mp

pi = np.pi


# Define the Tev function
def Tev(a, Chi):
    """
    Calculate the equilibrium temperature for a given grain size `a` and radiation field `Chi`.
    
    Parameters:
    - a: Grain size.
    - Chi: Radiation field strength.

    Returns:
    - Temperature (Tev) in K.
    """

    # Thermal spikes limit
    hnu_tab_vals = makelogtab(1e-2, 13.6, 500)  
    Qabs_vals = Qabs(a, 1, hnu_tab_vals)        
    nu_uisrf_vals = nu_uisrf(hnu_tab_vals)      

    E_photon = (np.sum(Qabs_vals * nu_uisrf_vals) / 
                np.sum(Qabs_vals * nu_uisrf_vals / hnu_tab_vals)) * eV

    Energy_modes_op, Energy_modes_ip, Energy_modes_ch  = Energy_modes(a)     
    T_q = Temp(a, np.array([E_photon]), Energy_modes_op, Energy_modes_ip, Energy_modes_ch)  

    # Steady emission limit (Take cross-section of ionized PAH)
    Dnu_over_nu = DX_over_X(1e-2, 13.6, 500)  
    Qu_star = Chi * np.sum(Qabs_vals * nu_uisrf_vals) * Dnu_over_nu

    hnu_0 = np.array([1e-4]) 
    Qlambda_0 = Qabs(a, 1, hnu_0) * (c / (hnu_0 * eV / h))**2

    # Constants for zeta(6) and gamma(6)
    zeta6 = np.pi**6 / 945
    gamma6 = 120

    # Calculate steady state temperature T_c
    T_c = h * c / k * (Qu_star / (8 * np.pi * h * c * Qlambda_0 * gamma6 * zeta6))**(1 / 6)

    # Return the maximum of T_q and T_c
    return max(T_q, T_c)

    
a_min, a_max, Na, chi_min, chi_max, Nchi \
= IR_arrays.a_min, IR_arrays.a_max, IR_arrays.Na, IR_arrays.chi_min, IR_arrays.chi_max, IR_arrays.Nchi
FIR_integral_charged, GIR_integral_charged, FIR_integral_neutral,GIR_integral_neutral \
= IR_arrays.FIR_integral_charged, IR_arrays.GIR_integral_charged, IR_arrays.FIR_integral_neutral, IR_arrays.GIR_integral_neutral

a_tab = IR_arrays.a_tab
chi_tab = IR_arrays.chi_tab

def compute_Tev():
    """
    Compute the Tev array and save it to a file.
    
    Parameters:
    - SpDust_data_dir: Directory path where the output file will be saved.
    - a_min, a_max: Minimum and maximum grain sizes.
    - Na: Number of grain size intervals.
    - chi_min, chi_max: Minimum and maximum Chi values.
    - Nchi: Number of Chi intervals.
    """

    output_file = os.path.join(SpDust_data_dir, "Tev_{}a_{}chi.txt".format(Na, Nchi))

    # If the Tev table already stored, load it and return
    if os.path.exists(output_file):
        print(f"Tev table already exists at {output_file}.")
        IR_arrays.Tev_tab = np.loadtxt(output_file)
        return
    
    # Initialize the Tev_tab array (Na x Nchi)
    Tev_tab = np.zeros((Na, Nchi))

    # Nested loops to compute Tev for each combination of `a` and `Chi`
    for ia in range(Na):
        a = a_tab[ia]
        for ichi in range(Nchi):
            Chi = chi_tab[ichi]
            Tev_tab[ia, ichi] = Tev(a, Chi)  

    # Save the result to a file
    # output_file = f"{SpDust_data_dir}/Tev_{Na}a_{Nchi}chi.txt"
    
    
    # Save using numpy's savetxt function
    np.savetxt(output_file, Tev_tab, fmt='%.6e')
    IR_arrays.Tev_tab = Tev_tab
    print(f"Tev table saved to {output_file}")
    return 

# Call the compute_Tev function
compute_Tev()

def Tev_interpol(a, Chi):
    """
    Perform interpolation to compute Tev for given grain size `a` and radiation field `Chi`.

    Parameters:
    - a: Grain size.
    - Chi: Radiation field strength.

    Returns:
    - Interpolated Tev value.
    """

    # Finding the indices and coefficients for `a`
    if a <= np.min(a_tab):
        ia = 0
        alpha = 1.0
    elif a >= np.max(a_tab):
        ia = Na - 2
        alpha = 0.0
    else:
        ia = np.max(np.where(a_tab <= a)[0])
        alpha = 1.0 - Na * np.log(a / a_tab[ia]) / np.log(a_max / a_min)

    # Finding the indices and coefficients for `Chi`
    if Chi <= np.min(chi_tab):
        print(f'Warning: you are using Chi={Chi}! The code is written for Chi > 1E-6. If you wish to extend its validity to lower values, set chi_min to a lower value, run "compute_Tev", and recompile.')
        ichi = 0
        beta = 1.0
    elif Chi >= np.max(chi_tab):
        print(f'Warning: you are using Chi={Chi}! The code is written for Chi < 1E8. If you wish to extend its validity to higher values, set chi_max to a higher value, run "compute_Tev", and recompile.')
        ichi = Nchi - 2
        beta = 0.0
    else:
        ichi = np.max(np.where(chi_tab <= Chi)[0])
        beta = 1.0 - Nchi * np.log(Chi / chi_tab[ichi]) / np.log(chi_max / chi_min)

    # Interpolating the Tev value
    Tev_log_interp = (
        alpha * (beta * np.log(IR_arrays.Tev_tab[ia, ichi]) + (1.0 - beta) * np.log(IR_arrays.Tev_tab[ia, ichi + 1])) +
        (1.0 - alpha) * (beta * np.log(IR_arrays.Tev_tab[ia + 1, ichi]) + (1.0 - beta) * np.log(IR_arrays.Tev_tab[ia + 1, ichi + 1]))
    )

    # Return the exponentiated result
    return np.exp(Tev_log_interp)

# Define the Tev_effective function
def Tev_effective(env, a, get_info=False):
    """
    Compute the effective evaporation temperature.

    Parameters:
    - env: A dictionary containing environment parameters like nh, T, Chi, Tev (optional).
    - a: Grain size.
    - get_info: If True, return the ratio R_{coll/abs}/Nsites instead of the temperature.

    Returns:
    - Effective evaporation temperature (Tev) or ratio if get_info is True.
    """

    a2, d = grainparams.a2, grainparams.d
    
    # Check if the constant evaporation temperature is present
    if 'Tev' in env:
        return env['Tev']

    # Extract environmental parameters
    nh = env['nh']
    T = env['T']
    Chi = env['Chi']

    # Generate the necessary arrays
    hnu_tab_vals = makelogtab(1e-2, 13.6, 500)  
    Dnu_over_nu = DX_over_X(1e-2, 13.6, 500) 
    Qabs_vals = Qabs(a, 1, hnu_tab_vals)  
    nu_uisrf_vals = nu_uisrf(hnu_tab_vals)  

    # Calculate the ratio of collision rate to absorption rate
    ratio = nh * np.sqrt(8 * k * T / (pi * mp)) / (
        Chi * np.sum(Qabs_vals * nu_uisrf_vals / (hnu_tab_vals * eV)) * Dnu_over_nu * c
    )

    # Number of sites (Nsites)
    if a < a2:
        Nsites = N_C(a)  
    else:
        Nsites = N_C(a) * 3 * d / a

    # If `get_info` is True, return the ratio/Nsites
    if get_info:
        return ratio / Nsites

    # Determine the effective temperature
    if Nsites > ratio:
        Tev = Tev_interpol(a, Chi)  
    else:
        Tev = T  # No more sticking collisions, atoms bounce back

    return Tev

# Define the FGn function
def FGn(env, a, T_ev, Zg_tab, tumbling=True):
    """
    Calculate F_n and G_n for neutral impactors, for an array of grain charges.
    
    Parameters:
    - env: Dictionary containing environment parameters like T, xh, y.
    - a: Grain size.
    - T_ev: Precomputed evaporation temperature.
    - Zg_tab: Array of grain charges.
    - tumbling: Boolean flag for tumbling condition.

    Returns:
    - Dictionary with Fn and Gn values.
    """

    #--- polarizabilities for neutral species : Hydrogen, Helium and H2 ---

    a0       = 0.52918e-8      # Bohr radius
    alpha_H  = 4.5 * a0**3      # J. Chem. Phys. 49, 4845 (1968); DOI:10.1063/1.1669968  
    alpha_He = 1.38 * a0**3     # M A Thomas et al 1972 J. Phys. B: At. Mol. Phys. 5 L229-L232 
    alpha_H2 = 5.315 * a0**3    # W.C. Marlow, Proc. Phys. Soc 1965, vol. 86 (ground state)
    
    # Extract environmental parameters
    Tval = env['T']
    xh = env['xh']
    y = env['y']
    acx_val = acx(a)  

    # Calculate e_n and e_e
    e_n = np.sqrt(q**2 / (2 * acx_val**4 * k * Tval) * Zg_tab**2)
    e_e = np.sqrt(q**2 / (2 * acx_val**4 * k * T_ev) * Zg_tab**2)

    # Contribution of atomic Hydrogen
    eps_n = e_n * np.sqrt(alpha_H)
    eps_e = e_e * np.sqrt(alpha_H)
    FnH_ev = (1 - xh - y) * (np.exp(-eps_n**2) + np.sqrt(pi) * eps_n * erf(eps_n)) / \
             (np.exp(-eps_e**2) + np.sqrt(pi) * eps_e * erf(eps_e)) * \
             (np.exp(-eps_e**2) + 2 * eps_e**2)
    GnH_in = 0.5 * (1 - xh - y) * (np.exp(-eps_n**2) + 2 * eps_n**2)

    if tumbling:
        FnH_in = FnH_ev * 2 / 3 / (1 + np.sqrt(2 / 3) * eps_n)
    else:
        FnH_in = 0.0

    # Contribution of neutral Helium
    eps_n = e_n * np.sqrt(alpha_He)
    eps_e = e_e * np.sqrt(alpha_He)
    FnHe_ev = 1 / 6 * (np.exp(-eps_n**2) + np.sqrt(pi) * eps_n * erf(eps_n)) / \
              (np.exp(-eps_e**2) + np.sqrt(pi) * eps_e * erf(eps_e)) * \
              (np.exp(-eps_e**2) + 2 * eps_e**2)
    GnHe_in = 1 / 12 * (np.exp(-eps_n**2) + 2 * eps_n**2)

    if tumbling:
        FnHe_in = FnHe_ev * 2 / 3 / (1 + np.sqrt(2 / 3) * eps_n)
    else:
        FnHe_in = 0.0

    # Contribution of molecular Hydrogen
    eps_n = e_n * np.sqrt(alpha_H2)
    eps_e = e_e * np.sqrt(alpha_H2)
    FnH2_ev = y / np.sqrt(2) * (np.exp(-eps_n**2) + np.sqrt(np.pi) * eps_n * erf(eps_n)) / \
              (np.exp(-eps_e**2) + np.sqrt(np.pi) * eps_e * erf(eps_e)) * \
              (np.exp(-eps_e**2) + 2 * eps_e**2)
    GnH2_in = 0.5 * y / np.sqrt(2) * (np.exp(-eps_n**2) + 2 * eps_n**2)

    if tumbling:
        FnH2_in = FnH2_ev * 2 / 3 / (1 + np.sqrt(2 / 3) * eps_n)
    else:
        FnH2_in = 0.0

    # Sum of contributions
    Fn_ev = FnH_ev + FnHe_ev + FnH2_ev
    Fn_in = FnH_in + FnHe_in + FnH2_in
    Fn = Fn_ev + Fn_in
    Gn = GnH_in + GnHe_in + GnH2_in + 0.5 * T_ev / Tval * Fn_ev

    return {'Fn': Fn, 'Gn': Gn}

def FGn_averaged(env, a, T_ev, fZ, tumbling=True):
    """
    Compute the averaged Fn and Gn over grain charges given the grain charge distribution.
    
    Parameters:
    - env: Dictionary containing environment parameters like T, xh, y.
    - a: Grain size.
    - T_ev: Precomputed evaporation temperature.
    - fZ: 2D array with grain charges in fZ[0, *] and charge distribution in fZ[1, *].
    - tumbling: Boolean flag for tumbling condition.

    Returns:
    - Dictionary with averaged Fn and Gn values.
    """
    
    # Call FGn to get Fn and Gn for specific grain charges
    FGn_result = FGn(env, a, T_ev, fZ[0, :], tumbling=tumbling)

    # Calculate averaged Fn and Gn
    Fn_averaged = np.sum(FGn_result['Fn'] * fZ[1, :])
    Gn_averaged = np.sum(FGn_result['Gn'] * fZ[1, :])

    return {'Fn': Fn_averaged, 'Gn': Gn_averaged}

def g1g2(psi, mu_tilde):
    """
    Compute g1 and g2 based on the given psi and mu_tilde array.
    
    Parameters:
    - psi: Scalar value representing psi.
    - mu_tilde: Array of values representing mu_tilde (size Ndipole).

    Returns:
    - Dictionary with g1 and g2 arrays.
    """

    # Number of elements in mu_tilde
    Ndipole = len(mu_tilde)

    # Initialize g1 and g2 arrays
    g1 = np.zeros(Ndipole)
    g2 = np.zeros(Ndipole)

    # Index where mu_tilde <= abs(psi)
    index = np.where(mu_tilde <= abs(psi))[0]
    if len(index) > 0:
        mu_i = mu_tilde[index]
        if psi < 0:
            g1[index] = 1 - psi
            g2[index] = 1 - psi + 0.5 * psi**2 + (1 / 6) * mu_i**2
        else:
            if psi > 600:  # To avoid floating underflow
                g1[index] = 0
                g2[index] = 0
            else:
                g1[index] = np.exp(-psi) * np.sinh(mu_i) / mu_i
                g2[index] = g1[index]

    # Index where mu_tilde > abs(psi)
    index = np.where(mu_tilde > abs(psi))[0]
    if len(index) > 0:
        mu_i = mu_tilde[index]
        g1[index] = (1 - np.exp(-(psi + mu_i)) + mu_i - psi + 0.5 * (mu_i - psi)**2) / (2 * mu_i)
        g2[index] = g1[index] + (mu_i - psi)**3 / (12 * mu_i)

    # Return a dictionary with g1 and g2
    return g1, g2

# Define the h1 function
def h1(phi, mu_tilde):
    """
    Compute h1(phi, mu_tilde) based on AHD09 Eq.(103).
    
    Parameters:
    - phi: Scalar value representing phi.
    - mu_tilde: Array or scalar representing mu_tilde.

    Returns:
    - h1: Computed value based on the formula. Shape depends on the input.
    """

    mu = mu_tilde

    # Calculate u_0
    u_0 = 0.5 * (-phi + np.sqrt(phi**2 + 4 * mu))

    # Compute the coefficients
    DL98 = 1 + np.sqrt(pi) / 2 * phi
    coeff_1 = mu / 4 + phi**2 / (4 * mu) + (1 - mu) / (2 * mu)
    coeff_erf = np.sqrt(pi) * phi * (3 - 2 * mu) / (8 * mu)
    coeff_exp = -(4 + phi**2 + phi * np.sqrt(phi**2 + 4 * mu)) / (8 * mu)

    # Compute the result for h1
    h1_result = DL98 + coeff_1 + coeff_erf * erf(u_0) + coeff_exp * np.exp(-u_0**2)

    return h1_result

def h2(phi, mu_tilde):
    """
    Compute h2(phi, mu_tilde) as per AHD09 Eq.(104).
    
    Parameters:
    - phi: Scalar value representing phi.
    - mu_tilde: Array of values representing mu_tilde.

    Returns:
    - Array of h2 values corresponding to the input phi and mu_tilde.
    """
    
    mu = mu_tilde

    # Calculate intermediate values
    u_0 = 0.5 * (-phi + np.sqrt(phi**2 + 4 * mu))
    Dl98 = 1 + 3 * np.sqrt(np.pi) / 4 * phi + 0.5 * phi**2
    coeff_1 = mu**2 / 12 + mu / 4 + phi**2 / (2 * mu) + (1 - mu) / (2 * mu) - phi**2 / 4
    coeff_erf = np.sqrt(np.pi) * phi / (32 * mu) * (4 * mu**2 - 12 * mu + 15 + 2 * phi**2)
    coeff_exp = (phi**2 * (2 * mu - 9) - 16 + (2 * mu - 7) * phi * np.sqrt(phi**2 + 4 * mu)) / (32 * mu)

    # Compute h2
    h2_values = Dl98 + coeff_1 + coeff_erf * erf(u_0) + coeff_exp * np.exp(-u_0**2)

    return h2_values

# Define the FGi function
def FGi(env, a, T_ev, Zg, mu_tab):
    """
    Compute Fi and Gi for a given grain charge Zg, based on mu_tab.
    
    Parameters:
    - env: Dictionary containing environment parameters like T, xh, xC (assumed to represent xM in the original code).
    - a: Grain size.
    - T_ev: Precomputed evaporation temperature.
    - Zg: Grain charge.
    - mu_tab: Array of mu values.

    Returns:
    - Dictionary with Fi and Gi values.
        Fi: shape (len(mu_tab),) array
    """

    # Extract environmental parameters
    T = env['T']
    xh = env['xh']
    xM = env['xC']  

    # Compute acx and mu_tilde
    acx_val = acx(a)  # Assuming acx(a) is defined elsewhere
    mu_tilde = q * mu_tab / (acx_val**2 * k * T)

    # Polarizabilities for neutral species corresponding to incoming ions (H and C)
    a0 = 0.53e-8  # Bohr radius in cm
    alpha_H = 4.5 * a0**3  # Polarizability for Hydrogen
    alpha_C = 1.54e-24  # Polarizability for Carbon

    # Neutral grains case
    if Zg == 0:
        phi = np.sqrt(2) * q / np.sqrt(acx_val * k * T)
        Fi = (xh + xM * np.sqrt(12)) * h1(phi, mu_tilde)  
        Gi_in = 0.5 * (xh + xM * np.sqrt(12)) * h2(phi, mu_tilde)  
    else:
        # Charged grains case
        psi = Zg * q**2 / (acx_val * k * T)
        e_i = np.sqrt(q**2 / (2 * acx_val**4 * k * T_ev) * Zg**2)

        # Contribution of H+ and C+
        eps_i = e_i * np.sqrt([alpha_H, alpha_C])
        Fi_contrib = [xh, xM] * (np.exp(-eps_i**2) + 2 * eps_i**2) / (np.exp(-eps_i**2) + np.sqrt(np.pi) * eps_i * erf(eps_i))
        g1, g2 = g1g2(psi, mu_tilde)  
        Fi = np.sum(Fi_contrib) * g1
        Gi_in = 0.5 * (xh + xM * np.sqrt(12)) * g2

    # Calculate Gi_ev and return Fi and Gi
    Gi_ev = 0.5 * T_ev / T * Fi
    Gi = Gi_in + Gi_ev

    return {'Fi': Fi, 'Gi': Gi}

def FGi_averaged(env, a, T_ev, mu_tab, fZ):
    """
    Compute the averaged Fi and Gi over grain charges given the grain charge distribution.
    
    Parameters:
    - env: Dictionary containing environment parameters like T, xh, xC (representing xM).
    - a: Grain size.
    - T_ev: Precomputed evaporation temperature.
    - mu_tab: Array of mu values.
    - fZ: 2D array where fZ[0, *] contains grain charges and fZ[1, *] contains their respective probabilities.

    Returns:
    - Dictionary with averaged Fi and Gi arrays.
    """

    # Initialize Fi and Gi as arrays of zeros with the same dimensions as mu_tab
    Fi = np.zeros_like(mu_tab)
    Gi = np.zeros_like(mu_tab)

    # Number of grain charge values in fZ
    NZg = fZ.shape[1]

    # Loop over all grain charges in fZ[0, *]
    for i in range(NZg):
        # Compute Fi and Gi for the current grain charge fZ[0, i]
        FGi_result = FGi(env, a, T_ev, fZ[0, i], mu_tab)

        # Average Fi and Gi by summing over the charge distribution
        Fi += fZ[1, i] * FGi_result['Fi']
        Gi += fZ[1, i] * FGi_result['Gi']

    # Return the averaged Fi and Gi
    return {'Fi': Fi, 'Gi': Gi}