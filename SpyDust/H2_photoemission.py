from .Grain import acx, asurf
from .charge_dist import refr_indices, hnu_pdt, hnu_pet, Y, Qabs, sigma_pdt, nu_uisrf, E_min, JPEisrf
from .util import DX_over_X, makelogtab, cgsconst
import numpy as np


eV = cgsconst.eV
mp = cgsconst.mp
me = cgsconst.me
q = cgsconst.q
k = cgsconst.k
c = cgsconst.c
hbar = cgsconst.h / (2 * np.pi)

hnu_tab = refr_indices.hnu_tab
la_tab = refr_indices.la_tab   

# Define the GH2 function
def GH2(env, a, beta):
    """
    Excitation rate due to random H2 formation.
    
    Parameters:
    - env: an object or dictionary containing environment properties T, nh, y, gamma
    - a, beta: parameter for which `acx(a, beta)` will be calculated

    Returns:
    - Excitation rate (GH2)
    """
    
    # Extract environment variables
    T = env['T']
    y = env['y']
    gamma = env['gamma']
    
    # Calculate ax from acx(a)
    ax = acx(a, beta)

    # Define Ef (formation energy) and JJplus1 (quantum number term)
    Ef = 0.2 * eV
    JJplus1 = 1e2  # Equivalent to 1d2 in IDL

    # Calculate GH2 according to the formula
    GH2_value = (gamma / 4) * (1 - y) * Ef / (k * T) * (1 + JJplus1 * hbar**2 / (2 * mp * Ef * ax**2))
    
    return GH2_value

# Function FGpeZ
def FGpeZ(env, a, beta, Z):
    """
    Calculate Fpe and Gpe for a given environment, grain size 'a', and charge 'Z'.
    
    Parameters:
    - env: Dictionary containing environment parameters like T, nh, Chi.
    - a: Grain size.
    - Z: Charge state.

    Returns:
    - A dictionary with Fpe and Gpe values.
    """

    # Environment variables
    T = env['T']
    nh = env['nh']
    Chi = env['Chi']
    asurf_val = asurf(a, beta)  

    # Get Jpeisrf 
    aux1, aux2 = JPEisrf(a)
    Jpepos = Chi * aux1
    Jpeneg = Chi * aux2

    # --- Gpe ---
    hnu_pet_val = max(1e-5, hnu_pet(Z, a)) 
    hnu_pdt_val = max(1e-5, hnu_pdt(Z, a)) 
    Nnu = 500
    hnu_pet_tab = makelogtab(hnu_pet_val, 13.6, Nnu) 
    Dnu_over_nu_pet = DX_over_X(hnu_pet_val, 13.6, Nnu)  
    hnu_pdt_tab = makelogtab(hnu_pdt_val, 13.6, Nnu)
    Dnu_over_nu_pdt = DX_over_X(hnu_pdt_val, 13.6, Nnu)

    Ytab = np.interp(np.log(hnu_pet_tab), np.log(hnu_tab), Y(Z,a))
    Qtab = Qabs(a, Z, hnu_pet_tab)  

    if Z >= 0:
        E_low = -(Z + 1) * q**2 / a
        E_high = (hnu_pet_tab - hnu_pet_val) * eV
        Epe = 0.5 * E_high * (E_high - 2 * E_low) / (E_high - 3 * E_low)  # In erg
        second_term = 0
        third_term = Jpepos[int(Z)] * (Z + 1) * q**2 / asurf_val
    else:
        E_min_val = E_min(Z, a) 
        E_low = E_min_val * eV
        E_high = (E_min_val + hnu_pet_tab - hnu_pet_val) * eV
        Epe = 0.5 * (E_high + E_low)  # In erg
        second_term = Dnu_over_nu_pdt * np.sum(sigma_pdt(hnu_pdt_tab, Z, a) * nu_uisrf(hnu_pdt_tab) / eV / hnu_pdt_tab 
                                               * (hnu_pdt_tab - hnu_pdt_val + E_min_val))
        third_term = Jpeneg[int(-Z)] * (Z + 1) * q**2 / asurf_val

    first_term = c * np.pi * a**2 * Dnu_over_nu_pet * np.sum(Ytab * Qtab * nu_uisrf(hnu_pet_tab) / eV / hnu_pet_tab * Epe)

    Gpe = me / (4 * nh * np.sqrt(8 * np.pi * mp * k * T) * asurf_val**2 * k * T) * (first_term + second_term + third_term)

    # --- Fpe ---
    if Z >= 0:
        Jpe = Jpepos[int(Z)]
    else:
        Jpe = Jpeneg[int(-Z)]

    Fpe = me / mp * Jpe / (2 * np.pi * asurf_val**2 * nh * np.sqrt(2 * k * T / (np.pi * mp)))

    return {'Fpe': Fpe, 'Gpe': Gpe}


def FGpe_averaged(env, a, beta, fZ):
    """
    Computes the averaged values of Fpe and Gpe over grain charges.

    Parameters:
    - env: Environment parameters.
    - a: Grain size.
    - fZ: Grain charge distribution, a 2D array where fZ[0, :] contains grain charges 
          and fZ[1, :] contains their corresponding probabilities.

    Returns:
    - A dictionary with averaged Fpe and Gpe values.
    """
    NZ = fZ.shape[1]  # Number of grain charges
    Fpe = 0.0
    Gpe = 0.0

    # Loop over all grain charge states
    for i in range(NZ):
        FGpe = FGpeZ(env, a, beta, fZ[0, i])  # Get the Fpe and Gpe for charge state Zg
        Fpe += fZ[1, i] * FGpe['Fpe']   # Weighted Fpe
        Gpe += fZ[1, i] * FGpe['Gpe']   # Weighted Gpe

    return {'Fpe': Fpe, 'Gpe': Gpe}
