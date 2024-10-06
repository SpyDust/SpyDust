import numpy as np
from utils.util import cgsconst, makelogtab, maketab, DX_over_X
from numba import jit, njit
from scipy.special import kv as beselk
from core.compile_spydust import SpDust_data_dir
import os
from spdust.grain_properties import acx

k = cgsconst.k
c = cgsconst.c
mp = cgsconst.mp
q = cgsconst.q
pi = np.pi
I = 1j  # complex number equivalent to IDL's "I = complex(0d, 1d)"

class smalletabs:
    smalle_tab = None
    Gamma_tab = None
    Gamma_max = None

class plasma_tabs:
    Ipos_tab = None
    Ineg_tab = None
    rot_tab = None
    e_1tab = None
    rot_min = None
    rot_max = None
    e_1min = None
    e_1max = None

class gp_arrays:
    psi_min = 1e-5
    psi_max = 1e6
    Npsi    = 110

    phi_min = 1e-2
    phi_max = 5e2
    Nphi    = 300

    Omega_min = 1e-10
    Omega_max = 1e5
    NOmega    = 150

    gp_pos, gp_neg, gp_neutral = None, None, None


@jit(nopython=True)
def compute_small_e():
    
    # Create arrays
    gamma_min = 1e-20
    gamma_max = 10 ** 1.5
    Ngamma = 300
    Gamma_tab = makelogtab(gamma_min, gamma_max, Ngamma)
    smalle_tab = np.zeros(Ngamma)  # Equivalent to dblarr in IDL
    
    # Parameters
    Ny = 20000
    ymed = 1.0
    ymax = 1e20
    
    # Create u and Du arrays
    u_part_1 = maketab(0, ymed, Ny)
    u_part_2 = makelogtab(ymed, ymax, Ny)
    u = np.exp(I * pi / 6) * np.concatenate((u_part_1, u_part_2))
    
    # Differential array Du
    Du_part_1 = ymed / Ny + np.zeros(Ny)
    Du_part_2 = DX_over_X(ymed, ymax, Ny) * u_part_2
    Du = np.exp(I * pi / 6) * np.concatenate((Du_part_1, Du_part_2))
    
    # Function components
    fcos = (u ** 2 - 1) / (u ** 2 + 1) ** 2
    fsin = u / (u ** 2 + 1) ** 2
    
    # Main computation loop
    for ig in range(Ngamma):
        Gamma = Gamma_tab[ig]
        time = Gamma * (u + u ** 3 / 3)
        
        # Integrals
        intcos = 4 * np.sum(np.real(np.exp(I * time) * fcos * Du)) ** 2
        intsin = 16 * np.sum(np.imag(np.exp(I * time) * fsin * Du)) ** 2
        
        # Store results in smalle_tab
        smalle_tab[ig] = intcos + intsin
    
    # Output the maximum gamma for debugging purposes
    gamma_max = np.max(Gamma_tab)
    
    smalletabs.Gamma_tab = Gamma_tab
    smalletabs.smalle_tab = smalle_tab
    smalletabs.Gamma_max = gamma_max
    print("I(Zg<0, parabolic) stored")
    
    return Gamma_tab, smalle_tab, gamma_max

#compute_small_e()

@jit(nopython=True)
def replace_zeros(Ipos_tab):
    """
    Replace zeros in a 2D array with a small value (1e-30), element by element.
    """
    # Iterate through the array and replace zeros
    for i in range(Ipos_tab.shape[0]):
        for j in range(Ipos_tab.shape[1]):
            if Ipos_tab[i, j] == 0:
                Ipos_tab[i, j] = 1e-30

    return Ipos_tab

@jit(nopython=False)
def compute_int_plasma():

    # Call `compute_small_e` to get the values for small e_1, Zg < 0
    Gamma_tab, smalle_tab, Gamma_max = compute_small_e()


    # Create arrays
    e_1min = 1e-15
    e_1max = 1e4
    Ne_1 = 100
    rot_min = 1e-7
    rot_max = 1.0 / e_1min * Gamma_max
    Nrot = 100

    e_1small = 1e-2
    rot_small = 1e3

    rot_tab = makelogtab(rot_min, rot_max, Nrot)
    e_1tab = makelogtab(e_1min, e_1max, Ne_1)
    Ipos_tab = np.zeros((Nrot, Ne_1))
    Ineg_tab = np.zeros((Nrot, Ne_1))

    # Parameters
    Ny = 10000
    ymax = 1e50
    ymed = 800.0

    y= np.zeros(2*Ny)
    y[:Ny] = maketab(0, ymed, Ny)
    y[Ny:] = makelogtab(ymed, ymax, Ny)

    # Zg > 0 case
    Dz = -I * np.concatenate((ymed / Ny + np.zeros(Ny), DX_over_X(ymed, ymax, Ny) * makelogtab(ymed, ymax, Ny)))
    log = 0.5 * np.log(1.0 + 4.0 / y**2) + I * np.arctan(2.0 / y)
    z = 1.0 - I * y

    for ie in range(Ne_1):
        e_1 = e_1tab[ie]
        Aval = e_1 / (e_1 + 2.0)
        time = 1.0 / np.sqrt(e_1 * (e_1 + 2.0)) * (log + 2.0 * (e_1 + 1.0) * z / (z**2 - 1.0))
        fcos = (z**2 - Aval) / (z**2 + Aval)**2
        fsin = z / (z**2 + Aval)**2
        for ir in range(Nrot):
            rot = rot_tab[ir]
            intcos = 4.0 * Aval * np.sum(np.real(np.exp(I * rot * time) * fcos * Dz))**2
            intsin = 16.0 * Aval**2 * np.sum(np.imag(np.exp(I * rot * time) * fsin * Dz))**2
            Ipos_tab[ir, ie] = intcos + intsin

    # Handling small e_1 and high rotation for Zg < 0
    inde = np.where(e_1tab < e_1small)[0][-1]
    indrot = np.where(rot_tab < rot_small)[0][-1]

    # Zg < 0 non-small e_1
    Dz = np.exp(I * pi / 4.0) * np.concatenate((ymed / Ny + np.zeros(Ny), DX_over_X(ymed, ymax, Ny) * makelogtab(ymed, ymax, Ny)))
    log = 0.5 * np.log(1.0 + 4.0 / y**2 + 2.0 * np.sqrt(2.0) / y) - I * np.arctan(np.sqrt(2.0) / (np.sqrt(2.0) + y))
    z = 1.0 + np.exp(I * pi / 4.0) * y

    for ie in range(Ne_1):
        e_1 = e_1tab[ie]
        Aval = e_1 / (e_1 + 2.0)
        time = 1.0 / np.sqrt(e_1 * (e_1 + 2.0)) * (log - 2.0 * (e_1 + 1.0) * z / (z**2 - 1.0))
        fcos = (1.0 - Aval * z**2) / (1.0 + Aval * z**2)**2
        fsin = z / (1.0 + Aval * z**2)**2
        for ir in range(indrot):
            rot = rot_tab[ir]
            intcos = 4.0 * Aval * np.sum(np.real(np.exp(I * rot * time) * fcos * Dz))**2
            intsin = 16.0 * Aval**2 * np.sum(np.imag(np.exp(I * rot * time) * fsin * Dz))**2
            Ineg_tab[ir, ie] = intcos + intsin

    # Extending Zg < 0 for nearly parabolic case
    for ie in range(inde):
        e_1 = e_1tab[ie]
        for ir in range(indrot, Nrot):
            rot = rot_tab[ir]
            Gamma = rot * e_1
            if Gamma < Gamma_max:
                Ineg_tab[ir, ie] = np.interp(Gamma, Gamma_tab, smalle_tab)

    # Some useful things for further interpolation
    rot_min = np.min(rot_tab)
    rot_max = np.max(rot_tab)
    e_1min = np.min(e_1tab)
    e_1max = np.max(e_1tab)

    # Replace zeros with small values for precision
    Ipos_tab = replace_zeros(Ipos_tab)
    Ineg_tab = replace_zeros(Ineg_tab)

    print('I(rot, e, Zg <> 0) stored')

    plasma_tabs.Ipos_tab = Ipos_tab
    plasma_tabs.Ineg_tab = Ineg_tab
    plasma_tabs.rot_tab = rot_tab
    plasma_tabs.e_1tab = e_1tab
    plasma_tabs.rot_min = rot_min
    plasma_tabs.rot_max = rot_max
    plasma_tabs.e_1min = e_1min
    plasma_tabs.e_1max = e_1max

    return 

# Example usage
# Ipos_tab, Ineg_tab, rot_min, rot_max, e_1min, e_1max = compute_int_plasma()

'''
@jit(nopython=True)
def compute_int_plasma_2():

    # Call `compute_small_e` to get the values for small e_1, Zg < 0
    Gamma_tab, smalle_tab, Gamma_max = compute_small_e()

    # Create arrays
    e_1min = 1e-15
    e_1max = 1e4
    Ne_1 = 100
    rot_min = 1e-7
    rot_max = 1.0 / e_1min * Gamma_max
    Nrot = 100


    e_1small = 1e-2
    rot_small = 1e3

    rot_tab = makelogtab(rot_min, rot_max, Nrot)
    e_1tab = makelogtab(e_1min, e_1max, Ne_1)
    Ipos_tab = np.zeros((Nrot, Ne_1))
    Ineg_tab = np.zeros((Nrot, Ne_1))

    # Parameters
    Ny = 10000
    ymax = 1e50
    ymed = 800.0


    y_log =  makelogtab(ymed, ymax, Ny)

    y = np.concatenate((maketab(0, ymed, Ny), y_log))

    # Zg > 0 case
    Dz = -I * np.concatenate((ymed / Ny + np.zeros(Ny), DX_over_X(ymed, ymax, Ny) * y_log))

    atan_y = np.arctan(2.0 / y)
    log_1 = 0.5 * np.log(1.0 + 4.0 / y**2) + I * atan_y
    z = 1.0 - I * y

    # Vectorize as much as possible
    for ie in range(Ne_1):
        e_1 = e_1tab[ie]
        A = e_1 / (e_1 + 2.0)
        time = 1.0 / np.sqrt(e_1 * (e_1 + 2.0)) * (log_1 + 2.0 * (e_1 + 1.0) * z / (z**2 - 1.0))
        fcos = (z**2 - A) / (z**2 + A)**2
        fsin = z / (z**2 + A)**2

        # Vectorized computation for all rot values at once
        rot_exp = np.exp(I * np.outer(rot_tab, time)) * Dz[np.newaxis, :]
        real_part = np.sum( np.real(rot_exp * fcos[np.newaxis, :] ) )
        imag_part = np.sum( np.imag(rot_exp * fsin[np.newaxis, :]) )

        Ipos_tab[:, ie] = 4.0 * A * real_part**2 + 16.0 * A**2 * imag_part**2

    # Handle Zg < 0 non-small e_1 (precompute to avoid nested loops)
    Dz_neg = np.exp(I * pi / 4.0) * np.concatenate((ymed / Ny + np.zeros(Ny), DX_over_X(ymed, ymax, Ny) * y_log))
    atan_y_neg = np.arctan(np.sqrt(2.0) / (np.sqrt(2.0) + y))
    log_2 = 0.5 * np.log(1.0 + 4.0 / y**2 + 2.0 * np.sqrt(2.0) / y) - I * atan_y_neg
    z_neg = 1.0 + np.exp(I * pi / 4.0) * y

    indrot = np.where(rot_tab < rot_small)[0][-1]  # Fix index for small rot_tab case

    for ie in range(Ne_1):
        e_1 = e_1tab[ie]
        A = e_1 / (e_1 + 2.0)
        time_neg = 1.0 / np.sqrt(e_1 * (e_1 + 2.0)) * (log_2 - 2.0 * (e_1 + 1.0) * z_neg / (z_neg**2 - 1.0))
        fcos_neg = (1.0 - A * z_neg**2) / (1.0 + A * z_neg**2)**2
        fsin_neg = z_neg / (1.0 + A * z_neg**2)**2

        rot_exp_neg = np.exp(I * np.outer(rot_tab[:indrot], time_neg)) * Dz_neg[np.newaxis, :]
        real_part_neg = np.sum(np.real(rot_exp_neg * fcos_neg[np.newaxis, :]))
        imag_part_neg = np.sum(np.imag(rot_exp_neg * fsin_neg[np.newaxis, :]))

        Ineg_tab[:indrot, ie] = 4.0 * A * real_part_neg**2 + 16.0 * A**2 * imag_part_neg**2

    # Extend Zg < 0 for parabolic case
    inde = np.where(e_1tab < e_1small)[0][-1]  # Fix index for small e_1tab case

    for ie in range(inde):
        e_1 = e_1tab[ie]
        for ir in range(indrot, Nrot):
            rot = rot_tab[ir]
            Gamma = rot * e_1
            if Gamma < Gamma_max:
                Ineg_tab[ir, ie] = np.interp(Gamma, Gamma_tab, smalle_tab)

    # Replace zeros with small values for precision
    Ipos_tab = replace_zeros(Ipos_tab)
    Ineg_tab = replace_zeros(Ineg_tab)

    return Ipos_tab, Ineg_tab, rot_min, rot_max, e_1min, e_1max
'''



compute_int_plasma()

@jit(nopython=False)
def int_plasma(rot_new, e_1_new, Zg):
    """
    Returns I(omega * b/v, e-1, Zg non zero) by interpolating arrays computed by `compute_int_plasma`.

    Parameters:
    - rot_new: Array of new rotation values.
    - e_1_new: Array of new e_1 values.
    - Zg: Grain charge (Zg > 0 or Zg < 0).
    - Ipos_tab, Ineg_tab: Precomputed positive and negative I tables.
    - rot_tab, e_1tab: Precomputed rotation and e_1 tables.
    - rot_min, rot_max: Minimum and maximum rotation values.
    - e_1min, e_1max: Minimum and maximum e_1 values.

    Returns:
    - result: Interpolated integrals for the input values.
    """
    Ipos_tab, Ineg_tab, rot_tab, rot_min, rot_max, e_1tab, e_1min, e_1max = plasma_tabs.Ipos_tab, plasma_tabs.Ineg_tab, \
         plasma_tabs.rot_tab, plasma_tabs.rot_min, plasma_tabs.rot_max, plasma_tabs.e_1tab, plasma_tabs.e_1min, plasma_tabs.e_1max              

    # Ensure rot_new and e_1_new have the same number of elements
    Nelem = np.size(rot_new)
    if Nelem != np.size(e_1_new):
        raise ValueError(f"Error in int_plasma: e_1_new and rot_new should have the same dimensions. "
                         f"Nelem (rot_new) = {Nelem}, N_elements(e_1_new) = {np.size(e_1_new)}")
    
    result = np.zeros(Nelem)

    # Case: e_1_new >= e_1max and rot_new < 100
    ind = np.where((e_1_new >= e_1max) & (rot_new < 100))[0]
    if np.size(ind) > 0:
        rot = rot_new[ind]
        result[ind] = rot**2 * (beselk(0, rot)**2 + beselk(1, rot)**2)

    # Case: rot_new <= rot_min
    ind = np.where(rot_new <= rot_min)[0]
    if np.size(ind) > 0:
        e_1 = e_1_new[ind]
        result[ind] = e_1 * (e_1 + 2) / (e_1 + 1)**2

    # Case: Intermediate values within computed tables' range
    ind = np.where((rot_new < rot_max) & (rot_new > rot_min) & 
                   (e_1_new < e_1max) & (e_1_new > e_1min))[0]
    if np.size(ind) > 0:
        rot = rot_new[ind]
        e_1 = e_1_new[ind]
        
        # Interpolation indices and interpolation factors
        Drot_over_rot = np.log(rot_tab[1] / rot_tab[0])
        irot = np.floor(np.log(rot / min(rot_tab)) / Drot_over_rot).astype(int)
        
        De_over_e = np.log(e_1tab[1] / e_1tab[0])
        ie = np.floor(np.log(e_1 / min(e_1tab)) / De_over_e).astype(int)
        
        alpha = np.log(rot / rot_tab[irot]) / Drot_over_rot
        beta = np.log(e_1 / e_1tab[ie]) / De_over_e
        
        if Zg > 0:
            result[ind] = np.exp(alpha * (beta * np.log(Ipos_tab[irot + 1, ie + 1]) + (1 - beta) * np.log(Ipos_tab[irot + 1, ie])) +
                                 (1 - alpha) * (beta * np.log(Ipos_tab[irot, ie + 1]) + (1 - beta) * np.log(Ipos_tab[irot, ie])))
        else:
            result[ind] = np.exp(alpha * (beta * np.log(Ineg_tab[irot + 1, ie + 1]) + (1 - beta) * np.log(Ineg_tab[irot + 1, ie])) +
                                 (1 - alpha) * (beta * np.log(Ineg_tab[irot, ie + 1]) + (1 - beta) * np.log(Ineg_tab[irot, ie])))

    return result

@jit(nopython=False)
def little_gp_charged(psi, Omega):
    u_min = 1e-10
    u_max = 5
    Nu = 250
    c_min = 1e-10
    c_max = 5e10
    Nc = 250

    # Create u_1d and differential over u
    u_1d = makelogtab(u_min, u_max, Nu)
    Du_over_u = DX_over_X(u_min, u_max, Nu)

    # Create arrays for u and c
    u_arr = np.repeat(u_1d, Nc)  # This simulates the column by row expansion
    c_arr = np.zeros(Nc * Nu)
    Dc_over_c = np.zeros(Nc * Nu)

    # Positively charged grains
    if psi > 0:
        for iu in range(Nu):
            u = u_1d[iu]
            if u**2 <= psi:
                c1 = c_min
            else:
                c1 = np.sqrt(1.0 - psi / u**2)
            
            if c1 < c_max:
                c_arr[iu * Nc:(iu + 1) * Nc] = makelogtab(c1, c_max, Nc)
                Dc_over_c[iu * Nc:(iu + 1) * Nc] = DX_over_X(c1, c_max, Nc)
        
        # Calculations for positively charged grains
        stuff = (2.0 / psi * c_arr * u_arr**2)**2
        e_1arr = np.sqrt(1.0 + stuff) - 1.0
        ind = np.where(stuff < 1e-10)
        if np.size(ind[0]) > 0:
            e_1arr[ind] = 0.5 * stuff[ind]

        # Rotation and plasma integral
        rot_arr = Omega * c_arr / u_arr
        Int_arr = int_plasma(rot_arr, e_1arr, 1.0)

        # Return the final result
        return 2.0 * np.sum(u_arr**2 * np.exp(-u_arr**2) * Int_arr * Dc_over_c * Du_over_u)

    # Negatively charged grains
    elif psi < 0:
        for iu in range(Nu):
            u = u_1d[iu]
            c1 = np.sqrt(1.0 - psi / u**2)
            if c1 < c_max:
                c_arr[iu * Nc:(iu + 1) * Nc] = makelogtab(c1, c_max, Nc)
                Dc_over_c[iu * Nc:(iu + 1) * Nc] = DX_over_X(c1, c_max, Nc)

        # Calculations for negatively charged grains
        stuff = (2.0 / psi * c_arr * u_arr**2)**2
        e_1arr = np.sqrt(1.0 + stuff) - 1.0
        ind = np.where(stuff < 1e-10)
        if np.size(ind[0]) > 0:
            e_1arr[ind] = 0.5 * stuff[ind]

        # Rotation and plasma integral
        rot_arr = Omega * c_arr / u_arr
        Int_arr = int_plasma(rot_arr, e_1arr, -1.0)

        # Return the final result
        return 2.0 * Du_over_u * np.sum(u_arr**2 * np.exp(-u_arr**2) * Int_arr * Dc_over_c)

    # If psi == 0, no result
    return None

def little_gp_neutral(phi, Omega):
    u_min = 1e-10
    u_max = 5.0
    Nu = 250
    X_max = 20.0
    NX = 250

    # Logarithmically spaced u values
    u_tab = makelogtab(u_min, u_max, Nu)
    Du_over_u = DX_over_X(u_min, u_max, Nu)
    
    # Array to store the integral for each u
    Int_c = np.zeros(Nu)

    # Loop over u values
    for iu in range(Nu):
        u = u_tab[iu]
        X_min = Omega / u * np.sqrt(1.0 + phi / u)

        if X_min < X_max:
            # Logarithmically spaced X values
            X_tab = makelogtab(X_min, X_max, NX)
            DX_over_X_val = DX_over_X(X_min, X_max, NX)

            # Compute the integral for this u
            Int_c[iu] = np.sum(X_tab**2 * (beselk(0, X_tab)**2 + beselk(1, X_tab)**2) * DX_over_X_val)

    # Final integral over u
    result = 2.0 * Du_over_u * np.sum(u_tab**2 * np.exp(-u_tab**2) * Int_c )

    return result

def compute_little_gp():
    # If the arrays have been computed and stored, load them
    gp_pos_filename = f"{SpDust_data_dir}/gp_pos_{gp_arrays.Npsi}psi_{gp_arrays.NOmega}Omega.txt"
    gp_neg_filename = f"{SpDust_data_dir}/gp_neg_{gp_arrays.Npsi}psi_{gp_arrays.NOmega}Omega.txt"
    gp_neutral_filename = f"{SpDust_data_dir}/gp_neutral_{gp_arrays.Nphi}phi_{gp_arrays.NOmega}Omega.txt"

    if os.path.exists(gp_pos_filename) and os.path.exists(gp_neg_filename) and os.path.exists(gp_neutral_filename):
        gp_arrays.gp_pos = np.loadtxt(gp_pos_filename)
        gp_arrays.gp_neg = np.loadtxt(gp_neg_filename)
        gp_arrays.gp_neutral = np.loadtxt(gp_neutral_filename)
        return

    # Generate log-spaced arrays for psi, phi, and Omega
    psi_tab = makelogtab(gp_arrays.psi_min, gp_arrays.psi_max, gp_arrays.Npsi)
    phi_tab = makelogtab(gp_arrays.phi_min, gp_arrays.phi_max, gp_arrays.Nphi)
    Omega_tab = makelogtab(gp_arrays.Omega_min, gp_arrays.Omega_max, gp_arrays.NOmega)

    # Initialize arrays for storing results
    gp_pos = np.zeros((gp_arrays.Npsi, gp_arrays.NOmega))
    gp_neg = np.zeros((gp_arrays.Npsi, gp_arrays.NOmega))
    gp_neutral = np.zeros((gp_arrays.Nphi, gp_arrays.NOmega))

    # --- Compute for charged grains ---
    for ipsi, psi in enumerate(psi_tab):
        for iOmega, Omega in enumerate(Omega_tab):
            gp_pos[ipsi, iOmega] = little_gp_charged(psi, Omega)
            gp_neg[ipsi, iOmega] = little_gp_charged(-psi, Omega)

    # Save gp_pos to file
    np.savetxt(gp_pos_filename, gp_pos)

    # Save gp_neg to file    
    np.savetxt(gp_neg_filename, gp_neg)

    # --- Compute for neutral grains ---
    for iphi, phi in enumerate(phi_tab):
        for iOmega, Omega in enumerate(Omega_tab):
            gp_neutral[iphi, iOmega] = little_gp_neutral(phi, Omega)

    # Save gp_neutral to file
    np.savetxt(gp_neutral_filename, gp_neutral)

    # Store the results
    gp_arrays.gp_pos = gp_pos
    gp_arrays.gp_neg = gp_neg
    gp_arrays.gp_neutral = gp_neutral

    return 

compute_little_gp()

class warnings:
    warning_phi_min, warning_phi_max, warning_psi_min, warning_psi_max, warning_Omega_min, warning_Omega_max \
    = None, None, None, None, None, None


def little_gp_charged_interpol(psi_arr, Omega_arr):

    Npsi = gp_arrays.Npsi
    NOmega = gp_arrays.NOmega
    psi_min = gp_arrays.psi_min
    psi_max = gp_arrays.psi_max
    Omega_min = gp_arrays.Omega_min
    Omega_max = gp_arrays.Omega_max

    gp_pos = gp_arrays.gp_pos
    gp_neg = gp_arrays.gp_neg

    psi_tab = makelogtab(psi_min, psi_max, Npsi)
    Omega_tab = makelogtab(Omega_min, Omega_max, NOmega)


    # Finding the indices and coefficients for psi
    psi_indices = np.floor(Npsi * np.log(abs(psi_arr) / psi_min) / np.log(psi_max / psi_min) - 0.5).astype(int)
    psi_indices = np.clip(psi_indices, 0, Npsi - 2)

    psi_coeff = 1 - Npsi * np.log(abs(psi_arr) / psi_tab[psi_indices]) / np.log(psi_max / psi_min)
    psi_coeff = np.clip(psi_coeff, 0, 1)

    # Finding the indices and coefficients for Omega
    Omega_indices = np.floor(NOmega * np.log(Omega_arr / Omega_min) / np.log(Omega_max / Omega_min) - 0.5).astype(int)
    Omega_indices = np.clip(Omega_indices, 0, NOmega - 2)

    Omega_coeff = 1 - NOmega * np.log(Omega_arr / Omega_tab[Omega_indices]) / np.log(Omega_max / Omega_min)
    Omega_coeff = np.clip(Omega_coeff, 0, 1)

    # Initializing the result array
    Npsi_arr = np.size(psi_arr)
    NOmega_arr = np.size(Omega_arr)
    gp_charged = np.zeros((Npsi_arr, NOmega_arr))

    # Positively charged grains
    ind_pos = np.where(psi_arr > 0)[0]
    if len(ind_pos) > 0:
        for idx in ind_pos:
            i, j = psi_indices[idx], Omega_indices[idx]
            pc, oc = psi_coeff[idx], Omega_coeff[idx]
            gp_charged[idx] = np.exp(
                pc * (oc * np.log(gp_pos[i, j]) + (1 - oc) * np.log(gp_pos[i, j + 1])) +
                (1 - pc) * (oc * np.log(gp_pos[i + 1, j]) + (1 - oc) * np.log(gp_pos[i + 1, j + 1]))
            )

    # Negatively charged grains
    ind_neg = np.where(psi_arr < 0)[0]
    if len(ind_neg) > 0:
        for idx in ind_neg:
            i, j = psi_indices[idx], Omega_indices[idx]
            pc, oc = psi_coeff[idx], Omega_coeff[idx]
            gp_charged[idx] = np.exp(
                pc * (oc * np.log(gp_neg[i, j]) + (1 - oc) * np.log(gp_neg[i, j + 1])) +
                (1 - pc) * (oc * np.log(gp_neg[i + 1, j]) + (1 - oc) * np.log(gp_neg[i + 1, j + 1]))
            )

    return gp_charged


def little_gp_neutral_interpol(phi, Omega_arr):
    phi_min = gp_arrays.phi_min
    phi_max = gp_arrays.phi_max
    Nphi = gp_arrays.Nphi
    Omega_min = gp_arrays.Omega_min
    Omega_max = gp_arrays.Omega_max
    NOmega = gp_arrays.NOmega
    gp_neutral = gp_arrays.gp_neutral

    phi_tab = makelogtab(phi_min, phi_max, Nphi)
    Omega_tab = makelogtab(Omega_min, Omega_max, NOmega)

    # Find the index and coefficient for phi
    phi_index = int(np.floor(Nphi * np.log(phi / phi_min) / np.log(phi_max / phi_min) - 0.5))
    phi_index = max(0, min(phi_index, Nphi - 2))  # Ensure phi_index is within bounds

    # Calculate interpolation coefficient for phi
    phi_coeff = 1 - Nphi * np.log(phi / phi_tab[phi_index]) / np.log(phi_max / phi_min)
    phi_coeff = np.clip(phi_coeff, 0, 1)  # Clamp phi_coeff between 0 and 1
    
    # Find indices and coefficients for Omega
    Omega_indices = np.floor(NOmega * np.log(Omega_arr / Omega_min) / np.log(Omega_max / Omega_min) - 0.5).astype(int)
    Omega_indices = np.clip(Omega_indices, 0, NOmega - 2)  # Ensure Omega_indices are within bounds

    # Calculate interpolation coefficients for Omega
    Omega_coeff = 1 - NOmega * np.log(Omega_arr / Omega_tab[Omega_indices]) / np.log(Omega_max / Omega_min)
    Omega_coeff = np.clip(Omega_coeff, 0, 1)  # Clamp Omega_coeff between 0 and 1

    # Interpolate gp_neutral values for phi
    gp_neu_phi = phi_coeff * gp_neutral[phi_index, :] + (1 - phi_coeff) * gp_neutral[phi_index + 1, :]

    # Return interpolated values for gp_neutral
    return Omega_coeff * gp_neu_phi[Omega_indices] + (1 - Omega_coeff) * gp_neu_phi[Omega_indices + 1]


def Gp_sphere_per_mu2_averaged(env, a, fZ, omega):
    """
    Returns G_p^(AHD09)(a, omega)/mu_ip^2 averaged over grain charges.

    Parameters:
    - env: Environment containing T, xh, and xC.
    - a: Grain size.
    - fZ: Grain charge distribution array.
    - omega: Frequency values.

    Returns:
    - Gp_over_mu2: Averaged G_p divided by mu_ip^2, as an array of the same dimension as omega.
    """
    T = env['T']
    acx_val = acx(a)
    xh = env['xh']
    xC = env['xC']

    Nomega = np.size(omega)

    Zg_arr = fZ[0, :]
    fZ_arr = fZ[1, :]

    little_gp_H = np.zeros(Nomega)
    little_gp_C = np.zeros(Nomega)

    Omega_H = np.sqrt(mp / (2 * k * T)) * acx_val * omega
    Omega_C = np.sqrt(12) * Omega_H

    # Neutral grain contribution
    phi = np.sqrt(2 / (acx_val * k * T)) * q
    little_gp_H = fZ[1, 0] * little_gp_neutral_interpol(phi, Omega_H)
    little_gp_C = fZ[1, 0] * little_gp_neutral_interpol(phi, Omega_C)

    # Charged grain contribution
    ind_charged = np.where(Zg_arr != 0)[0]
    if len(ind_charged) > 0:
        psi_arr = Zg_arr[ind_charged] * q**2 / (acx_val * k * T)
        little_gp_H += np.dot(little_gp_charged_interpol(psi_arr, Omega_H), fZ_arr[ind_charged])
        little_gp_C += np.dot(little_gp_charged_interpol(psi_arr, Omega_C), fZ_arr[ind_charged])

    Gp_over_mu2 = (q / (acx_val**2 * k * T))**2 * (xh * little_gp_H + xC * np.sqrt(12) * little_gp_C)

    return Gp_over_mu2


def FGp_averaged(env, a, fZ, omega, mu_ip, mu_op, tumbling=True):
    """
    Returns a structure {Fp, Gp} with each element as an array of dimensions [Nomega, Nmu].
    If tumbling is True, disklike tumbling grain is used.

    Parameters:
    - env: Environment containing T, xh, and xC.
    - a: Grain size.
    - fZ: Grain charge distribution array.
    - omega: Frequency values.
    - mu_ip, mu_op: Arrays of internal and external dipole moments.
    - tumbling: Boolean, whether the grain is tumbling.

    Returns:
    - A dictionary with Fp and Gp arrays.
    """
    if tumbling:
        # Disklike tumbling grain
        Gp_op = 2 / 3 * Gp_sphere_per_mu2_averaged(env, a, fZ, 2 * omega)
        Fp_op = 2 * Gp_op
        omegaG_plus = (3 + np.sqrt(3 / 5)) / 2 * omega
        omegaG_minus = (3 - np.sqrt(3 / 5)) / 2 * omega
        Gp_ip = (Gp_sphere_per_mu2_averaged(env, a, fZ, omegaG_plus) + 
                 Gp_sphere_per_mu2_averaged(env, a, fZ, omegaG_minus)) / 3

        omegaF_plus = (8 + np.sqrt(13 / 3)) / 5 * omega
        omegaF_minus = (8 - np.sqrt(13 / 3)) / 5 * omega
        Fp_ip = 0.5 * (Gp_sphere_per_mu2_averaged(env, a, fZ, omegaF_plus) +
                       Gp_sphere_per_mu2_averaged(env, a, fZ, omegaF_minus))

        Fp = np.dot(Fp_ip, mu_ip**2) + np.dot(Fp_op, mu_op**2)
        Gp = np.dot(Gp_ip, mu_ip**2) + np.dot(Gp_op, mu_op**2)

    else:
        # Standard spherical grain with K = J
        Gp = Gp_sphere_per_mu2_averaged(env, a, fZ, omega)
        Gp = np.dot(Gp, mu_ip**2)
        Fp = Gp

    return {'Fp': Fp, 'Gp': Gp}

