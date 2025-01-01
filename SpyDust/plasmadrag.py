import numpy as np
from scipy.special import kv as beselk

from . import SpDust_data_dir
from .util import cgsconst, makelogtab, maketab, DX_over_X
from .Grain import acx, Inertia_largest, grainparams

import os
from .mpiutil import *


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
    """
    for ig in range(Ngamma):
        Gamma = Gamma_tab[ig]
        time = Gamma * (u + u ** 3 / 3)
        
        # Integrals
        intcos = 4 * np.sum(np.real(np.exp(I * time) * fcos * Du)) ** 2
        intsin = 16 * np.sum(np.imag(np.exp(I * time) * fsin * Du)) ** 2
        
        # Store results in smalle_tab
        smalle_tab[ig] = intcos + intsin    
    """

    # Define function for parallel computation
    def compute_small_e_parallel(ig):
        Gamma = Gamma_tab[ig]
        time = Gamma * (u + u ** 3 / 3)
        
        # Integrals
        intcos = 4 * np.sum(np.real(np.exp(I * time) * fcos * Du)) ** 2
        intsin = 16 * np.sum(np.imag(np.exp(I * time) * fsin * Du)) ** 2
        
        return intcos + intsin
    
    # Parallel computation
    smalle_tab = np.array(parallel_map(compute_small_e_parallel, list(range(Ngamma))))
    
    # Output the maximum gamma for debugging purposes
    gamma_max = np.max(Gamma_tab)
    
    return Gamma_tab, smalle_tab, gamma_max

# Call `compute_small_e` to get the values for small e_1, Zg < 0
Gamma_tab, smalle_tab, Gamma_max = compute_small_e()
smalletabs.Gamma_tab = Gamma_tab
smalletabs.smalle_tab = smalle_tab
smalletabs.Gamma_max = Gamma_max
if rank0:
    print("I(Zg<0, parabolic) stored")

#@njit
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

def compute_int_plasma():
    file_path = os.path.join(SpDust_data_dir, 'int_plasma.npz')
    if os.path.exists(file_path):
        data = np.load(file_path)
        Ipos_tab = data['Ipos_tab']
        Ineg_tab = data['Ineg_tab']
        rot_tab = data['rot_tab']
        e_1tab = data['e_1tab']
        rot_min = data['rot_min']
        rot_max = data['rot_max']
        e_1min = data['e_1min']
        e_1max = data['e_1max']
        return Ipos_tab, Ineg_tab, rot_tab, e_1tab, rot_min, rot_max, e_1min, e_1max
    
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
    Dz1 = -I * np.concatenate((ymed / Ny + np.zeros(Ny), DX_over_X(ymed, ymax, Ny) * makelogtab(ymed, ymax, Ny)))
    log = 0.5 * np.log(1.0 + 4.0 / y**2) + I * np.arctan(2.0 / y)
    z = 1.0 - I * y

    '''
    for ie in range(Ne_1):
        e_1 = e_1tab[ie]
        Aval = e_1 / (e_1 + 2.0)
        time = 1.0 / np.sqrt(e_1 * (e_1 + 2.0)) * (log + 2.0 * (e_1 + 1.0) * z / (z**2 - 1.0))
        fcos = (z**2 - Aval) / (z**2 + Aval)**2
        fsin = z / (z**2 + Aval)**2
        for ir in range(Nrot):
            rot = rot_tab[ir]
            intcos = 4.0 * Aval * np.sum(np.real(np.exp(I * rot * time) * fcos * Dz1))**2
            intsin = 16.0 * Aval**2 * np.sum(np.imag(np.exp(I * rot * time) * fsin * Dz1))**2
            Ipos_tab[ir, ie] = intcos + intsin
    '''

    # Define function for parallel computation
    #@jit(nopython=True)
    def loop(ie):
        e_1 = e_1tab[ie]
        Aval = e_1 / (e_1 + 2.0)
        time = 1.0 / np.sqrt(e_1 * (e_1 + 2.0)) * (log + 2.0 * (e_1 + 1.0) * z / (z**2 - 1.0))
        fcos = (z**2 - Aval) / (z**2 + Aval)**2
        fsin = z / (z**2 + Aval)**2
        result = np.zeros(Nrot)
        for ir in range(Nrot):
            rot = rot_tab[ir]
            intcos = 4.0 * Aval * np.sum(np.real(np.exp(I * rot * time) * fcos * Dz1))**2
            intsin = 16.0 * Aval**2 * np.sum(np.imag(np.exp(I * rot * time) * fsin * Dz1))**2
            result[ir] = intcos + intsin
        return result
    
    # Parallel computation
    Ipos_tab = np.array(parallel_map(loop, list(range(Ne_1)))).T

    # Handling small e_1 and high rotation for Zg < 0
    inde = np.where(e_1tab < e_1small)[0][-1]
    indrot = np.where(rot_tab < rot_small)[0][-1]

    # Zg < 0 non-small e_1
    Dz2 = np.exp(I * pi / 4.0) * np.concatenate((ymed / Ny + np.zeros(Ny), DX_over_X(ymed, ymax, Ny) * makelogtab(ymed, ymax, Ny)))
    log = 0.5 * np.log(1.0 + 4.0 / y**2 + 2.0 * np.sqrt(2.0) / y) - I * np.arctan(np.sqrt(2.0) / (np.sqrt(2.0) + y))
    z = 1.0 + np.exp(I * pi / 4.0) * y

    # Define function for parallel computation
    #(nopython=True)
    def loop(ie):
        e_1 = e_1tab[ie]
        Aval = e_1 / (e_1 + 2.0)
        time = 1.0 / np.sqrt(e_1 * (e_1 + 2.0)) * (log - 2.0 * (e_1 + 1.0) * z / (z**2 - 1.0))
        fcos = (1.0 - Aval * z**2) / (1.0 + Aval * z**2)**2
        fsin = z / (1.0 + Aval * z**2)**2
        result = np.zeros(indrot)
        for ir in range(indrot):
            rot = rot_tab[ir]
            intcos = 4.0 * Aval * np.sum(np.real(np.exp(I * rot * time) * fcos * Dz2))**2
            intsin = 16.0 * Aval**2 * np.sum(np.imag(np.exp(I * rot * time) * fsin * Dz2))**2
            result[ir] = intcos + intsin
        return result
    
    # Parallel computation
    Ineg_tab[:indrot,:] = np.array(parallel_map(loop, list(range(Ne_1)))).T

    # Define function for parallel computation
    #@jit(nopython=True)
    def loop(ie):
        e_1 = e_1tab[ie]
        result = np.zeros(Nrot)
        for ir in range(indrot, Nrot):
            rot = rot_tab[ir]
            Gamma = rot * e_1
            if Gamma < Gamma_max:
                result[ir] = np.interp(Gamma, Gamma_tab, smalle_tab)
        return result
    
    # Parallel computation
    if Nrot > indrot:
        result = np.array(parallel_map(loop, list(range(inde)))).T
        Ineg_tab[indrot:, :inde] = result[indrot:, :]

    # Some useful things for further interpolation
    rot_min = np.min(rot_tab)
    rot_max = np.max(rot_tab)
    e_1min = np.min(e_1tab)
    e_1max = np.max(e_1tab)

    # Replace zeros with small values for precision
    Ipos_tab = replace_zeros(Ipos_tab)
    Ineg_tab = replace_zeros(Ineg_tab)

    # Save the arrays to file
    if rank0:
        np.savez(file_path, Ipos_tab=Ipos_tab, Ineg_tab=Ineg_tab, rot_tab=rot_tab, e_1tab=e_1tab,
             rot_min=rot_min, rot_max=rot_max, e_1min=e_1min, e_1max=e_1max)

    return Ipos_tab, Ineg_tab, rot_tab, e_1tab, rot_min, rot_max, e_1min, e_1max



Ipos_tab, Ineg_tab, rot_tab, e_1tab, rot_min, rot_max, e_1min, e_1max = compute_int_plasma()

plasma_tabs.Ipos_tab = Ipos_tab
plasma_tabs.Ineg_tab = Ineg_tab
plasma_tabs.rot_tab = rot_tab
plasma_tabs.e_1tab = e_1tab
plasma_tabs.rot_min = rot_min
plasma_tabs.rot_max = rot_max
plasma_tabs.e_1min = e_1min
plasma_tabs.e_1max = e_1max
if rank0:
    print('I(rot, e, Zg <> 0) stored')

barrier()

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
        irot = np.int64(np.floor(np.log(rot / np.min(rot_tab)) / Drot_over_rot))
        
        De_over_e = np.log(e_1tab[1] / e_1tab[0])
        ie = np.int64(np.floor(np.log(e_1 / np.min(e_1tab)) / De_over_e))
        
        alpha = np.log(rot / rot_tab[irot]) / Drot_over_rot
        beta = np.log(e_1 / e_1tab[ie]) / De_over_e
        
        if Zg > 0:
            result[ind] = np.exp(alpha * (beta * np.log(Ipos_tab[irot + 1, ie + 1]) + (1 - beta) * np.log(Ipos_tab[irot + 1, ie])) +
                                 (1 - alpha) * (beta * np.log(Ipos_tab[irot, ie + 1]) + (1 - beta) * np.log(Ipos_tab[irot, ie])))
        else:
            result[ind] = np.exp(alpha * (beta * np.log(Ineg_tab[irot + 1, ie + 1]) + (1 - beta) * np.log(Ineg_tab[irot + 1, ie])) +
                                 (1 - alpha) * (beta * np.log(Ineg_tab[irot, ie + 1]) + (1 - beta) * np.log(Ineg_tab[irot, ie])))

    return result

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

    '''
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
    '''

    # Define function for parallel computation
    def compute_integral(iu):
        u = u_tab[iu]
        X_min = Omega / u * np.sqrt(1.0 + phi / u)
        if X_min < X_max:
            X_tab = makelogtab(X_min, X_max, NX)
            DX_over_X_val = DX_over_X(X_min, X_max, NX)
            return np.sum(X_tab**2 * (beselk(0, X_tab)**2 + beselk(1, X_tab)**2) * DX_over_X_val)
        else:
            return 0.0
        
    # Parallel computation
    Int_c = np.array(parallel_map(compute_integral, list(range(Nu))))

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
    '''
    for ipsi, psi in enumerate(psi_tab):
        for iOmega, Omega in enumerate(Omega_tab):
            gp_pos[ipsi, iOmega] = little_gp_charged(psi, Omega)
            gp_neg[ipsi, iOmega] = little_gp_charged(-psi, Omega)
    '''
    # Define function for parallel computation
    def compute_pos(ipsi):
        psi = psi_tab[ipsi]
        result = np.zeros(gp_arrays.NOmega)
        for iOmega, Omega in enumerate(Omega_tab):
            result[iOmega] = little_gp_charged(psi, Omega)
        return result
    
    def compute_neg(ipsi):
        psi = psi_tab[ipsi]
        result = np.zeros(gp_arrays.NOmega)
        for iOmega, Omega in enumerate(Omega_tab):
            result[iOmega] = little_gp_charged(-psi, Omega)
        return result
    
    # Parallel computation
    gp_pos = np.array(parallel_map(compute_pos, list(range(gp_arrays.Npsi))))
    gp_neg = np.array(parallel_map(compute_neg, list(range(gp_arrays.Npsi))))
        

    if rank0:
        # Save gp_pos to file
        np.savetxt(gp_pos_filename, gp_pos)

        # Save gp_neg to file    
        np.savetxt(gp_neg_filename, gp_neg)

    # --- Compute for neutral grains ---
    """
    for iphi, phi in enumerate(phi_tab):
        for iOmega, Omega in enumerate(Omega_tab):
            gp_neutral[iphi, iOmega] = little_gp_neutral(phi, Omega)    
    """

    def compute_neutral(iphi):
        phi = phi_tab[iphi]
        result = np.zeros(gp_arrays.NOmega)
        for iOmega, Omega in enumerate(Omega_tab):
            result[iOmega] = little_gp_neutral(phi, Omega)
        return result
    
    gp_neutral = np.array(parallel_map(compute_neutral, list(range(gp_arrays.Nphi))))
    
    if rank0:
        # Save gp_neutral to file
        np.savetxt(gp_neutral_filename, gp_neutral)

    # Store the results
    gp_arrays.gp_pos = gp_pos
    gp_arrays.gp_neg = gp_neg
    gp_arrays.gp_neutral = gp_neutral

    return 

compute_little_gp()
barrier()

Npsi = gp_arrays.Npsi
NOmega = gp_arrays.NOmega
psi_min = gp_arrays.psi_min
psi_max = gp_arrays.psi_max
Omega_min = gp_arrays.Omega_min
Omega_max = gp_arrays.Omega_max

gp_pos = gp_arrays.gp_pos
gp_neg = gp_arrays.gp_neg


def loop_charged(ipsi, psi_arr, NOmega_arr, psi_indices, psi_coeff, Omega_indices, Omega_coeff, gp_pos, gp_neg):
    result = np.zeros(NOmega_arr)
    psi_indices_charged = psi_indices[ipsi]
    psi_coeff_charged = psi_coeff[ipsi]
    Omega_indices_charged = Omega_indices[ipsi, :]
    Omega_coeff_charged = Omega_coeff[ipsi, :]

    if psi_arr[ipsi] > 0:
        result = np.exp(psi_coeff_charged * (Omega_coeff_charged * np.log(gp_pos[psi_indices_charged, Omega_indices_charged]) + 
                        (1 - Omega_coeff_charged) * np.log(gp_pos[psi_indices_charged, Omega_indices_charged + 1])) +
                        (1 - psi_coeff_charged) * (Omega_coeff_charged * np.log(gp_pos[psi_indices_charged + 1, Omega_indices_charged]) + 
                        (1 - Omega_coeff_charged) * np.log(gp_pos[psi_indices_charged + 1, Omega_indices_charged + 1]))
                        )
    elif psi_arr[ipsi] < 0:
        result = np.exp(
            psi_coeff_charged * (Omega_coeff_charged * np.log(gp_neg[psi_indices_charged, Omega_indices_charged]) + 
                             (1 - Omega_coeff_charged) * np.log(gp_neg[psi_indices_charged, Omega_indices_charged + 1])) +
            (1 - psi_coeff_charged) * (Omega_coeff_charged * np.log(gp_neg[psi_indices_charged + 1, Omega_indices_charged]) + 
                               (1 - Omega_coeff_charged) * np.log(gp_neg[psi_indices_charged + 1, Omega_indices_charged + 1]))
            )
    return result


def little_gp_charged_interpol(psi_arr, Omega_arr):
    """
    Parameters:
    - psi_arr: Array of grain potentials
    - Omega_arr: Array of grain rotation frequencies

    Returns:
    - gp_charged: Interpolated values of gp_charged for the input psi_arr and Omega_arr
    should be a 2D array of size (Npsi, NOmega)
    """

    psi_tab = makelogtab(psi_min, psi_max, Npsi)
    Omega_tab = makelogtab(Omega_min, Omega_max, NOmega)


    # Finding the indices and coefficients for psi
    aux = np.floor(Npsi * np.log(np.absolute(psi_arr) / psi_min) / np.log(psi_max / psi_min) - 0.5)
    psi_indices = np.array([np.int64(x) for x in aux])
    psi_indices = np.clip(psi_indices, 0, Npsi - 2)

    psi_coeff = 1 - Npsi * np.log(np.absolute(psi_arr) / psi_tab[psi_indices]) / np.log(psi_max / psi_min)
    psi_coeff = np.clip(psi_coeff, 0, 1)

    # Finding the indices and coefficients for Omega
    aux = np.floor(NOmega * np.log(Omega_arr / Omega_min) / np.log(Omega_max / Omega_min) - 0.5)
    Omega_indices = np.array([np.int64(x) for x in aux])
    Omega_indices = np.clip(Omega_indices, 0, NOmega - 2)

    Omega_coeff = 1 - NOmega * np.log(Omega_arr / Omega_tab[Omega_indices]) / np.log(Omega_max / Omega_min)
    Omega_coeff = np.clip(Omega_coeff, 0, 1)

    # Initializing the result array
    Npsi_arr = np.size(psi_arr)
    NOmega_arr = np.size(Omega_arr)
    gp_charged = np.zeros((Npsi_arr, NOmega_arr))

    # Expand psi_indices and Omega_indices across the appropriate dimensions
    # Multiply psi_indices across the NOmega_arr dimension
    psi_indices = psi_indices.reshape(-1, 1) @ np.ones((1, NOmega_arr), dtype=np.int64)

    # Multiply Omega_indices across the Npsi_arr dimension
    Omega_indices = np.ones((Npsi_arr, 1), dtype=np.int64) @ Omega_indices.reshape(1, -1)

    # Step 4: Expand psi_coeff and Omega_coeff similarly across the dimensions
    # Expand psi_coeff across the NOmega_arr dimension
    psi_coeff = psi_coeff.reshape(-1, 1) @ np.ones((1, NOmega_arr))

    # Expand Omega_coeff across the Npsi_arr dimension
    Omega_coeff = np.ones((Npsi_arr, 1)) @ Omega_coeff.reshape(1, -1)
    
    # gp_charged = np.array(parallel_map(loop_charged, list(range(Npsi_arr))))
    
    # Not to do parallel computation

    for ipsi in range(Npsi_arr):
        gp_charged[ipsi, :] = loop_charged(ipsi, psi_arr, NOmega_arr, psi_indices, psi_coeff, Omega_indices, Omega_coeff, gp_pos, gp_neg)

    return gp_charged # shape: [Npsi_arr, NOmega_arr]


phi_min = gp_arrays.phi_min
phi_max = gp_arrays.phi_max
Nphi = gp_arrays.Nphi
Omega_min = gp_arrays.Omega_min
Omega_max = gp_arrays.Omega_max
NOmega = gp_arrays.NOmega
gp_neutral = gp_arrays.gp_neutral

#@jit
def little_gp_neutral_interpol(phi, Omega_arr):
    """
    Parameters:
    - phi: Grain potential
    - Omega_arr: Array of grain rotation frequencies

    Returns:
    - gp_neu: Interpolated values of gp_neutral for the input phi and Omega_arr
    should be a 1D array of the same size as Omega_arr
    """

    phi_tab = makelogtab(phi_min, phi_max, Nphi)
    Omega_tab = makelogtab(Omega_min, Omega_max, NOmega)

    # Find the index and coefficient for phi
    phi_index = int(np.floor(Nphi * np.log(phi / phi_min) / np.log(phi_max / phi_min) - 0.5))
    #phi_index = np.max((0, np.min((phi_index, Nphi - 2))))  # Ensure phi_index is within bounds
    phi_index = np.maximum(0, np.minimum(phi_index, Nphi - 2))  # Ensure phi_index is within bounds

    # Calculate interpolation coefficient for phi
    phi_coeff = 1 - Nphi * np.log(phi / phi_tab[phi_index]) / np.log(phi_max / phi_min)
    #phi_coeff = np.clip(phi_coeff, 0, 1)  # Clamp phi_coeff between 0 and 1
    phi_coeff = np.maximum(0, phi_coeff)
    phi_coeff = np.minimum(1, phi_coeff)
    
    # Find indices and coefficients for Omega
    # Omega_indices = np.int64(np.floor(NOmega * np.log(Omega_arr / Omega_min) / np.log(Omega_max / Omega_min) - 0.5))
    Omega_indices = np.array([int(x) for x in np.floor(NOmega * np.log(Omega_arr / Omega_min) / np.log(Omega_max / Omega_min) - 0.5)])
    Omega_indices = np.clip(Omega_indices, 0, NOmega - 2)  # Ensure Omega_indices are within bounds
    #Omega_indices = np.maximum(0, np.minimum(Omega_indices, NOmega - 2))  # Ensure Omega_indices are within bounds

    # Calculate interpolation coefficients for Omega
    Omega_coeff = 1 - NOmega * np.log(Omega_arr / Omega_tab[Omega_indices]) / np.log(Omega_max / Omega_min)
    Omega_coeff = np.clip(Omega_coeff, 0, 1)  # Clamp Omega_coeff between 0 and 1
    #Omega_coeff = np.maximum(0, Omega_coeff)
    #Omega_coeff = np.minimum(1, Omega_coeff)

    # Interpolate gp_neutral values for phi
    gp_neu_phi = phi_coeff * gp_neutral[phi_index, :] + (1 - phi_coeff) * gp_neutral[phi_index + 1, :]

    # Return interpolated values for gp_neutral
    return Omega_coeff * gp_neu_phi[Omega_indices] + (1 - Omega_coeff) * gp_neu_phi[Omega_indices + 1]



#@jit
def Gp_per_mu2_averaged(temp, xh, xC, a, beta, fZ, omega_vec):
    """
    Returns G_p^(AHD09)(a, omega)/mu_ip^2 averaged over grain charges.

    Parameters:
    - env: Environment containing T, xh, and xC.
    - a: Grain size.
    - fZ: Grain charge distribution array. 
    - omega: Frequency values.

    Returns:
    - Gp_over_mu2: Averaged G_p divided by mu_ip^2, as an array of the same dimension as omega. shape: [Nomega]
    """
   
    acx_val = acx(a, beta)

    Nomega = np.size(omega_vec)

    Zg_arr = fZ[0, :]
    fZ_arr = fZ[1, :] 

    little_gp_H = np.zeros(Nomega)
    little_gp_C = np.zeros(Nomega)

    Omega_H = np.sqrt(mp / (2 * k * temp)) * acx_val * omega_vec
    Omega_C = np.sqrt(12) * Omega_H

    # Neutral grain contribution
    phi = np.sqrt(2 / (acx_val * k * temp)) * q
    little_gp_H = fZ[1, 0] * little_gp_neutral_interpol(phi, Omega_H) # shape: [Nomega]
    little_gp_C = fZ[1, 0] * little_gp_neutral_interpol(phi, Omega_C)

    # Charged grain contribution
    ind_charged = np.where(Zg_arr != 0)[0]
    if len(ind_charged) > 0:
        psi_arr = Zg_arr[ind_charged] * q**2 / (acx_val * k * temp)
        #little_gp_H += np.matmul(little_gp_charged_interpol(psi_arr, Omega_H).T, fZ_arr[ind_charged])
        little_gp_H += little_gp_charged_interpol(psi_arr, Omega_H).T @ fZ_arr[ind_charged]
        #little_gp_C += np.matmul(little_gp_charged_interpol(psi_arr, Omega_C).T, fZ_arr[ind_charged])
        little_gp_C += little_gp_charged_interpol(psi_arr, Omega_C).T @ fZ_arr[ind_charged]

    Gp_over_mu2 = (q / (acx_val**2 * k * temp))**2 * (xh * little_gp_H + xC * np.sqrt(12) * little_gp_C)

    return Gp_over_mu2


def FGp_averaged(env, a, beta, fZ, omega_vec, mu_ip, mu_op, tumbling=True):
    """
    Returns a structure {Fp, Gp} with each element as an array of dimensions [Nomega, Nmu].
    (Averaged over grain charges.)
    If tumbling is True, disklike tumbling grain is used.

    Parameters:
    - env: Environment containing T, xh, and xC.
    - a: Grain size.
    - fZ: Grain charge distribution array.
    - omega_vec: Frequency values.
    - mu_ip, mu_op: Arrays of internal and external dipole moments.
    - tumbling: Boolean, whether the grain is tumbling.

    Returns:
    - A dictionary with Fp and Gp arrays. Shape: [Nomega, Nmu]
    """
    mu_ip_2 = np.array(mu_ip)**2
    mu_op_2 = np.array(mu_op)**2
    temp, xh, xC = env['T'], env['xh'], env['xC']
    Gp_op = 2/3 * Gp_per_mu2_averaged(temp, xh, xC, a, beta, fZ, 2 * omega_vec) 
    Nomegas = np.size(omega_vec)

    if beta == 0: # Spherical grain
        tumbling = False

    if tumbling:
        # Disklike tumbling grain 
        ## Corr: omega could be a vector
        def Gp_Fp_vals(omega_ind):
            omega = omega_vec[omega_ind]
            cos_theta_list = maketab(-1, 1, 21)
            Omega_arr = np.absolute(omega  * (1 - beta * cos_theta_list) / (1+beta) )
            weight_Gp = (1 - cos_theta_list)**2 / 4
            weight_Fp = - weight_Gp * cos_theta_list
            aux_result = Gp_per_mu2_averaged(temp, xh, xC, a, beta, fZ, Omega_arr)
            result_Gp = 2*np.average(aux_result*weight_Gp) *  mu_ip_2 + Gp_op[omega_ind] * mu_op_2 
            result_Fp = result_Gp/(1+beta) + (beta/(1+beta)) * 2*np.average(aux_result*weight_Fp) *  mu_ip_2 
            return result_Gp, result_Fp
        
        Gp, Fp = [], []

        for ind in np.arange(Nomegas):
            Gp_vals, Fp_vals = Gp_Fp_vals(ind)
            Gp.append(Gp_vals)
            Fp.append(Fp_vals)

        Gp = np.array(Gp).reshape(Nomegas, -1)
        Fp = np.array(Fp).reshape(Nomegas, -1)
       
        return {'Fp': Fp, 'Gp': Gp}

    # Standard spherical grain with K = J
    Gp = Gp_per_mu2_averaged(temp, xh, xC, a, beta, fZ, omega_vec)
    Gp = np.matmul(Gp.reshape(-1,1), mu_ip_2.reshape(1,-1))
    Fp = Gp

    return {'Fp': Fp, 'Gp': Gp}


