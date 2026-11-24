import numpy as np
from scipy.special import kv as beselk

from . import SpDust_data_dir
from ._small_e import compute_small_e, small_e_tables as _small_e_tables, table_attribute as __getattr__
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
a2 = grainparams.a2

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
        with np.load(file_path) as data:
            Ipos_tab = data['Ipos_tab']
            Ineg_tab = data['Ineg_tab']
            rot_tab = data['rot_tab']
            e_1tab = data['e_1tab']
            rot_min = data['rot_min']
            rot_max = data['rot_max']
            e_1min = data['e_1min']
            e_1max = data['e_1max']
        return Ipos_tab, Ineg_tab, rot_tab, e_1tab, rot_min, rot_max, e_1min, e_1max
    
    tables = _small_e_tables()
    Gamma_tab, smalle_tab, Gamma_max = tables.Gamma_tab, tables.smalle_tab, tables.Gamma_max

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

    barrier()
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
        
    # Each rank owns its phi from the outer table-building loop.
    Int_c = np.array([compute_integral(i) for i in range(Nu)])

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
    barrier()

    return 

compute_little_gp()

Npsi = gp_arrays.Npsi
NOmega = gp_arrays.NOmega
psi_min = gp_arrays.psi_min
psi_max = gp_arrays.psi_max
Omega_min = gp_arrays.Omega_min
Omega_max = gp_arrays.Omega_max

gp_pos = gp_arrays.gp_pos
gp_neg = gp_arrays.gp_neg


_psi_grid = makelogtab(psi_min, psi_max, Npsi)
_omega_grid = makelogtab(Omega_min, Omega_max, NOmega)


def _log_grid_weights(values, grid, lower, upper):
    """Indices and left-node weights on a logarithmic cell-centred table."""
    values = np.clip(values, grid[0], grid[-1])
    log_range = np.log(upper / lower)
    position = grid.size * np.log(values / lower) / log_range - 0.5
    index = np.clip(np.floor(position).astype(np.intp), 0, grid.size - 2)
    weight = 1 - grid.size * np.log(values / grid[index]) / log_range
    return index, np.clip(weight, 0., 1.)


def loop_charged(ipsi, psi_arr, NOmega_arr, psi_indices, psi_coeff, Omega_indices, Omega_coeff, gp_pos, gp_neg):
    """Interpolate one signed charge row of the logarithmic table."""
    if psi_arr[ipsi] == 0:
        return np.zeros(NOmega_arr)
    table = gp_pos if psi_arr[ipsi] > 0 else gp_neg
    i, wi = psi_indices[ipsi], psi_coeff[ipsi]
    j, wj = Omega_indices[ipsi], Omega_coeff[ipsi]
    left = wj * np.log(table[i, j]) + (1 - wj) * np.log(table[i, j + 1])
    right = wj * np.log(table[i + 1, j]) + (1 - wj) * np.log(table[i + 1, j + 1])
    return np.exp(wi * left + (1 - wi) * right)


def little_gp_charged_interpol(psi_arr, Omega_arr):
    """Return charged-grain coefficients on the (potential, frequency) grid."""
    psi_indices, psi_coeff = _log_grid_weights(np.abs(psi_arr), _psi_grid, psi_min, psi_max)
    omega_indices, omega_coeff = _log_grid_weights(Omega_arr, _omega_grid, Omega_min, Omega_max)
    shape = (len(psi_arr), len(Omega_arr))
    # Each charge uses the same frequency grid. Broadcast these arrays as views.
    omega_indices = np.broadcast_to(omega_indices, shape)
    omega_coeff = np.broadcast_to(omega_coeff, shape)
    result = np.empty(shape)
    for i in range(shape[0]):
        result[i] = loop_charged(i, psi_arr, shape[1], psi_indices, psi_coeff,
                                 omega_indices, omega_coeff, gp_pos, gp_neg)
    return result


phi_min = gp_arrays.phi_min
phi_max = gp_arrays.phi_max
Nphi = gp_arrays.Nphi
Omega_min = gp_arrays.Omega_min
Omega_max = gp_arrays.Omega_max
NOmega = gp_arrays.NOmega
gp_neutral = gp_arrays.gp_neutral

_phi_grid = makelogtab(phi_min, phi_max, Nphi)


def little_gp_neutral_interpol(phi, Omega_arr):
    """Interpolate neutral-grain coefficients on the log-spaced table axes."""
    i, wi = _log_grid_weights(phi, _phi_grid, phi_min, phi_max)
    j, wj = _log_grid_weights(Omega_arr, _omega_grid, Omega_min, Omega_max)
    row = wi * gp_neutral[i] + (1 - wi) * gp_neutral[i + 1]
    return wj * row[j] + (1 - wj) * row[j + 1]


#@jit
def Gp_per_mu2_averaged(temp, xh, xC, a, beta, fZ, omega_vec, a2=a2):
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
   
    acx_val = acx(a, beta, a2=a2)

    Nomega = np.size(omega_vec)

    Zg_arr = fZ[0, :]
    fZ_arr = fZ[1, :] 

    little_gp_H = np.zeros(Nomega)
    little_gp_C = np.zeros(Nomega)

    Omega_H = np.sqrt(mp / (2 * k * temp)) * acx_val * omega_vec
    Omega_C = np.sqrt(12) * Omega_H

    # Neutral grain contribution
    phi = np.sqrt(2 / (acx_val * k * temp)) * q
    f0 = fZ_arr[Zg_arr == 0].sum()
    little_gp_H = f0 * little_gp_neutral_interpol(phi, Omega_H) # shape: [Nomega]
    little_gp_C = f0 * little_gp_neutral_interpol(phi, Omega_C)

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


def FGp_averaged(env, a, beta, fZ, omega_vec, mu_ip, mu_op, tumbling=True, a2=a2):
    """
    Returns a structure {Fp, Gp} with each element as an array of dimensions [Nomega, Nmu].
    (Averaged over grain charges.)
    Uses the paper's Eqs. (42–46) convention, with tau_H proportional to I3.

    Parameters:
    - env: Environment containing T, xh, and xC.
    - a: Grain size.
    - fZ: Grain charge distribution array.
    - omega_vec: L/I3, where I3 is the moment about the body symmetry axis.
    - mu_ip, mu_op: Dipole magnitudes in the equatorial plane and along the body axis.
    - tumbling: Boolean, whether the grain is tumbling.

    Returns:
    - A dictionary with Fp and Gp arrays. Shape: [Nomega, Nmu]
    """
    omega = np.atleast_1d(omega_vec)
    mu_ip_2 = np.asarray(mu_ip).reshape(-1)**2
    mu_op_2 = np.asarray(mu_op).reshape(-1)**2

    def per_mu2(frequencies):
        return Gp_per_mu2_averaged(env['T'], env['xh'], env['xC'], a, beta,
                                   fZ, frequencies, a2=a2)

    if not tumbling and beta > 0:
        # At theta=pi/2, axial dipoles rotate at L/Iref. Half of the
        # equatorial dipole power lies transverse to L after azimuth averaging.
        Gp = per_mu2(omega / (1 + beta))[:, None] * (mu_op_2 + 0.5 * mu_ip_2)
        return {'Fp': Gp / (1 + beta), 'Gp': Gp}
    if beta == 0 or not tumbling:
        Gp = per_mu2(omega)[:, None] * mu_ip_2
        return {'Fp': Gp, 'Gp': Gp}

    # Eqs. (43) and (46): omega is L/I3 and omega/(1+beta) is L/Iref.
    ref_omega = omega / (1 + beta)
    Gp_op = (2/3) * per_mu2(ref_omega)
    cos_theta = maketab(-1, 1, 21)
    weight_g = (1 - cos_theta)**2 / 4
    weight_f = -cos_theta * weight_g
    Gp = np.empty((omega.size, mu_ip_2.size))
    Fp = np.empty_like(Gp)

    # Bound the (charge, frequency, angle) interpolation work array per grain.
    block_size = 32
    for start in range(0, omega.size, block_size):
        block = slice(start, start + block_size)
        mapped = ref_omega[block, None] * np.abs(1 - beta * cos_theta)
        drag = per_mu2(mapped.ravel()).reshape(mapped.shape)
        g_ip = 2 * np.mean(drag * weight_g, axis=1)
        f_cross = 2 * np.mean(drag * weight_f, axis=1)
        Gp[block] = g_ip[:, None] * mu_ip_2 + Gp_op[block, None] * mu_op_2
        Fp[block] = (Gp[block] + beta * f_cross[:, None] * mu_ip_2) / (1 + beta)

    return {'Fp': Fp, 'Gp': Gp}

# A wildcard import explicitly requests the historic small-e table attributes.
__all__ = [name for name in globals() if not name.startswith('_')] + [
    'Gamma_tab', 'smalle_tab', 'Gamma_max', 'smalletabs',
]
