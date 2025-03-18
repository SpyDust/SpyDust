from . import SpDust_data_dir
from .util import cgsconst
from .mpiutil import rank0

import numpy as np
import pandas as pd

pi = np.pi
c = cgsconst.c
h = cgsconst.h
k = cgsconst.k
eV = cgsconst.eV
me = cgsconst.me
q = cgsconst.q
mp = cgsconst.mp


class gff_data:
    gamma2_tab = None
    u_tab = None 
    gff_tab = None

def read_gaunt_factor():

    # Load the data using pandas
    gff_data_file = SpDust_data_dir + 'gff.dat'
    data = pd.read_csv(gff_data_file, delim_whitespace=True, comment=';', header=None, names=['gamma2', 'u', 'gff'])

    # Define dimensions
    Ngamma2 = 41
    Nu = 81

    # Initialize arrays
    gff_data.gamma2_tab = np.zeros(Ngamma2)
    gff_data.u_tab = np.zeros(Nu)
    gff_data.gff_tab = np.zeros((Ngamma2, Nu))

    # Assign values to gamma2_tab and u_tab
    gff_data.gamma2_tab = data['gamma2'].values[Nu * np.arange(Ngamma2)].astype(float)
    gff_data.u_tab = data['u'].values[0:Nu].astype(float)

    # Fill the gff_tab array
    for i in range(Ngamma2):
        gff_data.gff_tab[i, :] = data['gff'].values[i * Nu + np.arange(Nu)].astype(float)
    
    if rank0:
        print('Gaunt factor stored')

# Call the function to test it
read_gaunt_factor()

def gaunt_factor(gamma2, u):
    # Number of elements in gamma2_tab
    Ngamma2 = len(gff_data.gamma2_tab)

    # Determine the index based on gamma2 value
    if gamma2 >= np.max(gff_data.gamma2_tab):
        index = Ngamma2 - 1
    elif gamma2 <= np.min(gff_data.gamma2_tab):
        index = 0
    else:
        # Find the largest index where gamma2_tab is less than gamma2
        index = np.max(np.where(gff_data.gamma2_tab < gamma2)[0])
        # Adjust index based on the logarithmic condition
        if np.log(gff_data.gamma2_tab[index + 1] / gamma2) < np.log(gamma2 / gff_data.gamma2_tab[index]):
            index += 1

    # Extract the corresponding row from gff_tab
    gff_new = gff_data.gff_tab[index, :]

    # Interpolate the value for the given 'u' using u_tab and gff_new
    result = np.interp(u, gff_data.u_tab, gff_new)

    return result

def free_free(env, nu_tab):
    """
    Returns j_nu/nH in cgs units (ergs/s/sr/Hatom) for free-free emission, for the given environment.
    
    Parameters:
    - env: an object or dictionary containing environment properties nh, T, xh, xC
    - nu_tab: array of frequencies in Hz
    
    Returns:
    - free-free emission rate j_nu/nH in cgs units
    """
    
    # Extract values from the environment object or dictionary
    nh = env['nh']
    T = env['T']
    xh = env['xh']
    xC = env['xC']


    # Calculate the factor
    factor = (2**5 * np.pi * q**6 / (3 * me * c**3)) * np.sqrt(2 * np.pi / (3 * k * me)) / (4 * np.pi)

    # Rydberg constant in erg (13.6 eV converted to ergs)
    Ry = 13.6 * eV

    # Gamma2 and u calculations
    gamma2 = Ry / (k * T)  # Assuming Zion = 1
    u = h * nu_tab / (k * T)

    # Call the gaunt_factor function
    gff = gaunt_factor(gamma2, u)

    # Return the free-free emission rate
    result = factor * (xh + xC)**2 * nh / np.sqrt(T) * np.exp(-h * nu_tab / (k * T)) * gff
    return result
