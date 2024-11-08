from .. import SpDust_data_dir
from ..util import cgsconst, DX_over_X, maketab, makelogtab 
from .grain_properties import acx, Inertia, grainparams, rms_dipole, size_dist
from .charge_dist import charge_dist
from .infrared import FGIR_averaged
from .collisions import Tev_effective, FGn_averaged, FGi_averaged
from .plasmadrag import FGp_averaged
from .H2_photoemission import GH2, FGpe_averaged

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d



pi = np.pi
c = cgsconst.c
h = cgsconst.h
k = cgsconst.k
eV = cgsconst.eV
me = cgsconst.me
q = cgsconst.q
mp = cgsconst.mp

# Function to calculate the characteristic damping time through collisions with neutral H atoms
def tau_H(env, a):
    """
    Returns the characteristic damping time through collisions with neutral H atoms.
    This corresponds to Eq. (24) in AHD09.

    Parameters:
    - env: Environment parameters (temperature T, hydrogen density nh).
    - a: Grain size.

    Returns:
    - Damping time tau_H (in seconds).
    """
    T = env['T']  # temperature in K
    nh = env['nh']  # hydrogen density in cm^-3
    acx_val = acx(a)  # characteristic length scale
    Inertia_val = Inertia(a)  # grain's moment of inertia

    # Calculate tau_H using the given equation
    return 1.0 / (nh * mp * np.sqrt(2 * k * T / (pi * mp)) * 4 * pi * acx_val**4 / (3 * Inertia_val))

# Function to calculate the inverse of the characteristic damping time through electric dipole radiation
def tau_ed_inv(env, a, mu_ip, mu_op, tumbling=True, corrected=False):
    """
    Returns the inverse of the characteristic damping time through electric dipole radiation.
    This corresponds to Eq. (29) in AHD09 and Eq. (47) in SAH10 for tumbling disklike grains.

    Parameters:
    - env: Environment parameters (temperature T).
    - a: Grain size.
    - mu_ip: Electric dipole moment in-plane.
    - mu_op: Electric dipole moment out-of-plane.
    - tumbling: Boolean, if True, use the tumbling disk model.

    Returns:
    - Inverse of the damping time (in seconds^-1).
    """
    T = env['T']  # temperature in K
    Inertia_val = Inertia(a)  # grain's moment of inertia

    if corrected:
        if tumbling and a < grainparams.a2:
            # Tumbling disklike grains, SAH10 Eq. (47)
            return (3 * k * T * (82 / 45 * mu_ip**2 + 32 / 9 * mu_op**2)) / (Inertia_val**2 * c**3)
        else:
            # Spherical grains or disklike grains with K = J, AHD09 Eq. (29)
            return (2 * k * T * mu_ip**2) / (Inertia_val**2 * c**3)
    else: # Use the old  IDL version - without the correction
        if tumbling:
            # Tumbling disklike grains, SAH10 Eq. (47)
            return (3 * k * T * (82 / 45 * mu_ip**2 + 32 / 9 * mu_op**2)) / (Inertia_val**2 * c**3)
        else:
            # Spherical grains or disklike grains with K = J, AHD09 Eq. (29)
            return (2 * k * T * mu_ip**2) / (Inertia_val**2 * c**3)

    
def f_rot(env, a, fZ, mu_ip, mu_op, tumbling=True):
    """
    Returns the rotational distribution function f_a normalized such that:
    ∫ f_a(omega) * 4 * pi * omega^2 d(omega) = 1.
    The function is calculated at frequencies centered on the approximate peak frequency
    of the emitted power. The output is a dictionary {omega, f_a} containing:
    - omega: array of frequencies.
    - f_a: rotational distribution function (2D array with dimensions [Nmu, Nomega]).

    Parameters:
    - env: Environmental parameters (T, nh).
    - a: Grain size.
    - ia: Index or identifier for intermediate value saving (optional).
    - fZ: Distribution of grain charges.
    - mu_ip: Electric dipole moment in-plane.
    - mu_op: Electric dipole moment out-of-plane.
    - tumbling: Boolean, whether tumbling disklike grains are considered.

    Returns:
    - Dictionary containing omega and f_a arrays.
    """
    Nomega = 1000  # Number of omega values for which the function is calculated
    Nmu = np.size(mu_ip)  # Number of dipole moments

    Tval = env['T']
    Inertia_val = Inertia(a)  # Grain's moment of inertia

    # Characteristic timescales
    tau_H_val = tau_H(env, a)
    tau_ed_inv_val = tau_ed_inv(env, a, mu_ip, mu_op, tumbling=tumbling)
    # Evaporation temperature
    Tev_val = Tev_effective(env, a)

    # F's and G's (except for plasma drag)
    FGn = FGn_averaged(env, a, Tev_val, fZ, tumbling=tumbling)
    Fn = FGn['Fn']
    Gn = FGn['Gn']

    mu_tot = np.sqrt(mu_ip**2 + mu_op**2)
    FGi = FGi_averaged(env, a, Tev_val, mu_tot, fZ)
    Fi = FGi['Fi']
    Gi = FGi['Gi']

    FGIR = FGIR_averaged(env, a, fZ)
    FIR = FGIR['FIR']
    GIR = FGIR['GIR']

    FGpe = FGpe_averaged(env, a, fZ)
    Fpe = FGpe['Fpe']
    Gpe = FGpe['Gpe']

    GH2_val = GH2(env, a)

    # Array of omegas around the approximate peak
    omega_peak_th = np.sqrt(6 * k * Tval / Inertia_val)  # Peak of spectrum if thermal rotation

    # Peak frequency for the lowest and highest values of mu_ip, mu_op
    FGp = FGp_averaged(env, a, fZ, omega_peak_th, [np.min(mu_ip), np.max(mu_ip)], [np.min(mu_op), np.max(mu_op)], tumbling=tumbling)
    Fp_th = FGp['Fp']
    Gp_th = FGp['Gp']
    

    F_low = Fn + np.min(Fi) + FIR + Fpe + np.min(Fp_th)
    G_low = Gn + np.min(Gi) + GIR + Gpe + GH2_val + np.min(Gp_th)
    xi_low = 8 * G_low / F_low**2 * tau_H_val * np.min(tau_ed_inv_val)

    F_high = Fn + np.max(Fi) + FIR + Fpe + np.max(Fp_th)
    G_high = Gn + np.max(Gi) + GIR + Gpe + GH2_val + np.max(Gp_th)
    xi_high = 8 * G_high / F_high**2 * tau_H_val * np.max(tau_ed_inv_val)

    omega_peak_low = omega_peak_th * np.sqrt(2 * G_low / F_low / (1 + np.sqrt(1 + xi_low)))
    omega_peak_high = omega_peak_th * np.sqrt(2 * G_high / F_high / (1 + np.sqrt(1 + xi_high)))


    # Array omega
    omega_min = 5e-3 * np.min((omega_peak_low, omega_peak_high))
    omega_max = 6 * np.max((omega_peak_low, omega_peak_high))
    omega = makelogtab(omega_min, omega_max, Nomega) 
    Dln_omega = DX_over_X(omega_min, omega_max, Nomega) 

    # Fp(omega), Gp(omega)
    FGp = FGp_averaged(env, a, fZ, omega, mu_ip, mu_op, tumbling=tumbling)
    Fp = FGp['Fp'] # shape (Nomega, Nmu)
    Gp = FGp['Gp']

    # Rotational distribution function, AHD09 Eq.(33)
    f_a = np.zeros((Nomega, Nmu))

    F = Fn + FIR + Fpe + np.matmul(np.ones((Nomega, 1)), Fi.reshape(1, -1)) + Fp
    G = Gn + GIR + Gpe + GH2_val + np.matmul(np.ones((Nomega, 1)), Gi.reshape(1, -1)) + Gp
    tau_ed_inv_matrix = np.matmul(np.ones((Nomega, 1)), tau_ed_inv_val.reshape(1, -1))
    omega_tab = np.matmul(omega.reshape(-1, 1), np.ones((1, Nmu)))

    X = Inertia_val * omega_tab**2 / (k * Tval)
    integrand = F / G * X + tau_H_val / (3 * G) * tau_ed_inv_matrix * X**2
    exponent = np.cumsum(integrand, axis=0) * Dln_omega
    norm = 4 * pi * np.sum(omega_tab**3 * np.exp(-exponent), axis=0) * Dln_omega
    norm = np.matmul(np.ones((Nomega, 1)), norm.reshape(1, -1))

    f_a = 1 / norm * np.exp(-exponent)

    return {'omega': omega, 'f_a': f_a.T}  # Transpose for consistency with dimensions [Nmu, Nomega]


def mu2_fa(env, a, fZ, mu_rms, ip, Ndipole, tumbling=True):
    """
    Returns a [3, Nomega] array:
    [omega, <mu_ip^2 fa(omega)>/<mu_ip^2>, <mu_op^2 fa(omega)>/<mu_op^2>].
    
    Parameters:
    - env: Environment parameters (such as temperature).
    - a: Grain size.
    - ia: Index for saving intermediate results.
    - fZ: Distribution of grain charges.
    - mu_rms: Root mean square dipole moment.
    - ip: Fraction of in-plane dipole moment.
    - Ndipole: Number of dipole values used in averaging.
    - tumbling: Boolean to indicate whether the grains are tumbling disklike grains.

    Returns:
    - A numpy array [3, Nomega] containing omega, <mu_ip^2 fa(omega)>/<mu_ip^2>, and <mu_op^2 fa(omega)>/<mu_op^2>.
    """

    op = 1.0 - ip  # Out-of-plane dipole moment fraction

    if Ndipole == 1:  # If no averaging over dipoles
        f_rot_data = f_rot(env, a, fZ, mu_rms * np.sqrt(ip), mu_rms * np.sqrt(op), tumbling=tumbling)
        omega = f_rot_data['omega']
        f_a = f_rot_data['f_a']
        Nomega = len(omega)
        mu_ip2_fa = f_a
        mu_op2_fa = f_a

    else:
        # Set up for averaging
        xmin = 5e-3
        xmed = 0.5
        xmax = 5.0
        aux_Nd = int(Ndipole/2)
        x_tab = np.concatenate((makelogtab(xmin, xmed, aux_Nd), 
                                maketab(xmed, xmax, aux_Nd)))
        Dx_tab = np.concatenate((DX_over_X(xmin, xmed, aux_Nd) * makelogtab(xmin, xmed, aux_Nd), 
                                 (xmax - xmed)/aux_Nd + np.zeros(aux_Nd)))

        if a < grainparams.a2:  # Disk-like grains, need 2D Gaussian
            mu_ip = np.sqrt(ip) * mu_rms * np.outer(x_tab, np.ones(Ndipole))
            mu_op = np.sqrt(op) * mu_rms * np.outer(np.ones(Ndipole), x_tab)
            Dmu_ip = np.outer(Dx_tab, np.ones(Ndipole))
            Dmu_op = Dmu_ip.T

            # Probability calculation for dipoles
            if ip == 0.0:
                Proba = np.exp(-0.5 * mu_op**2 / mu_rms**2) * Dmu_op
            elif op == 0.0:
                Proba = mu_ip / mu_rms * np.exp(-mu_ip**2 / mu_rms**2) * Dmu_ip
            else:
                Proba = (mu_ip / mu_rms * np.exp(-mu_ip**2 / (ip * mu_rms**2)) * 
                         np.exp(-0.5 * mu_op**2 / (op * mu_rms**2)) * Dmu_ip * Dmu_op)

            Proba = Proba / np.sum(Proba)
            # Flatten 2D arrays to 1D
            mu_ip = mu_ip.flatten()
            mu_op = mu_op.flatten()
            Proba = Proba.flatten()

            f_rot_data = f_rot(env, a, fZ, mu_ip, mu_op, tumbling=tumbling)
            omega = f_rot_data['omega']
            f_a = f_rot_data['f_a']
            Nomega = np.size(omega)

            Proba = np.outer(Proba, np.ones(Nomega))
            mu_ip = np.outer(mu_ip, np.ones(Nomega))
            mu_op = np.outer(mu_op, np.ones(Nomega))

            # Calculate <mu_ip^2 fa(omega)> and <mu_op^2 fa(omega)>
            mu_ip2_fa = np.sum((mu_ip**2 / mu_rms**2) * Proba * f_a, axis=0)
            mu_op2_fa = np.sum((mu_op**2 / mu_rms**2) * Proba * f_a, axis=0)

            if ip != 0.0 and op != 0.0:
                mu_ip2_fa /= ip
                mu_op2_fa /= op

        else:  # Spherical grains, average over grain orientation first
            Proba = x_tab**2 * np.exp(-1.5 * x_tab**2) * Dx_tab
            Proba = Proba / np.sum(Proba)

            f_rot_data = f_rot(env, a, fZ, np.sqrt(2/3) * mu_rms * x_tab, mu_rms / np.sqrt(3) * x_tab, tumbling=tumbling)
            omega = f_rot_data['omega']
            f_a = f_rot_data['f_a']
            Nomega = np.size(omega)
            Proba = np.outer(Proba, np.ones(Nomega))
            x_tab = np.outer(x_tab, np.ones(Nomega))

            mu2_fa_aux = np.sum(x_tab**2 * Proba * f_a, axis=0)
            mu_ip2_fa = mu2_fa_aux 
            mu_op2_fa = mu2_fa_aux 

    # Prepare the result as a [3, Nomega] array
    result = np.zeros((3, Nomega))
    result[0, :] = omega
    result[1, :] = mu_ip2_fa
    result[2, :] = mu_op2_fa

    # Save the intermediate results (optional)
    # print('Saving results...')
    # np.savez(f'{ia}_mu_fa.npz', omega=omega, mu_ip2_fa=mu_ip2_fa, mu_op2_fa=mu_op2_fa, mu_rms=mu_rms, ip=ip)

    return result


def fa_ip_fa_op(omega, mu_ip2_fa, mu_op2_fa):
    """
    Calculates fa_ip and fa_op functions for tumbling grains.

    Parameters:
    - omega: Array of angular frequencies (logarithmically spaced).
    - mu_ip2_fa: Array <mu_ip^2 fa>/<mu_ip^2>.
    - mu_op2_fa: Array <mu_op^2 fa>/<mu_op^2>.

    Returns:
    - A 2D array of shape [3, count] containing:
      - omega values where omega/3 is possible,
      - fa_ip values,
      - fa_op values.
    """

    Dln_omega = np.sqrt(omega[1] / omega[0]) - np.sqrt(omega[0] / omega[1])

    # Find indices where omega/3 is possible
    ind = np.where(omega / 3 > np.min(omega))[0]

    # Interpolate fa_op at omega/2 using cubic spline interpolation
    interp_op = interp1d(np.log(omega), mu_op2_fa, kind='cubic', fill_value="extrapolate")
    fa_op = interp_op(np.log(omega[ind] / 2)) / 8.0

    # Cumulative integrals for mu_ip2_fa
    int0 = np.cumsum(mu_ip2_fa[::-1])[::-1] * Dln_omega
    int1 = np.cumsum((omega * mu_ip2_fa)[::-1])[::-1] / omega * Dln_omega
    int2 = np.cumsum((omega**2 * mu_ip2_fa)[::-1])[::-1] / omega**2 * Dln_omega

    # Interpolate these integrals at omega/3
    interp_int0 = interp1d(np.log(omega), int0, kind='cubic', fill_value="extrapolate")
    interp_int1 = interp1d(np.log(omega), int1, kind='cubic', fill_value="extrapolate")
    interp_int2 = interp1d(np.log(omega), int2, kind='cubic', fill_value="extrapolate")


    int0_low = interp_int0(np.log(omega[ind] / 3))
    int1_low = interp_int1(np.log(omega[ind] / 3))
    int2_low = interp_int2(np.log(omega[ind] / 3))

    # Calculate fa_ip
    fa_ip = (int0_low - 2 * int1_low + int2_low 
             - 3 * int0[ind] + 6 * int1[ind] - 7 * int2[ind]) / 4.0

    # Prepare the result as a [3, count] array
    result = np.zeros((3, len(ind)))
    result[0, :] = omega[ind]
    result[1, :] = fa_ip
    result[2, :] = fa_op

    return result


def dP_dnu_dOmega(env, a, beta, ip, Ndipole, tumbling=True):
    """
    Returns the power radiated by a grain of radius `a`, per frequency interval, per steradian.
    The result is an array Power_per_grain[2, Nnu] such that:
        Power_per_grain[0,*] = nu
        Power_per_grain[1,*] = dP/dnu/dOmega
    """
    
    op = 1.0 - ip
    
    # Charge distribution and rms dipole moment
    fZ = charge_dist(env, a)
    Z2 = np.sum(fZ[0, :]**2 * fZ[1, :])
    mu_rms = rms_dipole(a, Z2, beta)
    
    # Calculate rotational distribution for in-plane and out-of-plane dipole moments
    mu2_fa_aux  = mu2_fa(env, a, fZ, mu_rms, ip, Ndipole, tumbling=tumbling)
    omega = mu2_fa_aux[0, :]
    mu_ip2_fa = mu2_fa_aux[1, :]
    mu_op2_fa = mu2_fa_aux[2, :]
    
    # Tumbling case for disklike grains
    if tumbling:
        fa_ip_fa_op_aux = fa_ip_fa_op(omega, mu_ip2_fa, mu_op2_fa)
        nu = fa_ip_fa_op_aux[0, :] / (2.0 * pi)
        fa_ip = fa_ip_fa_op_aux[1, :]
        fa_op = fa_ip_fa_op_aux[2, :]
        dPdnudOmega = 2.0 / (3.0 * c**3) * (2.0 * np.pi * nu)**6 * 2.0 * np.pi * mu_rms**2 * (ip * fa_ip + 2.0/3.0 * op * fa_op)
    else:
        # Spherical / disklike with K = J case
        nu = omega / (2.0 * pi)
        dPdnudOmega = 2.0 / (3.0 * c**3) * omega**6 * 2.0 * np.pi * mu_rms**2 * ip * mu_ip2_fa
    
    # Keep only non-zero values
    non_zero_indices = np.where(dPdnudOmega > 0)[0]
    if len(non_zero_indices) > 0:
        non_zero_power = np.zeros((2, len(non_zero_indices)))
        non_zero_power[0, :] = nu[non_zero_indices]
        non_zero_power[1, :] = dPdnudOmega[non_zero_indices]
        return non_zero_power
    else:
        print(f"Power vanishes for a = {a}")
        return np.array([0.0, 0.0])

def emissivity(env, beta, ip, Ndipole, nu_tab, tumbling=True):
    """
    Parameters:
    - env: Environment parameters.


    Returns j_nu/nH in cgs units (ergs/s/sr/H atom) for the given environment.
    """
    
    # Grain parameters
    a_min = 3.5e-8
    a_max = 3.5e-7  # Large grains do not contribute near the peak
    Na = 30
    a_tab = makelogtab(a_min, a_max, Na)
    Da_over_a = DX_over_X(a_min, a_max, Na)
    
    # Initialize emissivity array
    emiss = np.zeros(len(nu_tab))
    line = env['line'] - 1

    # Loop over each grain size
    for ia in range(Na):
        a = a_tab[ia]
        
        # Use tumbling case only for disklike grains
        if a < grainparams.a2:
            power_per_grain = dP_dnu_dOmega(env, a, beta, ip, Ndipole, tumbling=tumbling)
        else:
            power_per_grain = dP_dnu_dOmega(env, a, beta, 2/3, Ndipole, tumbling=False)

        
        nu_tab_a = power_per_grain[0, :]
        emiss_a = power_per_grain[1, :]
        
        # Find the frequencies where nu_tab is within the range of nu_tab_a
        ind = np.where((nu_tab > np.min(nu_tab_a)) & (nu_tab < np.max(nu_tab_a)))[0]
        if len(ind) > 0:
            # Interpolate emissivity and accumulate
            emiss[ind] += np.exp(interp1d(np.log(nu_tab_a), np.log(emiss_a), kind='cubic')(np.log(nu_tab[ind]))) * size_dist(a, line) * a * Da_over_a
    
    return emiss



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

    me = cgsconst.me
    k = cgsconst.k
    h = cgsconst.h
    eV = cgsconst.eV
    c = cgsconst.c
    q = cgsconst.q

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
