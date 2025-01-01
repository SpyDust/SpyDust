import numpy as np
from .util import cgsconst, makelogtab, DX_over_X
from .Grain import acx, Inertia_largest, grainparams
from .infrared import FGIR_averaged
from .collisions import Tev_effective, FGn_averaged, FGi_averaged
from .plasmadrag import FGp_averaged
from .H2_photoemission import GH2, FGpe_averaged

# To compare with the old plasma drag calculation in SpDust
from .SPDUST_as_is import plasmadrag as spd_plasmadrag
from scipy.interpolate import interp1d


c = cgsconst.c
mp = cgsconst.mp
k = cgsconst.k
pi = np.pi

a2 = grainparams.a2



# Function to calculate the characteristic damping time through collisions with neutral H atoms
# @njit
def tau_H(temp, nh, a, beta):
    """
    Returns the characteristic damping time through collisions with neutral H atoms.
    This corresponds to Eq. (24) in AHD09.

    Parameters:
    - env: Environment parameters (temperature T, hydrogen density nh).
    - a: Grain size.
    - temp:   temperature in K
    - nh:  hydrogen density in cm^-3

    Returns:
    - Damping time tau_H (in seconds).
    """

    acx_val = acx(a, beta)  # characteristic length scale
    Inertia_val = Inertia_largest(a, beta)  # grain's moment of inertia

    # Calculate tau_H using the given equation
    return 1.0 / (nh * mp * np.sqrt(2 * k * temp / (pi * mp)) * 4 * pi * acx_val**4 / (3 * Inertia_val))

# @njit
def dissipation_vec_ed( L_val, theta_b, beta, mu_abs, ip, I_ref):
    '''
    Calculate the angular momentum dissipation rate for the back-reaction of the electric dipole radiation.

    Parameters:
    -----------
    L_val: float
        Magnitude of the angular momentum.
    theta_b: float
        The internal angle between the grain body z-axis and the angular momentum.
    beta: float
        The grain parameter, describing the shape of the grain.
    mu_abs: float
        The absolute value of the dipole moment.
    ip: float
        The in-plane ratio of the dipole moment. ip = mu_perp/mu_abs.
    I_ref: float
        The reference moment of inertia.

    Returns:
    --------
        The angular momentum dissipation rate.
    '''

    mu_parallel_2 = mu_abs**2 * (1 - ip**2)
    mu_perp_2 = mu_abs**2 * ip**2

    aux_perp = 1 + ( 1 + 3 * beta * (2 + beta) ) * np.cos(theta_b)**2 + beta**2 * ( 3 + 2 * beta ) * np.cos(theta_b)**4
    aux_parallel = 2 * np.sin(theta_b)**2 

    result = - (1/3) * ( mu_perp_2 * aux_perp + mu_parallel_2 * aux_parallel ) * (L_val / (I_ref * c)) ** 3
    return result

# @njit
def dissipation_rate_ed_theta_avrg( L_val, beta, mu_abs, ip, I_ref):
    '''
    Calculate the ed angular momentum dissipation rate, averaging over nutation angle.
    '''

    mu_perp_2 = mu_abs**2 * ip**2
    mu_parallel_2 = mu_abs**2 * (1 - ip**2)

    aux_perp = (2/5) * beta ** 3 + (8/5) * beta ** 2 + 2 * beta
    aux_parallel = 4 / 3 

    result = - (1/3) * ( mu_perp_2 * aux_perp + mu_parallel_2 * aux_parallel ) * (L_val / (I_ref * c)) ** 3
    return result


# Function to calculate the inverse of the characteristic damping time through electric dipole radiation
# @njit
def tau_ed_inv(temp, a, beta, mu_ip, mu_op, tumbling=True):
    """
    Returns the inverse of the characteristic damping time through electric dipole radiation.
    This corresponds to Eq. (29) in AHD09 and Eq. (47) in SAH10 for tumbling disklike grains.

    Parameters:
    - temp: Environment temperature T in K.
    - a: Grain size.
    - mu_ip: Electric dipole moment in-plane.
    - mu_op: Electric dipole moment out-of-plane.
    - tumbling: Boolean, if True, use the tumbling disk model.

    Returns:
    - Inverse of the damping time (in seconds^-1).
    """  
    
    Inertia_val = Inertia_largest(a, beta)  # grain's moment of inertia
    result = k * temp / (Inertia_val**2 * c**3 * (1+beta)**3)

    if beta==0: # spherical grain
        tumbling = False

    if tumbling:
        # Tumbling disklike grains, SAH10 Eq. (47)
        return result * ((2 * beta**3 / 5 + 8 * beta**2 / 5 + 2*beta + 4/3) * mu_ip**2 + 4 / 3 * mu_op**2) 
    elif beta<=0:
            # oblate grain, theta=0
        return result * (2 * beta**3 + 6*beta**2 + 6*beta + 2 ) * mu_ip**2
    else:
        # prolate grain, theta=pi/2
        return result * (mu_ip**2 + 2 * mu_op**2) 



# Manually implemented cumulative sum function
# @njit
def cumsum_axis_0(arr):
    result = np.zeros_like(arr)
    result[0] = arr[0]
    for i in range(1, arr.shape[0]):
        result[i] = result[i-1] + arr[i]
    return result


def aux_int(Tval, F, G, omega, Nomega, Nmu, tau_H_val, tau_ed_inv_val, Inertia_val, Dln_omega):

     # Rotational distribution function, AHD09 Eq.(33)
    f_a = np.zeros((Nomega, Nmu))

    tau_ed_inv_matrix = np.ones((Nomega, 1)) @ tau_ed_inv_val.reshape(1, -1) # shape (Nomega, Nmu)
    omega_tab = omega.reshape(-1, 1) @ np.ones((1, Nmu)) # shape (Nomega, Nmu)

    X = Inertia_val * omega_tab**2 / (k * Tval)     # shape (Nomega, Nmu)
    integrand = F / G * X + tau_H_val / (3 * G) * tau_ed_inv_matrix * X**2 # shape (Nomega, Nmu)

    #exponent = np.cumsum(integrand, axis=0) * Dln_omega 
    
    #exponent = cumsum_axis_0(integrand)* Dln_omega 
    exponent = np.cumsum(integrand, axis=0) * Dln_omega  # shape (Nomega, Nmu)

    #expo_min = np.min(exponent, axis=0)
    #expo_max = np.max(exponent, axis=0)
    #expo_mid = (expo_max + expo_min) / 2    # shape (Nmu,)
    #print("expo_min, expo_max", expo_min, expo_max)
    #exponent = exponent - expo_mid[np.newaxis, :]
    #exponent = np.clip(exponent, -500, 500)
    #exponent = exponent - (expo_max + expo_min) / 2
    #exponent_aux = 3*np.log(omega_tab) - exponent + np.log(4 * pi * Dln_omega) 
    #norm = np.sum(np.exp(exponent_aux), axis=0) 
    
    # norm = 4 * pi * np.sum(omega_tab**3 * np.exp(-exponent), axis=0) * Dln_omega
    # norm = np.ones((Nomega, 1)) @ norm.reshape(1, -1)
    # f_a = 4 * pi * omega_tab**2 / norm * np.exp(-exponent)

    norm = 4 * pi * np.sum( np.exp(3 * np.log(omega_tab) - exponent), axis=0) * Dln_omega
    log_norm = np.log(norm)

    log_norm = np.ones((Nomega, 1)) @ log_norm.reshape(1, -1)
    
    log_f_a = np.log(4 * pi * omega_tab**2) - exponent - log_norm
    return log_f_a.T   # shape (Nmu, Nomega)


# @njit
def rescale_f_rot(omega, f_a, beta, log=True):
    """
    This function rescales omega in spdust convention to our convention.

    Parameters:
    - omega: array of frequencies.
    - f_a: rotational distribution function (2D array with dimensions [Nmu, Nomega]).
    - log: Boolean, whether the input f_a is in log scale.
    """
    
    omega_new = omega / (1 + beta)
    if log:
        result = f_a + np.log(1+beta)
        return np.log(omega_new), result
    f_a_new =  f_a * (1 + beta)
    return omega_new, f_a_new

    
def log_f_rot(env, a, beta, fZ, mu_ip, mu_op, tumbling=True, omega_min=1e8, omega_max=1e16, Nomega=1000, use_spdust_plasma=False):
    """
    Returns the rotational distribution function f_a:
    f(Omega | a, beta, mu)
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
    Nmu = np.size(mu_ip)  # Number of dipole moments

    Inertia_val = Inertia_largest(a, beta)  # Grain's moment of inertia

    Tval, nH = env['T'], env['nh']

    if beta==0: # spherical grain
        tumbling = False

    # Characteristic timescales
    tau_H_val = tau_H(Tval, nH, a, beta)
    tau_ed_inv_val = tau_ed_inv(Tval, a, beta, mu_ip, mu_op, tumbling=tumbling)

    # Evaporation temperature
    Tev_val = Tev_effective(env, a)

    # F's and G's (except for plasma drag)
    FGn = FGn_averaged(env, a, beta, Tev_val, fZ, tumbling=tumbling)
    Fn = FGn['Fn']
    Gn = FGn['Gn']

    mu_tot = np.sqrt(mu_ip**2 + mu_op**2)
    FGi = FGi_averaged(env, a, beta, Tev_val, mu_tot, fZ)
    Fi = FGi['Fi']
    Gi = FGi['Gi']

    FGIR = FGIR_averaged(env, a, beta, fZ)
    FIR = FGIR['FIR']
    GIR = FGIR['GIR']

    FGpe = FGpe_averaged(env, a, beta, fZ)
    Fpe = FGpe['Fpe']
    Gpe = FGpe['Gpe']

    GH2_val = GH2(env, a, beta)

    # Array of omegas around the approximate peak
    omega_peak_th = np.array([np.sqrt(6 * k * Tval / Inertia_val)])  # Peak of spectrum if thermal rotation

    # Peak frequency for the lowest and highest values of mu_ip, mu_op
    FGp = FGp_averaged(env, a, beta, fZ, omega_peak_th, [min(mu_ip), max(mu_ip)], [min(mu_op), max(mu_op)], tumbling=tumbling)
    Fp_th = FGp['Fp']
    Gp_th = FGp['Gp']

    F_low = Fn + min(Fi) + FIR + Fpe + min(Fp_th)
    G_low = Gn + min(Gi) + GIR + Gpe + GH2_val + min(Gp_th)
    xi_low = 8 * G_low / F_low**2 * tau_H_val * min(tau_ed_inv_val)

    F_high = Fn + max(Fi) + FIR + Fpe + max(Fp_th)
    G_high = Gn + max(Gi) + GIR + Gpe + GH2_val + max(Gp_th)
    xi_high = 8 * G_high / F_high**2 * tau_H_val * max(tau_ed_inv_val)

    omega_peak_low = omega_peak_th * np.sqrt(2 * G_low / F_low / (1 + np.sqrt(1 + xi_low)))
    omega_peak_high = omega_peak_th * np.sqrt(2 * G_high / F_high / (1 + np.sqrt(1 + xi_high)))

    # Array omega
    #aux_omega_min = 5e-3 * np.min((omega_peak_low, omega_peak_high))
    aux_omega_min = omega_min * (1+beta)
    aux_omega_max = 3 * np.max((omega_peak_low, omega_peak_high))
    aux_omega_max = np.min((aux_omega_max, omega_max * (1+beta)))
    aux_omega = makelogtab(aux_omega_min, aux_omega_max, Nomega) 
    Dln_omega = DX_over_X(aux_omega_min, aux_omega_max, Nomega) 

    # Fp(omega), Gp(omega)
    #FGp = FGp_averaged(env, a, beta, fZ, aux_omega, mu_ip, mu_op, tumbling=tumbling)
    #Fp = FGp['Fp'] # shape (Nomega, Nmu)
    #Gp = FGp['Gp']

    if use_spdust_plasma:
        FGp = spd_plasmadrag.FGp_averaged(env, a, fZ, aux_omega, mu_ip, mu_op, tumbling=tumbling)
        Fp = FGp['Fp'] # shape (Nomega, Nmu)
        Gp = FGp['Gp']
    else:
        FGp = FGp_averaged(env, a, beta, fZ, aux_omega, mu_ip, mu_op, tumbling=tumbling)
        Fp = FGp['Fp'] # shape (Nomega, Nmu)
        Gp = FGp['Gp']

    F = Fn + FIR + Fpe + np.matmul(np.ones((Nomega, 1)), Fi.reshape(1, -1)) + Fp
    G = Gn + GIR + Gpe + GH2_val + np.matmul(np.ones((Nomega, 1)), Gi.reshape(1, -1)) + Gp
    aux_result = aux_int(Tval, F, G, aux_omega, Nomega, Nmu, tau_H_val, tau_ed_inv_val, Inertia_val, Dln_omega) # shape: (Nmu, Nomega)
    log_Omega, aux_result = rescale_f_rot(aux_omega, aux_result, beta) 
    # Interpolate per mu
    result = np.zeros((Nmu, Nomega))
    omegaVec = makelogtab(omega_min, omega_max, Nomega)
    omegaVec_log = np.log(omegaVec)
    for ind in range(Nmu):
        aux_result_mu = aux_result[ind, :]
        interp_func = interp1d(log_Omega, aux_result_mu, kind='cubic', fill_value='extrapolate')
        result[ind, :] = interp_func(omegaVec_log)
    return result

    
def f_rot_old(env, a, beta, fZ, mu_ip, mu_op, tumbling=True, omega_min=1e7, omega_max=1e11, Nomega=1000):
    """
    Returns the rotational distribution function f_a:
    f(Omega | a, beta, mu)
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

    if beta==0: # spherical grain
        tumbling = False

    Nmu = np.size(mu_ip)  # Number of dipole moments

    Inertia_val = Inertia_largest(a, beta)  # Grain's moment of inertia

    Tval, nH = env['T'], env['nh']

    # Characteristic timescales
    tau_H_val = tau_H(Tval, nH, a, beta)
    tau_ed_inv_val = tau_ed_inv(Tval, a, beta, mu_ip, mu_op, tumbling=tumbling)

    # Evaporation temperature
    Tev_val = Tev_effective(env, a)

    # F's and G's (except for plasma drag)
    FGn = FGn_averaged(env, a, beta, Tev_val, fZ, tumbling=tumbling)
    Fn = FGn['Fn']
    Gn = FGn['Gn']

    mu_tot = np.sqrt(mu_ip**2 + mu_op**2)
    FGi = FGi_averaged(env, a, beta, Tev_val, mu_tot, fZ)
    Fi = FGi['Fi']
    Gi = FGi['Gi']

    FGIR = FGIR_averaged(env, a, beta, fZ)
    FIR = FGIR['FIR']
    GIR = FGIR['GIR']

    FGpe = FGpe_averaged(env, a, beta, fZ)
    Fpe = FGpe['Fpe']
    Gpe = FGpe['Gpe']

    GH2_val = GH2(env, a, beta)

    omega = makelogtab(omega_min, omega_max, Nomega) * (1+beta)
    Dln_omega = DX_over_X(omega_min*(1+beta) , omega_max*(1+beta), Nomega) 

    # Fp(omega), Gp(omega)
    FGp = FGp_averaged(env, a, beta, fZ, omega, mu_ip, mu_op, tumbling=tumbling)
    Fp = FGp['Fp'] # shape (Nomega, Nmu)
    Gp = FGp['Gp']

    F = Fn + FIR + Fpe + np.matmul(np.ones((Nomega, 1)), Fi.reshape(1, -1)) + Fp
    G = Gn + GIR + Gpe + GH2_val + np.matmul(np.ones((Nomega, 1)), Gi.reshape(1, -1)) + Gp
    result1, result2 = aux_int(Tval, F, G, omega, Nomega, Nmu, tau_H_val, tau_ed_inv_val, Inertia_val, Dln_omega)
    return rescale_f_rot(result1, result2, beta)
   
