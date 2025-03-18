from .util import cgsconst, DX_over_X, maketab, makelogtab 
from .mpiutil import *
from .Grain import grainparams, rms_dipole
from .charge_dist import charge_dist
from .AngMomDist import log_f_rot
import numpy as np
from scipy.interpolate import interp1d


pi = np.pi
c = cgsconst.c
h = cgsconst.h
k = cgsconst.k
eV = cgsconst.eV
me = cgsconst.me
q = cgsconst.q
mp = cgsconst.mp

def mu2_f_cond(env, a, beta, fZ, mu_rms, ip, Ndipole, tumbling=True, omega_min=1e7, omega_max=1e12, Nomega=1000, spdust_plasma=False):
    """
    Returns a [3, Nomega] array:
    [omega, <mu_ip^2 fa(omega)>, <mu_op^2 fa(omega)>].
    
    Parameters:
    - env: Environment parameters (such as temperature).
    - a: Grain size.
    - beta: Grain shape parameter.
    - fZ: Distribution of grain charges.
    - mu_rms: Root mean square dipole moment.
    - ip: Fraction of in-plane dipole moment.
    - Ndipole: Number of dipole values used in averaging.
    - tumbling: Boolean to indicate whether the grains are tumbling disklike grains.
    Returns:
    - A numpy array [2, Nomega] containing omega, <mu_ip^2 fa(omega)>, and <mu_op^2 fa(omega)>.
    """
    op = 1.0 - ip  # Out-of-plane dipole moment fraction

    if Ndipole == 1:  # Not averaging over dipoles
        mu_ip = np.array([np.sqrt(ip) * mu_rms]) # shape (1,)
        mu_op = np.array([np.sqrt(op) * mu_rms]) # shape (1,)
        log_f_a = log_f_rot(env, a, beta, fZ, mu_ip, mu_op, 
                           tumbling=tumbling, 
                           omega_min=omega_min, omega_max=omega_max, Nomega=Nomega, use_spdust_plasma=spdust_plasma)
        f_a = np.exp(log_f_a)
        mu_ip2_fa = f_a * mu_ip[0]**2
        mu_op2_fa = f_a * mu_op[0]**2
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
                Proba = mu_ip / mu_rms * np.exp(-mu_ip**2 / (ip * mu_rms**2)) * \
                         np.exp(-0.5 * mu_op**2 / (op * mu_rms**2)) * Dmu_ip * Dmu_op
            Proba = Proba / np.sum(Proba)
            # Flatten 2D arrays to 1D
            mu_ip = mu_ip.flatten()
            mu_op = mu_op.flatten()
            Proba = Proba.flatten()
            log_f_a = log_f_rot(env, a, beta, fZ, mu_ip, mu_op, 
                               tumbling=tumbling,  
                               omega_min=omega_min, 
                               omega_max=omega_max, 
                               Nomega=Nomega,
                               use_spdust_plasma=spdust_plasma)
            f_a = np.exp(log_f_a)
            Proba = np.outer(Proba, np.ones(Nomega))
            mu_ip = np.outer(mu_ip, np.ones(Nomega)) # shape (Ndipole, Nomega)
            mu_op = np.outer(mu_op, np.ones(Nomega))
            # Calculate <mu_ip^2 fa(omega)> and <mu_op^2 fa(omega)>
            mu_ip2_fa = np.sum(mu_ip**2  * Proba * f_a, axis=0) # sum over Ndipole
            mu_op2_fa = np.sum(mu_op**2  * Proba * f_a, axis=0)
            #mu_ip2_fa = np.sum((mu_ip**2 / mu_rms**2) * Proba * f_a, axis=0) # sum over Ndipole
            #mu_op2_fa = np.sum((mu_op**2 / mu_rms**2) * Proba * f_a, axis=0)
            #if ip != 0.0 and op != 0.0:
            #    mu_ip2_fa /= ip
            #    mu_op2_fa /= op
        else:  # Spherical grains, average over grain orientation first
            Proba = x_tab**2 * np.exp(-1.5 * x_tab**2) * Dx_tab
            Proba = Proba / np.sum(Proba)
            log_f_a = log_f_rot(env, a, beta, fZ, np.sqrt(2/3) * mu_rms * x_tab, mu_rms / np.sqrt(3) * x_tab, 
                               tumbling=tumbling,  omega_min=omega_min, omega_max=omega_max, Nomega=Nomega, use_spdust_plasma=spdust_plasma)
            f_a = np.exp(log_f_a)
            Proba = np.outer(Proba, np.ones(Nomega))
            x_tab = np.outer(x_tab, np.ones(Nomega))
            mu2_fa_aux = np.sum(x_tab**2 * Proba * f_a, axis=0)
            mu_ip2_fa = mu2_fa_aux * mu_rms**2 * 2/3
            mu_op2_fa = mu2_fa_aux * mu_rms**2 * 1/3
    # Prepare the result as a [3, Nomega] array
    result = np.zeros((2, Nomega))
    
    result[0, :] = mu_ip2_fa
    result[1, :] = mu_op2_fa
    # Save the intermediate results (optional)
    # print('Saving results...')
    # np.savez(f'{ia}_mu_fa.npz', omega=omega, mu_ip2_fa=mu_ip2_fa, mu_op2_fa=mu_op2_fa, mu_rms=mu_rms, ip=ip)
    return result

def mu2_f(env, a_tab, beta_tab, f_a_beta, mole_dipole, ip, Ndipole, tumbling=True, parallel=True, contract_a=True, omega_min=1e7, omega_max=1e15, Nomega=1000, spdust_plasma=False):
    '''
    Calculate mu^2 f(omega, a, beta, mu) and marginalize over "mu" and "a" (if contract_a is True).
    '''
    num_a = np.size(a_tab)
    num_beta = np.size(beta_tab)
    
    def aux_func(a_ind):
        result = []
        a = a_tab[a_ind]
        for beta_ind in range(num_beta):
            beta = beta_tab[beta_ind]
            if f_a_beta[a_ind, beta_ind] == 0.:
                result.append(np.zeros((2, Nomega)))
                continue
            fZ = charge_dist(env, a, beta)
            Z2 = np.sum(fZ[0, :]**2 * fZ[1, :])
            mu_rms = rms_dipole(a, beta, Z2, mole_dipole)
            mu2_fa_aux  = mu2_f_cond(env, a, beta, fZ, mu_rms, ip, Ndipole, tumbling=tumbling, omega_min=omega_min, omega_max=omega_max, Nomega=Nomega, spdust_plasma=spdust_plasma)
            result.append(mu2_fa_aux)
        return np.array(result)
    
    if parallel:
        aux = np.array(parallel_map(aux_func, np.arange(num_a)))
    else:
        aux=[]
        for ai in np.arange(num_a):
            aux.append(aux_func(ai))
        aux = np.array(aux)
    result = aux * f_a_beta[:,:,np.newaxis,np.newaxis]
    if contract_a:
        # Only when f_a_beta incorporate the differential element da 
        # we can simply sum over a as the integral over a.
        # This is the case when the keyword "a_weighted" is set to True in the function size_and_shape_dist. (See main.Grain)
        return np.sum(result, axis=0) # shape (Nbeta, 2, Nomega)
    else:
        return result # shape (Na, Nbeta, 2, Nomega)
    
def SED(nu_tab, mu2_f_arr, beta_tab, angular_Omega_tab, cos_theta_list, cos_theta_weights):
    '''
    Calculate the SED for an ensemble of spinning dust grains.
    Parameters:
    - nu_tab: array of frequencies in Hz.
    - mu2_f_arr: array of shape (Nbeta, 2, Nomega), 
                where (:, 0, :) and (:, 1, :) are mu2_f_ip and mu2_f_op, respectively:
                    - mu2_f_ip := mu_perp^2 f(omega, a, beta, mu) marginalized over grain sizes (a) and dipole moments (mu) distributions
                    - mu2_f_op := mu_parallel^2 f(omega, a, beta, mu) marginalized over grain sizes (a) and dipole moments (mu) distributions
    - beta_tab: array of grain shape parameters, shape (Nbeta,)
    - angular_Omega_tab: array of rescaled angular momentum, shape (Nomega,). 
                This corresponds to the last axis of mu2_f_arr.
    - cos_theta_list: array of cos(theta) values for averaging over nutation angle, shape (N_cos_theta,)
    - cos_theta_weights: array of weights for averaging over nutation angle, shape (Nbeta, N_cos_theta)
                Note the shape: we allow for different weights for each beta value.
    '''
    mu2_f_ip = mu2_f_arr[:, 0, :]
    mu2_f_op = mu2_f_arr[:, 1, :]
    n_cos_theta = np.size(cos_theta_list)
    log_angular_Omega = np.log(angular_Omega_tab)
    omega_tab = nu_tab * 2 * pi
    N_freqs = np.size(omega_tab)
    
    def emiss_beta(beta_ind):
        beta = beta_tab[beta_ind]
        aux_ip = mu2_f_ip[beta_ind, :]
        mask_ip = aux_ip > 1e-300
        aux_ip = aux_ip[mask_ip]
        log_angular_Omega_ip = log_angular_Omega[mask_ip]
        interp_ip = interp1d( log_angular_Omega_ip , np.log(aux_ip), kind='cubic', fill_value='extrapolate', bounds_error=False)
        aux_op = mu2_f_op[beta_ind, :]
        mask_op = aux_op > 0
        aux_op = aux_op[mask_op]
        log_angular_Omega_op = log_angular_Omega[mask_op]
        interp_op = interp1d( log_angular_Omega_op , np.log(aux_op), kind='cubic', fill_value='extrapolate', bounds_error=False)
        emiss_mode_1 = 8/9 *  np.exp(interp_op(np.log(omega_tab)))  * omega_tab**4
        if beta == 0: 
            aux = 8/9 *  np.exp(interp_ip(np.log(omega_tab)))  * omega_tab**4
            return aux + emiss_mode_1
        aux = np.zeros((n_cos_theta, N_freqs))
        for theta_ind in range(n_cos_theta):
            cos_theta = cos_theta_list[theta_ind]
            log_omega_tab_mode2 = np.log(omega_tab/np.abs(1+beta*cos_theta))
            log_omega_tab_mode3 = np.log(omega_tab/np.abs(1-beta*cos_theta))
            part2= 1/3 *  np.exp(interp_ip(log_omega_tab_mode2))  * (1+cos_theta)**2 / np.abs(1+beta*cos_theta)
            part3= 1/3 *  np.exp(interp_ip(log_omega_tab_mode3))  * (1-cos_theta)**2 / np.abs(1-beta*cos_theta)
            if cos_theta == 0:
                aux[theta_ind, :] = (part2 + part3) * omega_tab**4
                continue
            log_omega_tab_mode4 = np.log(omega_tab/np.abs(beta*cos_theta))
            part4= 2/3 * np.exp(interp_ip(log_omega_tab_mode4)) * (1-cos_theta**2) / np.abs(beta*cos_theta)
            aux[theta_ind, :] = (part2 + part3 + part4) * omega_tab**4
        return np.average(aux, axis=0, weights=cos_theta_weights[beta_ind]) + emiss_mode_1 # shape (N_freqs,)

    result = parallel_map(emiss_beta, np.arange(len(beta_tab)))
    result = np.array(result)
    result = np.sum(result, axis=0) # shape (N_freqs,)
    return result # shape (N_freqs,)
    

def SED_imp(nu_tab, mu2_f_arr, beta_tab, angular_Omega_tab, cos_theta_list, cos_theta_weights):
    '''
    Calculate the SED for an ensemble of spinning dust grains.
    Parameters:
    - nu_tab: array of frequencies in Hz.
    - mu2_f_arr: array of shape (Nbeta, 2, Nomega), 
                where (:, 0, :) and (:, 1, :) are mu2_f_ip and mu2_f_op, respectively:
                    - mu2_f_ip := mu_perp^2 f(omega, a, beta, mu) marginalized over grain sizes (a) and dipole moments (mu) distributions
                    - mu2_f_op := mu_parallel^2 f(omega, a, beta, mu) marginalized over grain sizes (a) and dipole moments (mu) distributions
    - beta_tab: array of grain shape parameters, shape (Nbeta,)
    - angular_Omega_tab: array of rescaled angular momentum, shape (Nomega,). 
                This corresponds to the last axis of mu2_f_arr.
    - cos_theta_list: array of cos(theta) values for averaging over nutation angle, shape (N_cos_theta,)
    - cos_theta_weights: array of weights for averaging over nutation angle, shape (Nbeta, N_cos_theta)
                Note the shape: we allow for different weights for each beta value.
    '''
    mu2_f_ip = mu2_f_arr[:, 0, :]
    mu2_f_op = mu2_f_arr[:, 1, :]
    n_cos_theta = np.size(cos_theta_list)
    log_angular_Omega = np.log(angular_Omega_tab)
    omega_tab = nu_tab * 2 * pi
    log_omega_tab = np.log(omega_tab)
    omega_tab_4 = omega_tab**4
    N_freqs = np.size(omega_tab)
    
    def emiss_beta(beta_ind):
        beta = beta_tab[beta_ind]

        aux_ip = mu2_f_ip[beta_ind, :]
        mask_ip = aux_ip > 1e-300
        
        log_aux_ip = np.log(aux_ip[mask_ip]) 

        interp_ip = interp1d( log_angular_Omega[mask_ip] , log_aux_ip, kind='cubic', fill_value='extrapolate', bounds_error=False)
        aux_op = mu2_f_op[beta_ind, :]
        mask_op = aux_op > 1e-300

        
        log_aux_op = np.log(aux_op[mask_op])
        interp_op = interp1d( log_angular_Omega[mask_op] , log_aux_op, kind='cubic', fill_value='extrapolate', bounds_error=False)
        emiss_mode_1 = 8/9 *  np.exp(interp_op(log_omega_tab))  * omega_tab_4

        if beta == 0: 
            return  emiss_mode_1 + 8/9 *  np.exp(interp_ip(log_omega_tab))  * omega_tab_4

        aux = np.zeros((n_cos_theta, N_freqs))

        beta_cos_theta = beta * cos_theta_list[:, np.newaxis]
        one_plus_beta_cos = np.abs(1 + beta_cos_theta)
        one_minus_beta_cos = np.abs(1 - beta_cos_theta)
        
        log_omega_mode2 = log_omega_tab - np.log(one_plus_beta_cos)
        log_omega_mode3 = log_omega_tab - np.log(one_minus_beta_cos)
        
        part2 = 1/3 * np.exp(interp_ip(log_omega_mode2)) * \
                (1 + cos_theta_list[:, np.newaxis])**2 / one_plus_beta_cos
        part3 = 1/3 * np.exp(interp_ip(log_omega_mode3)) * \
                (1 - cos_theta_list[:, np.newaxis])**2 / one_minus_beta_cos
        
        zero_mask = cos_theta_list == 0
        aux[~zero_mask] = (part2[~zero_mask] + part3[~zero_mask] + \
                          2/3 * np.exp(interp_ip(log_omega_tab - \
                          np.log(np.abs(beta_cos_theta[~zero_mask])))) * \
                          (1 - cos_theta_list[~zero_mask, np.newaxis]**2) / \
                          np.abs(beta_cos_theta[~zero_mask])) * omega_tab_4
        aux[zero_mask] = (part2[zero_mask] + part3[zero_mask]) * omega_tab_4

        return np.average(aux, axis=0, weights=cos_theta_weights[beta_ind]) + emiss_mode_1 # shape (N_freqs,)

    result = parallel_map(emiss_beta, np.arange(len(beta_tab)))
    return np.sum(np.array(result), axis=0)
    