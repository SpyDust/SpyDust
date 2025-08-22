import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import functools

import SpyDust.Grain as Grain
from SpyDust.SpyDust import SpyDust_given_grain_size_shape
from SpyDust.util import makelogtab


DC_params = {'nh' : 1e4, 'T': 10., 'Chi': 1e-4, 'xh': 0., 'xC': 1e-6, 'y' : 0.999,  'gamma': 0, 'dipole': 9.3, 'line':7}

MC_params = {'nh' : 3e2, 'T': 20., 'Chi': 1e-2, 'xh': 0., 'xC': 1e-4, 'y' : 0.99,  'gamma': 0, 'dipole': 9.3, 'line':7}    

CNM_params = {'nh' : 30, 'T': 100., 'Chi': 1, 'xh': 1.2e-3, 'xC': 3e-4, 'y' : 0, 'gamma': 0, 'dipole': 9.3, 'line':7}

WNM_params = {'nh' : 0.4, 'T': 6000., 'Chi': 1., 'xh': 0.1, 'xC': 3e-4, 'y' : 0, 'gamma': 0, 'dipole': 9.3, 'line':7}

WIM_params = {'nh' : 0.1, 'T': 8000., 'Chi': 1., 'xh': 0.99, 'xC': 1e-3, 'y' : 0, 'gamma': 0, 'dipole': 9.3, 'line':7}

RN_params = {'nh' : 1000., 'T': 100., 'Chi': 1000., 'xh': 0.001, 'xC': 2e-4, 'y' : 0.5, 'gamma': 0, 'dipole': 9.3, 'line':7}

PDR_params = {'nh' : 1e5, 'T': 300., 'Chi': 3000., 'xh': 0.0001, 'xC': 2e-4, 'y' : 0.5, 'gamma': 0, 'dipole': 9.3, 'line':7}

a_min=3.5e-8
a_max=3.5e-7
a_tab = makelogtab(a_min, a_max, 30)

# Designa a range of the parameters:
#   (1) C \in [-3.5, 1.0]
#   (2) ln_a0 \in [ln(10^-8), ln(10^-6)] (a0 from 1 angstrom to 100 angstrom)
#   (3) sigma_inv \in [0, 10] (i.e., sigma from 0.01 to infinity)
#   (4) d \in [0.5 * Grain.d, 2 * Grain.d]

C_min = -3.5
C_max = 1.0
C_tab = np.linspace(C_min, C_max, 20)
log_a0_min = np.log(1e-8)
log_a0_max = np.log(1e-6)
log_a0_tab = np.linspace(log_a0_min, log_a0_max, 20)
sigma_inv_min = 0
sigma_inv_max = 10
sigma_inv_tab = np.linspace(sigma_inv_min, sigma_inv_max, 20)
d_min = 0.5 * Grain.d
d_max = 2 * Grain.d
d_tab = np.linspace(d_min, d_max, 20)

def eval_beta_arbitrary(a, a2, d):
    if a <= a2:
        beta_val = Grain.cylindrical_params(a, d)[1]
    else:
        beta_val=0.
    return beta_val

def cache_last_call(func):
    last_args = {"args": None, "kwargs": None, "result": None}

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Compare with last args
        if (last_args["args"] is not None 
            and len(args) == len(last_args["args"]) 
            and all(np.array_equal(a, b) if isinstance(a, np.ndarray) else a == b
                    for a, b in zip(args, last_args["args"])) 
            and kwargs == last_args["kwargs"]):
            return last_args["result"]

        # Otherwise, compute new
        result = func(*args, **kwargs)
        last_args["args"] = args
        last_args["kwargs"] = kwargs
        last_args["result"] = result
        return result

    return wrapper

@cache_last_call
def generate_same_grains_SED(env, a, d):
    # if "a" is a number, return a single SED result; if "a" is a table, return a table of SED results
    a2 = Grain.a2
    if isinstance(a, (int, float)):
        result = SpyDust_given_grain_size_shape(env, a, eval_beta_arbitrary(a, a2, d), 
                                                min_freq=0.1, max_freq=100.0, n_freq=200,
                                                N_angular_Omega=500)
        result_SED = result[1]
    else:
        result_SED = []
        for a_val in a:
            result = SpyDust_given_grain_size_shape(env, a_val, eval_beta_arbitrary(a_val, a2, d), 
                                                min_freq=0.1, max_freq=100.0, n_freq=200,
                                                N_angular_Omega=500)
            result_SED.append(result[1])
        result_SED = np.array(result_SED)

    freqs = result[0]
    return freqs, result_SED

def _normalise(w):
    """Turn any non–negative weight array into a PDF."""
    w = np.asarray(w, dtype=float)
    if np.any(w < 0):                       # sanity check
        raise ValueError("Weights must be non-negative.")
    s = w.sum()
    if s == 0:
        raise ValueError("All weights are zero → cannot normalise.")
    return w / s

def grain_size_dist(a_list, C, log_a0, sigma_inv):
    """
    dn/da propto exp[C lna - 1/2 ((ln(a) - ln(a0))^2 * sigma_inv^2)]
    """
    exponent = C * np.log(a_list) - 0.5 * ( (np.log(a_list) - log_a0) * sigma_inv ) ** 2
    max_exponent = np.max(exponent)
    exponent -= max_exponent
    weights = np.exp(exponent)
    return _normalise(weights)

def generate_grain_ensemble_SED(env, a_list, d, C, log_a0, sigma_inv):
    a_distribution = grain_size_dist(a_list, C, log_a0, sigma_inv)
    a_distr_weighted = np.array(a_distribution) * np.array(a_list) 
    a_distr_weighted /= np.sum(a_distr_weighted)
    freqs, SED_list = generate_same_grains_SED(env, a_list, d)
    resultSED = np.sum(a_distr_weighted[:, np.newaxis] * SED_list, axis=0)
    return freqs, resultSED

def generate_synth_SED(SED_list, a_list, a_distribution, log_spacing_weight=False):
    if log_spacing_weight: # if the a_list is a even table at log-scale, then we will need to weight the distribution density
        a_weighted_distribution = np.array(a_distribution) * np.array(a_list) 
    else:
        a_weighted_distribution = np.array(a_distribution)
    a_weighted_distribution /= np.sum(a_weighted_distribution)
    resultSED = np.sum(a_weighted_distribution[:, np.newaxis] * np.array(SED_list), axis=0)
    return resultSED

def SED_list_given_env(env, normalise=True):
    params_list = []
    SED_list = []
    for d in d_tab:
        for C_val in C_tab:
            for sigma_inv in sigma_inv_tab:
                for log_a0 in log_a0_tab:
                    params_list.append((d, C_val, log_a0, sigma_inv))
                    freqs, result = generate_grain_ensemble_SED(env, a_tab, d, C_val, log_a0, sigma_inv)
                    if normalise:
                        result /= np.max(result)
                    SED_list.append(result)
    return np.array(params_list), freqs, np.array(SED_list)

from SpyDust.SED_fit import lognormal_sed_fit

def fit_sed_ensemble(freqs, sed_ensemble, thres=1e-3, parameter_list=None):
    """
    Fit SED ensemble to log-normal model.
    
    Parameters:
    -----------
    freqs : array_like
        Frequency array
    sed_ensemble : array_like
        2D array of SED values (rows are different realizations)
    thres : float, optional
        Threshold parameter for fitting. Default is 1e-3.
    """
    if sed_ensemble.ndim == 1:
        sed_ensemble = sed_ensemble.reshape(1, -1)
    n_sed = sed_ensemble.shape[0]
    fit_params = np.zeros((n_sed, 2))

    from tqdm import tqdm
    for i in tqdm(range(n_sed)):
        try:
            fit_params[i] = lognormal_sed_fit(freqs, sed_ensemble[i], thres=thres, brief_return=True)
        except:
            if parameter_list is not None:
                print(parameter_list[i])
            raise ValueError(f"Failed to fit SED {i}")
    if sed_ensemble.shape[0] == 1:
        fit_params = fit_params[0]
    return fit_params

if __name__ == "__main__":
    parameter_list, freqs, CNM_SED_list = SED_list_given_env(CNM_params)
    # Save the parameter list and SED list
    np.save('CNM_parameter_list.npy', parameter_list)
    np.save('CNM_SED_list.npy', CNM_SED_list)

    CNM_feature_list = fit_sed_ensemble(freqs, CNM_SED_list, parameter_list=parameter_list)

    from MomentEmu import PolyEmu

    # Create emulator with both forward and inverse capabilities
    CNM_feature_emulator = PolyEmu(parameter_list, 
                                    CNM_feature_list, 
                                    forward=True,              # Enable forward emulation: parameters → observables
                                    max_degree_forward=25,     # Max polynomial degree for forward mapping (lower for high-dimensional problems)
                                    RMSE_lower=0.01,
                                    fRMSE_tol=1e-1,
                                    )

    CNM_logSED_emulator = PolyEmu(parameter_list, 
                            np.log(CNM_SED_list), 
                            forward=True,              # Enable forward emulation: parameters → observables
                            max_degree_forward=20,     # Max polynomial degree for forward mapping (lower for high-dimensional problems)
                            RMSE_lower=0.01,
                            fRMSE_tol=3e-1,
                            )

    import pickle

    with open("CNM_feature_emulator.pkl", "wb") as f:
        pickle.dump(CNM_feature_emulator, f)

    with open("CNM_logSED_emulator.pkl", "wb") as f:
        pickle.dump(CNM_logSED_emulator, f)