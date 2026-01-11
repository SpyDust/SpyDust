#!/usr/bin/env python3
"""Utilities for running the Monte-Carlo SED analysis in parallel.

This module makes it possible to execute the heavy sampling step from either a
stand-alone Python script (via ``python parallel_analysis.py``) or by importing
``run_analysis`` inside a notebook.  The multiprocessing logic lives in this
module so that workers spawned by ``multiprocessing`` can import the code
without running into the usual Jupyter ``__main__`` pickling issues.
"""

from __future__ import annotations

from multiprocessing import cpu_count

import numpy as np
from tqdm.auto import tqdm
import SpyDust.Grain as Grain
from SpyDust.util import makelogtab, cgsconst
from SpyDust.SpyDust import SpyDust_given_grain_size_shape

# from SpyDust.SPDUST_as_is import emissivity
# def spdust_SED_given_grain(env, a, tumbling=True):
#         # Determine the dipole moment
#     mu_1d_7 = env['dipole']
#     dipole_mom = mu_1d_7 / np.sqrt(Grain.N_C(1e-7) + Grain.N_H(1e-7)) * cgsconst.debye
#     ip=2./3.
#     result=emissivity.dP_dnu_dOmega(env, a, dipole_mom,ip, 20, tumbling=tumbling)
#     return result

import tqdm 
from multiprocessing import Pool, cpu_count
import functools
from functools import partial


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

# Define MC environment




class MC_env:

    a_min = 3.5e-8
    a_max = 3.5e-7
    num_a = 20
    a_tab = makelogtab(a_min, a_max, num_a)


    beta_amin = -0.47
    beta_amax = -0.2
    num_beta = 20
    # beta_tab  = np.linspace(beta_amin, beta_amax, num_beta)
    beta_tab = makelogtab(beta_amin, beta_amax, num_beta)


    xC_min = 1e-5
    xC_max = 1e-3
    num_xC = 30
    xC_tab = makelogtab(xC_min, xC_max, num_xC)

    a_beta_env_tab = np.array(np.meshgrid(a_tab, beta_tab, xC_tab)).T.reshape(-1, 3)

    a_samples = a_beta_env_tab[:, 0]
    beta_samples = a_beta_env_tab[:, 1]
    xC_samples = a_beta_env_tab[:, 2]

    dim = a_beta_env_tab.shape[0]

    def __init__(self):
        pass

    # def generate_sample_coords(self, num):
    #     # randomly draw samples in log space
    #     self.nh_samples = np.exp(np.random.uniform(np.log(self.nh_min), np.log(self.nh_max), num))
    #     self.T_samples = np.exp(np.random.uniform(np.log(self.T_min), np.log(self.T_max), num))
    #     self.chi_samples = np.exp(np.random.uniform(np.log(self.chi_min), np.log(self.chi_max), num))
    #     self.xh_samples = np.random.uniform(self.xh_min, self.xh_max, num)
    #     self.xC_samples = np.exp(np.random.uniform(np.log(self.xC_min), np.log(self.xC_max), num))
    #     self.y_samples = np.random.uniform(self.y_min, self.y_max, num)

    #     self.a_samples = np.exp(np.random.uniform(np.log(self.a_min), np.log(self.a_max), num))
    #     self.beta_samples = np.random.uniform(self.beta_amin, self.beta_amax, num)

    #     self.params = np.vstack((self.nh_samples, self.T_samples, self.chi_samples, self.xh_samples, self.xC_samples, self.y_samples, self.a_samples, self.beta_samples)).T
    #     return 

    def generate_SED(self, env, a, beta, min_freq=0.1, max_freq=1000.0, n_freq=500, tumbling=True):
        a2_aux = 1.0 if tumbling else 0.0
        result = SpyDust_given_grain_size_shape(env, a, beta, 
                                                min_freq=min_freq, max_freq=max_freq, n_freq=n_freq,
                                                N_angular_Omega=800,
                                                a2=a2_aux)
        return result[0], result[1]
    
    def _process_single_sample(self, i, min_freq, max_freq, n_freq):
        """Process a single sample - helper function for parallelization"""
        env = {'nh': 3e2, 'T': 20, 'Chi': 0.01, 
               'xh': 0, 'xC': self.xC_samples[i], 'y': 0.99, 
               'gamma': 0, 'dipole': 9.3}
        
        freq, result_SED_tumbling = self.generate_SED(env, 
                                                a=self.a_samples[i], 
                                                beta=self.beta_samples[i],
                                                min_freq=min_freq, max_freq=max_freq, n_freq=n_freq, tumbling=True
                                                )

        return freq, result_SED_tumbling#, #result_SED_no_tumbling

    def generate(self, min_freq=0.1, max_freq=1000.0, n_freq=500, n_jobs=None):
        """
        Perform SED analysis with optional parallelization
        
        Parameters:
        -----------
        min_freq, max_freq, n_freq : float, float, int
            Frequency range and resolution parameters
        n_jobs : int or None
            Number of parallel jobs. If None, uses all available CPUs.
            Set to 1 for sequential processing.
        """
        num = self.dim

        if n_jobs == 1:
            # Sequential processing (original behavior)
            tumbling_result = []
            # no_tumbling_result = []

            for i in tqdm.tqdm(range(num)):
                freq, result_SED_tumbling = self._process_single_sample(
                    i, min_freq, max_freq, n_freq)
                tumbling_result.append(result_SED_tumbling)

            return freq, tumbling_result
        
        else:
            # Parallel processing
            if n_jobs is None:
                n_jobs = cpu_count()
            
            print(f"Processing {num} samples using {n_jobs} parallel workers...")
            
            # Create partial function with fixed parameters
            process_func = partial(self._process_single_sample, 
                                 min_freq=min_freq, max_freq=max_freq, n_freq=n_freq)
            
            # Process in parallel
            with Pool(processes=n_jobs) as pool:
                # Use imap for progress tracking
                results = list(tqdm.tqdm(
                    pool.imap(process_func, range(num)), 
                    total=num, 
                    desc="Processing samples"
                ))
            
            # Unpack results
            freq = results[0][0]  # Frequency array should be the same for all
            tumbling_result = [r[1] for r in results]
            # no_tumbling_result = [r[2] for r in results]

            return freq, tumbling_result#, no_tumbling_result
        

def a_beta_env_dist(gamma, log_a0, sigma, log_beta0, delta, log_p_c, log_p_width):
    """Generate a distribution function for (a, beta, p) based on given parameters.
    
    Note that we assume the log-space sampling...
    dn/da propto exp[(gamma - 1) lna - 1/2 ((ln(a) - ln(a0) + sigma**2/2)^2 / sigma^2)]
    equivalent to
    dn/dlna propto exp[gamma * lna - 1/2 ((ln(a) - ln(a0) + sigma**2/2)^2 / sigma^2)]
    """
    def dist_func(theta):
        a, beta, p = theta[:, 0], theta[:, 1], theta[:, 2]
        ln_a_list = np.log(a)
        # exponent_a = (gamma-1)  * ln_a_list - 0.5 * ( (ln_a_list - log_a0 + sigma**2/2) / sigma ) ** 2
        exponent_a = gamma  * ln_a_list - 0.5 * ( (ln_a_list - log_a0 + sigma**2/2) / sigma ) ** 2


        ln_beta_t_tab = np.log( beta + 0.5 )
        # exponent_beta = - ln_beta_t_tab - 0.5 * ( (ln_beta_t_tab  - log_beta0 + delta**2/2 ) / delta ) ** 2
        exponent_beta = - 0.5 * ( (ln_beta_t_tab  - log_beta0 + delta**2/2 ) / delta ) ** 2


        log_p = np.log(p)  
        # exponent_env = - log_p - 0.5 * ( (log_p  - log_p_c + log_p_width**2/2 ) / log_p_width ) ** 2
        exponent_env = - 0.5 * ( (log_p  - log_p_c + log_p_width**2/2 ) / log_p_width ) ** 2


        exponent = exponent_a + exponent_beta + exponent_env

        max_exponent = np.max(exponent)
        exponent -= max_exponent
        weights = np.exp(exponent)
        normalised_weights = weights / np.sum(weights)
        return normalised_weights
    return dist_func

def synthesize_SED(gamma, sigma, log_p_c, log_p_width, SED_tab):
    log_a0 = 0.5 * (np.log(MC_env.a_min) + np.log(MC_env.a_max))
    log_beta0, delta = np.log(0.05), 0.1  # Fixed beta distribution parameters
    params_tab = MC_env.a_beta_env_tab
    aux_func = a_beta_env_dist(gamma, log_a0, sigma, log_beta0, delta, log_p_c, log_p_width)
    prob_weights = aux_func(params_tab)
    SED_syn = np.sum(prob_weights[:, np.newaxis] * SED_tab, axis=0)
    return SED_syn


if __name__ == "__main__":
    # Example usage
    mc_instance = MC_env()

    try:
        freqs = np.loadtxt("data/MC_emu/freq.txt")
        SED_tab = np.loadtxt("data/MC_emu/MC_SED_array.txt")
    except:
        freqs, SED_tab = mc_instance.generate(n_jobs=20)
        # check all values are finite
        assert np.all(np.isfinite(SED_tab)), "SED_tab contains non-finite values"
        np.savetxt("data/MC_emu/freq.txt", freqs)
        np.savetxt("data/MC_emu/MC_SED_array.txt", SED_tab)

    # # filter out non-finite samples
    # SED_tab = np.array(SED_tab)
    # mask = np.all(np.isfinite(SED_tab), axis=1)
    # SED_tab = SED_tab[mask, :]



    N_samples = 20000
    gamma_list = np.random.uniform(-2, 2, N_samples)
    # log_a0_list = np.random.uniform(np.log(MC_env.a_min), np.log(MC_env.a_max), N_samples)
    sigma_list = np.random.uniform(0.1, 5, N_samples)
    # sigma_list = np.ones(N_samples) * 0.5

    log_pc_list = np.random.uniform(np.log(MC_env.xC_min), np.log(MC_env.xC_max), N_samples)
    # log_pc_list = np.ones(N_samples) * np.log(1e-4)
    log_pwidth_list = np.random.uniform(0.1, 10, N_samples)

    param_array_training = np.column_stack((gamma_list, sigma_list, log_pc_list, log_pwidth_list)) # Generate the parameter array (N_samples, 4)
    def calculate_moments(dist_weights):
        a_list = MC_env.a_beta_env_tab[:, 0] * 1e8  # Convert to Angstrom
        xC_list = MC_env.a_beta_env_tab[:, 2]
        a_mean = np.sum(dist_weights * a_list)
        a_variance = np.sum(dist_weights * (a_list - a_mean)**2)
        xC_mean = np.sum(dist_weights * xC_list)
        xC_variance = np.sum(dist_weights * (xC_list - xC_mean)**2)

        return [a_mean, a_variance, xC_mean, xC_variance]

    log_beta0, delta = np.log(0.05), 0.1  # Fixed beta distribution parameters
    log_a0 = 0.5 * (np.log(MC_env.a_min) + np.log(MC_env.a_max))
    params_tab = MC_env.a_beta_env_tab
    moments_training = []
    SED_array_training = []

    for i in tqdm.tqdm(range(N_samples)):
        aux_func = a_beta_env_dist(gamma_list[i], log_a0, sigma_list[i], log_beta0, delta, log_pc_list[i], log_pwidth_list[i])
        prob_weights = aux_func(params_tab)
        moments = calculate_moments(prob_weights)
        moments_training.append(moments)
        SED_syn = np.sum(prob_weights[:, np.newaxis] * SED_tab, axis=0)
        SED_array_training.append(SED_syn)
    moments_training = np.array(moments_training)
    SED_array_training = np.array(SED_array_training)

    from SpyDust.SED_fit import fit_sed_ensemble, measure_sed_log_moments_batch
    features_training = fit_sed_ensemble(freqs, SED_array_training, thres=1e-3)
    SED_log_moments = measure_sed_log_moments_batch(freqs, SED_array_training, thres=1e-2, excess_kurtosis=True)


    np.savetxt("data/MC_emu/MC_dist_Mom_4.txt", moments_training)
    np.savetxt("data/MC_emu/MC_SED_feature_4.txt", features_training)
    np.savetxt("data/MC_emu/MC_dist_params_4.txt", param_array_training)
    np.savetxt("data/MC_emu/MC_SED_logmoments_4.txt", SED_log_moments)




