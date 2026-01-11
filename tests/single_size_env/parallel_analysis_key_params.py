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
from functools import partial

# Define environment classes (copied from main notebook)
class MC_env:

    a_min = 3.5e-8
    a_max = 3.5e-7
    beta_amin = -0.47
    beta_amax = 0.5
    xC_min = 1e-5
    xC_max = 1e-3

    nh_min = 3e2
    nh_max = 3e2
    T_min = 20
    T_max = 20
    chi_min = 0.01
    chi_max = 0.01
    xh_min = 0.0
    xh_max = 0.0
    y_min = 0.99
    y_max = 0.99

    def __init__(self):
        pass

    def generate_sample_coords(self, num):
        # randomly draw samples in log space
        self.nh_samples = np.exp(np.random.uniform(np.log(self.nh_min), np.log(self.nh_max), num))
        self.T_samples = np.exp(np.random.uniform(np.log(self.T_min), np.log(self.T_max), num))
        self.chi_samples = np.exp(np.random.uniform(np.log(self.chi_min), np.log(self.chi_max), num))
        self.xh_samples = np.random.uniform(self.xh_min, self.xh_max, num)
        self.xC_samples = np.exp(np.random.uniform(np.log(self.xC_min), np.log(self.xC_max), num))
        self.y_samples = np.random.uniform(self.y_min, self.y_max, num)

        self.a_samples = np.exp(np.random.uniform(np.log(self.a_min), np.log(self.a_max), num))
        self.beta_samples = np.random.uniform(self.beta_amin, self.beta_amax, num)

        self.params = np.vstack((self.nh_samples, self.T_samples, self.chi_samples, self.xh_samples, self.xC_samples, self.y_samples, self.a_samples, self.beta_samples)).T
        return 

    def generate_SED(self, env, a, beta, min_freq=0.1, max_freq=1000.0, n_freq=500, tumbling=True):
        a2_aux = 1.0 if tumbling else 0.0
        result = SpyDust_given_grain_size_shape(env, a, beta, 
                                                min_freq=min_freq, max_freq=max_freq, n_freq=n_freq,
                                                N_angular_Omega=800,
                                                a2=a2_aux)
        return result[0], result[1]
    
    def _process_single_sample(self, i, min_freq, max_freq, n_freq):
        """Process a single sample - helper function for parallelization"""
        env = {'nh': self.nh_samples[i], 'T': self.T_samples[i], 'Chi': self.chi_samples[i], 
               'xh': self.xh_samples[i], 'xC': self.xC_samples[i], 'y': self.y_samples[i], 
               'gamma': 0, 'dipole': 9.3}
        
        freq, result_SED_tumbling = self.generate_SED(env, 
                                                a=self.a_samples[i], 
                                                beta=self.beta_samples[i],
                                                min_freq=min_freq, max_freq=max_freq, n_freq=n_freq, tumbling=True
                                                )
        # _, result_SED_no_tumbling = self.generate_SED(env,
        #                                         a=self.a_samples[i],
        #                                         beta=self.beta_samples[i],
        #                                         min_freq=min_freq, max_freq=max_freq, n_freq=n_freq, tumbling=False
        #                                         )
        return freq, result_SED_tumbling#, #result_SED_no_tumbling

    def analysis(self, num, min_freq=0.1, max_freq=1000.0, n_freq=500, n_jobs=None):
        """
        Perform SED analysis with optional parallelization
        
        Parameters:
        -----------
        num : int
            Number of samples to process
        min_freq, max_freq, n_freq : float, float, int
            Frequency range and resolution parameters
        n_jobs : int or None
            Number of parallel jobs. If None, uses all available CPUs.
            Set to 1 for sequential processing.
        """
        self.generate_sample_coords(num)

        if n_jobs == 1:
            # Sequential processing (original behavior)
            tumbling_result = []
            # no_tumbling_result = []

            for i in tqdm.tqdm(range(num)):
                freq, result_SED_tumbling, result_SED_no_tumbling = self._process_single_sample(
                    i, min_freq, max_freq, n_freq)
                tumbling_result.append(result_SED_tumbling)
                # no_tumbling_result.append(result_SED_no_tumbling)

            return freq, tumbling_result#, no_tumbling_result
        
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

# Parameter array: MC_instance.params; Data array: Y_tumbling -- used for global sensitivity analysis

class DC_env(MC_env):
    a_min = 3.5e-8
    a_max = 3.5e-7
    beta_amin = -0.47
    beta_amax = 0.5
    xC_min = 1e-7
    xC_max = 1e-5

    nh_min = 1e4
    nh_max = 1e4
    T_min = 10
    T_max = 10
    chi_min = 1e-4
    chi_max = 1e-4
    xh_min = 0.0
    xh_max = 0.0
    y_min = 0.999
    y_max = 0.999

class HII_env(MC_env):
    a_min = 3.5e-8
    a_max = 3.5e-7
    beta_amin = -0.47
    beta_amax = 0.5
    nh_min = 10.0
    nh_max = 1e4

    T_min = 10000.0
    T_max = 10000.0
    chi_min = 1e4
    chi_max = 1e4
    xh_min = 0.999
    xh_max = 0.999
    xC_min = 1e-4
    xC_max = 1e-4
    y_min = 0.0
    y_max = 0.0

if __name__ == "__main__":
    # Example usage
    mc_instance = MC_env()
    freq, tumbling_results = mc_instance.analysis(num=50000, n_jobs=20)
    # Save results to files
    np.savetxt("data/freq.txt", freq)
    np.savetxt("data/MC_key_params.txt", mc_instance.params)
    np.savetxt("data/MC_SED_key_params.txt", tumbling_results)
    # np.savetxt("data/MC_SED_no_tumbling.txt", no_tumbling_results)

    DC_instance = DC_env()
    freq, tumbling_results_DC = DC_instance.analysis(num=50000, n_jobs=20)
    # Save results to files
    np.savetxt("data/DC_key_params.txt", DC_instance.params)
    np.savetxt("data/DC_SED_key_params.txt", tumbling_results_DC)
    # np.savetxt("data/DC_SED_no_tumbling.txt", no_tumbling_results_DC)

    HII_instance = HII_env()
    freq, tumbling_results_HII = HII_instance.analysis(num=50000, n_jobs=20)
    # Save results to files
    np.savetxt("data/HII_key_params.txt", HII_instance.params)
    np.savetxt("data/HII_SED_key_params.txt", tumbling_results_HII)
    # np.savetxt("data/HII_SED_no_tumbling.txt", no_tumbling_results_HII)



