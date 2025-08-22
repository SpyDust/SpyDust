import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add the root directory of the package to sys.path
sys.path.append('/Users/user/SpyDust/')

from SpyDust.util import cgsconst
from SpyDust.SpyDust import SpyDust, SpyDust_imp
from SpyDust.mpiutil import rank0

import SpyDust.SPDUST_as_is.spdust as spdust
import SpyDust.SPDUST_as_is.grain_properties as spdust_grain



CNM_params = {'nh' : 30, 'T': 100., 'Chi': 1, 'xh': 1.2e-3, 'xC': 3e-4, 'y' : 0, 'gamma': 0, 'dipole': 9.3, 'line':7}

import time

min_freq=1 # in GHz
max_freq=300 # in GHz
n_freq=500

t1 = time.time()
spy_SED_old = SpyDust(CNM_params, min_freq=min_freq, max_freq=max_freq, n_freq=n_freq, single_beta=True)
t2 = time.time()

t3 = time.time()
spy_SED_new = SpyDust_imp(CNM_params, min_freq=min_freq, max_freq=max_freq, n_freq=n_freq, single_beta=True)
t4 = time.time()

if rank0:
    print("Old", t2-t1)
    print("New", t4-t3)

    plt.loglog(spy_SED_old[0,:], spy_SED_old[1,:], label="Old")
    plt.loglog(spy_SED_new[0,:], spy_SED_new[1,:], linestyle='--',label="New")
    plt.loglog(spy_SED_old[0,:], np.abs(spy_SED_old[1,:]-spy_SED_new[1,:]), linestyle='-.',label="Ratio")
    plt.legend()
    plt.show()



