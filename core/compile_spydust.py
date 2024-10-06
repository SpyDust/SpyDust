import numpy as np
import os

import yaml


with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Define the directory path
SpDust_data_dir = '/Users/user/SPDUST.2.01/Data_Files/'

SpDust_data_dir = config['SpDust_data_dir']

# Define the path to the size distribution file
size_dist_file = os.path.join(SpDust_data_dir, 'sizedists_table1.out')

class size_dist_arrays:
    bc1e5_tab, alpha_tab, beta_tab, at_tab, ac_tab, C_tab = \
    np.loadtxt(size_dist_file, usecols=(1, 3, 4, 5, 6, 7), unpack=True, comments=';')

class size_params():
    bc, alpha_g, beta_g, at_g, ac_g, C_g = None, None, None, None, None, None

    def __call__(self, line):
        self.bc = size_dist_arrays.bc1e5_tab[line] * 1e-5
        self.alpha_g = size_dist_arrays.alpha_tab[line]
        self.beta_g = size_dist_arrays.beta_tab[line]
        self.at_g = size_dist_arrays.at_tab[line] * 1e-4
        self.ac_g = size_dist_arrays.ac_tab[line] * 1e-4
        self.C_g = size_dist_arrays.C_tab[line]
        pass

import spdust.infrared as ir

def spydust()


