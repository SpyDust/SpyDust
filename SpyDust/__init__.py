# Initialisation file for the 'SpyDust' package.

# Initialize package variables
package_version = "1.1"

# Importing submodules for ease of access
import os
SpDust_data_dir = os.path.join(__path__[0], 'Data_Files/')
#SpDust_data_dir = os.path.join("/home/phil/software/SpyDust/SpyDust", 'Data_Files/')


from . import AngMomDist, charge_dist, collisions, free_free, Grain, \
              H2_photoemission, infrared, mpiutil, plasmadrag, SED, \
              SpyDust, util
