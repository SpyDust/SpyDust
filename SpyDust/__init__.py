# Initialisation file for the 'SpyDust' package.

# Initialize package variables
package_version = "1.0"

# Importing submodules for ease of access
import os
from .main import *
SpDust_data_dir = os.path.join(__path__[0], 'Data_Files/')
