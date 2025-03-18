# SpyDust

[**SpyDust**](https://arxiv.org/abs/2412.03431) is an advanced Python package for modeling spinning dust radiation in astrophysical environments. Building upon previous models (the IDL [**SPDUST**](https://arxiv.org/pdf/1003.4732)), SpyDust offers enhanced capabilities and corrections, making it a valuable tool for researchers in astrophysics and related fields.

For a detailed understanding of the underlying models and theoretical background, refer to the publication: [SpyDust: an improved and extended implementation for modeling spinning dust radiation](https://arxiv.org/abs/2412.03431).

---

## Features

- **Comprehensive Grain Shape Modeling**: Considers a wide range of grain geometries, providing corresponding grain dynamics, directional radiation fields, and angular momentum transports.

- **Updated Physical Processes**: Incorporates corrections and extensions, including updated expressions for the effects of electrical dipole radiation back-reaction and plasma drag on angular momentum dissipation.

- **Parallisation**: Functions can be run in parallel (implemented by mpi4py) by simply setting the corresponding keyword to True.

- **SPDUST as is** mode: Provides a submodule, **SPDUST_as_is**, which is exactly the full equivalent of the IDL spdust. Just in case the user wants to stick with the spdust simulation.

- **Example notebooks**: Provide some Jupyter notebooks to help users get started with this package.

---

## Requirements

SpyDust requires Python 3.7 or higher (up to Python 3.9) and the following dependencies:

**Required**:
- `numpy`
- `scipy`
- `mpi4py`

**Optional**:
- `pandas`: This is not needed for spinning dust spectra, but if free free emission is also desired (using free_free.py), then it is needed.

---

## Installation

You can install SpyDust using pip:

```bash
pip install SpyDust --no-deps
```

Otherwise, you can directly `git clone' this repo and set up the environment by yourself.

---

# Usage

Import the package in your project and explore its functionalities for modeling spinning dust radiation. Data files required for computations are bundled with the package.

---

## Example usage:

Here is an example of using the `SpyDust.SpyDust' function to generate a spectrum for a sample CNM environment:
```python
CNM_params = {'nh' : 30, 'T': 100., 'Chi': 1, 'xh': 1.2e-3, 'xC': 3e-4, 'y' : 0, 'gamma': 0, 'dipole': 9.3, 'line':7}

# The parameters are as follows: 
# 'nH': total hydrogen number density (cm3), 
# 'T': gas temperature (K), 
# 'chi': intensity of the radiation field relative to the average interstellar radiation field, 
# 'xh': hydrogen ionization fraction, 
# 'xC': ionized carbon fractional abundance, 
# 'y': molecular hydrogen fractional abundance, 
# 'gamma': H2 formation efficiency, 
# 'dipole': rms dipole moment for dust grains.

min_freq=1 # in GHz
max_freq=300 # in GHz
n_freq=500

spectrum = SpyDust.SpyDust(CNM_params, min_freq=min_freq, max_freq=max_freq, n_freq=n_freq, single_beta=True)
# Here the boolean keyword `single_beta' means: for any given grain size, consider only one value of the shape parameter beta.
```

Instead of using the SpyDust rotation distribution method described in the paper, you can generate spectra using your own arbitrary distributions of configuration parameters as inputs to the `SpyDust.SED' function.

--- 

# Resources

- **Author**: Zheng Zhang
- **License**: MIT License
- **Paper**: https://arxiv.org/abs/2412.03431 (Z Zhang and J Chluba 2024)

---

## History

- **Version 1.0.0**: Initial release of SpyDust, introducing comprehensive grain shape modeling, updated physical processes, and degeneracy analysis tools.

- **Version 1.0.1**: Numba dependency removed and installation issues fixed. 

## TODO (Future Versions)
- Future upgrades will include features such as the SED fitting tool (based on perturbation statistics tools like moment expansion) and improve the treatment of the grain rotation distribution.

---

## Resources

- **Author**: Zheng Zhang
- **License**: This project is licensed under the MIT License - see the LICENSE file for details.
- **Collaboration**: [SpyDust GitHub Collaboration](https://github.com/SpyDust/SpyDust)
- **Spdust Documentation**: We refer users to the website of [spdust](https://cosmo.nyu.edu/yacine/spdust/spdust.html) for better documentation of environmental parameters 

---


