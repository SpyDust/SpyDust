# SpyDust

SpyDust models the microwave emission from rotating dust grains. Given gas conditions, a radiation field and a grain population, it calculates charge and rotation distributions and sums the emission from each grain size and shape. The package also includes tools for spectral fitting, moment expansion and Stokes spectra. [SpyDust paper](https://arxiv.org/abs/2412.03431)

## Installation

Install an MPI runtime (Open MPI or MPICH), then install SpyDust in the Python environment used by your terminal or Jupyter kernel.

```bash
python -m pip install SpyDust
```

SpyDust requires NumPy, SciPy and mpi4py and includes the model's lookup tables. Use `python -m pip install 'SpyDust[plot]'` for plotting. To run the research notebooks, clone the repository, check out `v1.4.0`, and run `python -m pip install -e '.[notebooks]'` from its root.

## Calculate a spectrum

This example uses cold neutral medium (CNM) conditions and the fixed-shape model. It can be saved as `cnm.py` and run with `python cnm.py`.

```python
from SpyDust.SpyDust import SpyDust

CNM_params = {
    "nh": 30.0,       # Total hydrogen number density, cm^-3
    "T": 100.0,       # Gas temperature, K
    "Chi": 1.0,       # Intensity relative to the reference interstellar radiation field
    "xh": 1.2e-3,     # Hydrogen ion fraction
    "xC": 3e-4,       # Ionized carbon abundance per H nucleus
    "y": 0.0,         # Molecular hydrogen fraction
    "gamma": 0.0,     # H2 formation efficiency
    "dipole": 9.3,    # Intrinsic RMS dipole moment at grain radius 10^-7 cm, in debye
    "line": 7,        # One-based row of Weingartner & Draine (2001a), Table 1
}

frequency_GHz, mode_spectrum = SpyDust(
    CNM_params, min_freq=1, max_freq=400, n_freq=200, single_beta=True
)
emissivity = mode_spectrum / 4  # Jy sr^-1 cm^2 per H nucleus
```

The first row gives logarithmic frequency-bin centres in GHz. The second row retains the historical SpyDust radiation-mode normalization. Divide it by four to obtain emissivity per Hz per steradian, in Jy sr⁻¹ cm² H⁻¹. For a transverse rotating dipole the Larmor power is 2μ²ω⁴/(3c³). Averaging over solid angle and changing frequency from ω to ν gives μ²ω⁴/(3c³) times the rotation probability density, one quarter of the coefficient in the SpyDust mode sum. The legacy `SPDUST` entry point already uses the conventional emissivity normalization. To save the two returned rows as columns in the historical convention, pass `output_file="cnm.txt"`. With MPI, run the same script on every rank, for example `mpiexec -n 2 python cnm.py`. Every rank receives the spectrum and rank zero writes the requested file.

`single_beta=True` uses one shape per radius, with the specified disk thickness below the size transition and spherical grains above it. The default `single_beta=False` averages over the conditional shape distribution at each radius. Its default is `Nbeta=101`. Set `Nbeta` to check convergence for your environment and frequency range. `spdust_plasma=True` selects the legacy plasma prescription within the modern model. To reproduce the original SPDUST model, use `from SpyDust.SPDUST_as_is.spdust import SPDUST`. [Original SPDUST documentation](https://cosmo.nyu.edu/yacine/spdust/spdust.html)

## Validation and examples

Install pytest in the development environment, then run `python -m pytest tests`. The compatibility tests compare the legacy spectra in seven ISM environments with archived IDL results. The modern model has separate checks for grain shapes, angular emission, plasma drag and photoelectric excitation. The research notebooks illustrate the wider analysis. Their stored plots are historical. Run them in a scratch copy to regenerate the figures with the current model. [IDL comparison](https://github.com/SpyDust/SpyDust/blob/v1.4.0/paper-products/spydust_paper/spdust_IDLvsPython.ipynb) · [Grain-shape comparison](https://github.com/SpyDust/SpyDust/blob/v1.4.0/paper-products/spydust_paper/SpDust_vs_SpyDust.ipynb) · [Stokes example](https://github.com/SpyDust/SpyDust/blob/v1.4.0/paper-products/full_stokes/test_funcs.ipynb)

To supply your own rotation or orientation distribution, use `SpyDust.SED.SED` or `SpyDust.full_Stokes.full_Stokes_SED`. Spectral fits and moment expansions are available in `SpyDust.SED_fit` and `SpyDust.mom_exp`.

## Changes in 1.4.0

The modern model now resolves the conditional shape distribution, evaluates the shape-dependent plasma frequency and scales photoelectric excitation with the radiation field. Rotation densities are normalized on a physical spin grid independently of the observing band. The rotational solver uses the largest principal inertia and returns the density in Ω = L/Iref for both oblate and prolate grains. These changes can alter spectra and radiation-field derivatives relative to the 1.3.2 release. For emission, the population model uses empirical internal alignment, including alignment for nearly spherical grains. Its rotational damping and excitation use the selected tumbling setting. For isotropic internal orientations, integration over the tabulated spin frequencies resolves narrow rotation features and the low-frequency tail of the fourth mode.

The v2 lognormal fit retains its return tuple and reports the physical peak. Its redundant power-law exponent is fixed to zero. The moment-expansion cache tracks the physical settings used to construct each basis. [SpyDust equations](https://arxiv.org/html/2412.03431v2)

## History

- Version 1.3 added spectral fitting, moment expansion, full Stokes and the spectral-feature notebooks.
- Version 1.2 fixed example notebook imports.
- Version 1.1 removed the Numba dependency and fixed installation issues.
- Version 1.0 introduced the Python implementation, extended grain shapes and degeneracy analysis.

SpyDust's original code is released under the MIT license. Its MPI utilities include code adapted from caput under GPL version 2. [MIT license](https://github.com/SpyDust/SpyDust/blob/v1.4.0/LICENSE) · [caput license](https://github.com/SpyDust/SpyDust/blob/v1.4.0/LICENSE.caput) · [Project repository](https://github.com/SpyDust/SpyDust)
