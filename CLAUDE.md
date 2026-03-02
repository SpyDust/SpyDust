# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SpyDust is a Python package for modeling spinning dust (anomalous microwave emission) radiation in astrophysical environments. It is an improved reimplementation of the IDL SPDUST code, described in the paper at https://arxiv.org/abs/2412.03431.

## Commands

**Install for development:**
```bash
pip install -e . --no-deps
```

**Build distribution:**
```bash
python -m build
```

**Run a notebook-based validation** (no traditional test suite exists; validation is via Jupyter notebooks in `paper-products/`):
```bash
jupyter notebook paper-products/spydust_paper/SpDust_vs_SpyDust.ipynb
```

**Dependencies:** numpy (<2.4), scipy, mpi4py (required); pandas (optional, only for free-free emission).

**CI/CD:** PyPI publishing is triggered automatically on GitHub releases via `.github/workflows/python-publish.yml`.

## Architecture

The package lives in `SpyDust/`. There are no traditional unit tests; correctness is validated by cross-checking against the original IDL SPDUST via the `SPDUST_as_is/` compatibility subpackage and research notebooks in `paper-products/`.

### Core Data Flow

1. **User calls `SpyDust()` in [SpyDust/SpyDust.py](SpyDust/SpyDust.py)** with ISM environment parameters (`nH`, `T`, `Chi`, `xh`, `xC`, `y`, `gamma`, `dipole`, `line`).
2. **For each grain size**, the code computes:
   - Grain shape/geometry via [SpyDust/Grain.py](SpyDust/Grain.py) (cylindrical/ellipsoidal, moments of inertia, N_C, N_H)
   - Charge distribution via [SpyDust/charge_dist.py](SpyDust/charge_dist.py) (photoemission/collision balance)
   - Excitation and damping rates from [SpyDust/infrared.py](SpyDust/infrared.py), [SpyDust/collisions.py](SpyDust/collisions.py), [SpyDust/plasmadrag.py](SpyDust/plasmadrag.py), [SpyDust/H2_photoemission.py](SpyDust/H2_photoemission.py)
3. **Angular momentum distribution** f(ω) is solved in [SpyDust/AngMomDist.py](SpyDust/AngMomDist.py).
4. **Emissivity spectrum** is integrated in [SpyDust/SED.py](SpyDust/SED.py) and returned.

### Module Responsibilities

| Module | Role |
|---|---|
| `SpyDust.py` | Primary public API; orchestrates all other modules |
| `Grain.py` | Grain shape parameters, moments of inertia, N_C/N_H counts |
| `charge_dist.py` | Grain charge distribution equilibrium |
| `AngMomDist.py` | Angular momentum (rotation rate) distribution |
| `infrared.py` | IR emission and damping rates (~1360 lines, the largest module) |
| `plasmadrag.py` | Plasma drag corrections |
| `collisions.py` | Collision physics with neutrals/ions |
| `SED.py` | Spectral energy distribution computation |
| `SED_fit.py` | SED fitting and degeneracy analysis |
| `mom_exp.py` | Moment expansion perturbation method |
| `full_Stokes.py` | Stokes parameters for polarization |
| `util.py` | CGS constants, interpolation helpers |
| `mpiutil.py` | MPI parallelization utilities |
| `SPDUST_as_is/` | Exact Python port of original IDL SPDUST for cross-validation |

### Pre-computed Data Files

`SpyDust/Data_Files/` contains lookup tables (`.txt`, `.npz`, `.out`) loaded at module import time by `infrared.py`, `charge_dist.py`, and `plasmadrag.py`. These include IR integral tables (`FIR_integral_*.txt`, `GIR_integral_*.txt`), photoemission rates (`jpeisrf_data.npz`), and Weingartner & Draine 2001a grain size distributions (`sizedists_table1.out`).

### Conventions

- Physics equations are referenced inline (e.g., `# Eq. (24) in AHD09`). AHD09 refers to Ali-Haïmoud, Hirata & Dickinson 2009; DL98b refers to Draine & Lazarian 1998b.
- CGS units throughout.
- `@njit` (Numba) decorators are commented out — do not re-enable without testing, as Numba support was intentionally removed.
- The grain shape parameter `beta` (β) controls the cylindrical/ellipsoidal aspect ratio and is central to many computations.
- `line` parameter in the public API selects the Weingartner & Draine 2001a grain size distribution (integer index into the lookup table).
