# SpyDust

**SpyDust** is an advanced Python package for modeling spinning dust radiation in astrophysical environments. Building upon previous models (the IDL [**SPDUST**](https://arxiv.org/pdf/1003.4732)), SpyDust offers enhanced capabilities and corrections, making it a valuable tool for researchers in astrophysics and related fields.

---

## Features

- **Comprehensive Grain Shape Modeling**: Considers a wide range of grain geometries, providing corresponding grain dynamics, directional radiation fields, and angular momentum transports.

- **Updated Physical Processes**: Incorporates corrections and extensions, including updated expressions for the effects of electrical dipole radiation back-reaction and plasma drag on angular momentum dissipation.

- **Parallisation**: Functions can be run in parallel (implemented by mpi4py) by simply setting the corresponding keyword to True.

- **SPDUST as is** mode: Provides a submodule, **SPDUST_as_is**, which is exactly the full equivalent of the IDL spdust. Just in case the user wants to stick with the spdust simulation.

---

## Requirements

SpyDust requires Python 3.7 or higher (up to Python 3.9) and the following dependencies:

- `numpy <= 2.0`
- `scipy`
- `numba`
- `logging`
- `mpi4py`
- `pandas`

---

## Installation

You can install SpyDust using pip:

```bash
pip install SpyDust
```

---

# Usage

Import the package in your project and explore its functionalities for modeling spinning dust radiation. Data files required for computations are bundled with the package.

---

## Example usage:

```python

```

--- 

# Resources

- **Author**: Zheng Zhang
- **License**: MIT License
- **Paper**: https://arxiv.org/abs/2412.03431 (Z Zhang and J Chluba 2024)

---

## History

- **Version 1.0.0**: Initial release of SpyDust, introducing comprehensive grain shape modeling, updated physical processes, and degeneracy analysis tools.
- **Future Versions**: Upcoming releases will include an SED fitting tool and improved grain rotation models.

---

## Resources

- **Author**: Zheng Zhang
- **License**: MIT License
- **Collaboration**: [SpyDust GitHub Collaboration](https://github.com/SpyDust/SpyDust)
- ** Spdust Documentation**: We refer users to the website of [spdust](https://cosmo.nyu.edu/yacine/spdust/spdust.html) for better documentation of environmental parameters 

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

For a detailed understanding of the underlying models and theoretical background, refer to the publication: [SpyDust: an improved and extended implementation for modeling spinning dust radiation](https://arxiv.org/abs/2412.03431).