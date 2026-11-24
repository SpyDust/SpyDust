"""Nearly parabolic trajectories used to regenerate the plasma integral tables."""
from functools import lru_cache

import numpy as np

from .util import DX_over_X, makelogtab, maketab


def compute_small_e():
    """Compute the small-eccentricity-limit table on the calling process."""
    gamma = makelogtab(1e-20, 10**1.5, 300)
    n = 20000
    linear = maketab(0., 1., n)
    logarithmic = makelogtab(1., 1e20, n)
    phase = np.exp(1j * np.pi / 6)
    u = phase * np.concatenate((linear, logarithmic))
    du = phase * np.concatenate((np.full(n, 1/n), DX_over_X(1., 1e20, n) * logarithmic))
    trajectory = u + u**3 / 3
    fcos = (u**2 - 1) / (u**2 + 1)**2
    fsin = u / (u**2 + 1)**2
    integrals = np.empty_like(gamma)
    for i, value in enumerate(gamma):
        oscillation = np.exp(1j * (value * trajectory))
        cosine = np.sum(np.real(oscillation * fcos * du))
        sine = np.sum(np.imag(oscillation * fsin * du))
        integrals[i] = 4 * cosine**2 + 16 * sine**2
    return gamma, integrals, gamma[-1]


@lru_cache(maxsize=1)
def small_e_tables():
    # Preserve the historical class and attribute interface for direct users.
    class smalletabs:
        Gamma_tab, smalle_tab, Gamma_max = compute_small_e()

    return smalletabs


def table_attribute(name):
    """Resolve legacy table attributes only when someone requests them."""
    if name == 'smalletabs':
        return small_e_tables()
    if name in ('Gamma_tab', 'smalle_tab', 'Gamma_max'):
        return getattr(small_e_tables(), name)
    raise AttributeError(name)
