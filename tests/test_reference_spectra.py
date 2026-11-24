"""End-to-end checks against the IDL spectra distributed with the paper."""
from pathlib import Path

import numpy as np
import pytest

from SpyDust.SPDUST_as_is.spdust import SPDUST


IDL = Path(__file__).resolve().parents[1] / "paper-products/spydust_paper/IDL_outputs"


@pytest.mark.parametrize("phase, nh, T, Chi, xh, xC, y, tumbling", [
    ("DC", 1e4, 10, 1e-4, 0, 1e-6, 0.999, False),
    ("MC", 300, 20, 1e-2, 0, 1e-4, 0.99, False),
    ("CNM", 30, 100, 1, 1.2e-3, 3e-4, 0, True),
    ("WNM", 0.4, 6000, 1, 0.1, 3e-4, 0, True),
    ("WIM", 0.1, 8000, 1, 0.99, 1e-3, 0, True),
    ("RN", 1000, 100, 1000, 0.001, 2e-4, 0.5, True),
    ("PDR", 1e5, 300, 3000, 1e-4, 2e-4, 0.5, True),
])
def test_legacy_idl_spectrum(phase, nh, T, Chi, xh, xC, y, tumbling):
    env = dict(nh=nh, T=T, Chi=Chi, xh=xh, xC=xC, y=y, gamma=0, dipole=9.3, line=7)
    nu, sed = SPDUST(env, min_freq=1, max_freq=300, n_freq=500, tumbling=tumbling)
    nu_idl, sed_idl = np.loadtxt(IDL / f"{phase}_emiss.out", comments=";", unpack=True)
    np.testing.assert_allclose(nu, nu_idl, rtol=1e-7, atol=0)
    assert np.isfinite(sed).all() and sed.argmax() == sed_idl.argmax()
    # IDL prints rounded frequencies. Use a peak-scaled floor in the tails
    # and independently constrain fractional errors in the emitting band.
    peak = sed_idl.max()
    assert np.max(np.abs(sed - sed_idl)) < 1e-4 * peak
    bright = sed_idl > 1e-3 * peak
    np.testing.assert_allclose(sed[bright], sed_idl[bright], rtol=6e-4, atol=0)
