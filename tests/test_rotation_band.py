"""The normalized spin distribution belongs to the grain and its environment."""
import unittest

import numpy as np

from SpyDust.AngMomDist import aux_int, f_rot_old, log_f_rot
from SpyDust.SpyDust import SpyDust_given_grain_size_shape
from SpyDust.charge_dist import charge_dist
from SpyDust.util import DX_over_X, cgsconst, makelogtab


CNM = dict(nh=30., T=100., Chi=1., xh=1.2e-3, xC=3e-4, y=0.,
           gamma=0., dipole=9.3, line=7)


class RotationBandTests(unittest.TestCase):
    def test_thermal_solution(self):
        # For F=G and no dipole damping, p(omega) is the Maxwell density.
        lo, hi, n = 1.e-4, 12., 1000
        omega = makelogtab(lo, hi, n)
        rates = np.ones((n, 1))
        log_f = aux_int(1., rates, rates, omega, n, 1, 1., np.zeros(1),
                        cgsconst.k, DX_over_X(lo, hi, n))[0]
        expected = np.sqrt(2/np.pi) * omega**2 * np.exp(-omega**2 / 2.)
        bright = expected > 0.01 * expected.max()
        np.testing.assert_allclose(np.exp(log_f[bright]), expected[bright], rtol=3e-4)

    def test_fixed_grid_solver_returns_frequency_and_each_density(self):
        a, beta = 4e-8, -.3
        charges = charge_dist(CNM, a, beta)
        for count in (1, 2, 3):
            dipoles = np.arange(1., count + 1.) * 1e-18
            omega, density = f_rot_old(CNM, a, beta, charges, dipoles, dipoles, Nomega=40)
            np.testing.assert_allclose(omega, makelogtab(1e7, 1e11, 40))
            self.assertEqual(density.shape, (count, 40))
            self.assertTrue(np.all(np.isfinite(density)) and np.all(density >= 0))
            np.testing.assert_allclose((density * omega).sum(axis=1)*DX_over_X(1e7, 1e11, 40), 1.)

    def test_overlapping_observing_bands(self):
        def value(low, high, count, nspin):
            nu, sed = SpyDust_given_grain_size_shape(
                CNM, 1e-7, -0.05, tumbling=False, Ndipole=1,
                min_freq=low, max_freq=high, n_freq=count, N_angular_Omega=nspin,
            )
            # All three observing grids include this exact logarithmic bin.
            i = np.argmin(np.abs(nu-10.2329299228))
            self.assertAlmostEqual(nu[i], 10.2329299228, places=8)
            return sed[i]

        bands = [(1., 100., 100), (10., 100., 50), (10., 1000., 100)]
        low_res = [value(*band, 500) for band in bands]
        high_res = [value(*band, 2000) for band in bands]
        np.testing.assert_allclose(low_res, low_res[0], rtol=3.e-5)
        np.testing.assert_allclose(high_res, high_res[0], rtol=1.e-6)
        np.testing.assert_allclose(low_res, high_res, rtol=2.e-3)

    def test_dipole_distributions_integrate_to_one_and_have_bounded_tails(self):
        # Evaluate each normalized intrinsic density on an independent output
        # grid significantly wider than its support. Include both shape models.
        lo, hi, n = 1.e2, 1.e16, 500
        omega = makelogtab(lo, hi, n)
        mu_ip = np.array([0.001, 1., 10.]) * 1e-18
        mu_op = np.array([0.001, 2., 5.]) * 1e-18
        for a, beta, tumbling in ((4.e-8, -0.3, True), (1.e-7, -0.05, False),
                                  (3.e-7, 0., False)):
            with self.subTest(a=a, beta=beta):
                log_f = log_f_rot(CNM, a, beta, charge_dist(CNM, a, beta),
                                  mu_ip, mu_op, tumbling=tumbling,
                                  omega_min=lo, omega_max=hi, Nomega=n)
                mass = (np.exp(log_f) * omega).sum(axis=1) * DX_over_X(lo, hi, n)
                np.testing.assert_allclose(mass, 1., atol=2.5e-4, rtol=0.)
                np.testing.assert_allclose(log_f[:, 1] - log_f[:, 0],
                                           2. * np.log(omega[1]/omega[0]), atol=1e-12)
                self.assertTrue(np.all(np.isneginf(log_f[:, -1])))
