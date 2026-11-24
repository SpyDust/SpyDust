"""Regression tests for the size/shape quadrature and angular emission modes."""
import unittest

import numpy as np

from SpyDust.Grain import eval_beta, grain_distribution, grainparams
from SpyDust.SED import SED, SED_imp
from SpyDust.SpyDust import SpyDust, SpyDust_imp, SpyDust_given_grain_size_shape
from SpyDust.util import cgsconst


CNM = dict(nh=30., T=100., Chi=1., xh=1.2e-3, xC=3e-4, y=0.,
           gamma=0., dipole=9.3, line=7)


class TestShapeDistribution(unittest.TestCase):
    def test_exact_fixed_shapes_and_size_weights(self):
        grains = grain_distribution()
        weights = grains.shape_and_size_dist(6, fixed_thickness=True, Nbeta=5)
        np.testing.assert_array_equal(np.count_nonzero(weights, axis=1), 1)
        expected = [eval_beta(a) for a in grains.a_tab]
        np.testing.assert_array_equal(grains.beta_tab[weights.argmax(axis=1)], expected)
        np.testing.assert_array_equal(weights.sum(axis=1), grains.a_dist)

    def test_resolved_broadening(self):
        grains = grain_distribution()
        widths = []
        for n in (grain_distribution.default_Nbeta, 301):
            weights = grains.shape_and_size_dist(6, Nbeta=n)
            np.testing.assert_allclose(weights.sum(axis=1), grains.a_dist, rtol=1e-14)
            self.assertGreater(np.count_nonzero(weights[-1]), 2)
            mean = np.average(grains.beta_tab, weights=weights[-1])
            self.assertLess(abs(mean), 1e-4)
            widths.append(np.sqrt(np.average((grains.beta_tab-mean)**2, weights=weights[-1])))
        self.assertGreater(widths[0], 0.015)
        self.assertAlmostEqual(widths[0], widths[1], delta=0.0001)


class TestScalarRadiation(unittest.TestCase):
    # Constant input spectra isolate angular powers and frequency Jacobians.
    def spectrum(self, beta, cosines, weights, ip, op, spectrum=SED):
        omega = np.array([1., 2.])
        rotation = np.geomspace(0.1, 50., 30)
        moments = np.array([[np.full_like(rotation, ip), np.full_like(rotation, op)]])
        power = spectrum(omega/(2*np.pi), moments, np.array([beta]), rotation,
                         np.asarray(cosines), [weights])
        return power / (omega**4 / cgsconst.c**3)

    def test_tabulated_endpoints_and_larmor_normalization(self):
        rotation = np.geomspace(0.1, 10., 200)
        moments = np.array([[np.ones_like(rotation), np.zeros_like(rotation)]])
        power = SED(rotation/(2*np.pi), moments, [0.], rotation, [1.], [[1.]])
        # One transverse dipole; dP/dnu/dOmega = (2/3 w^4/c^3) * 2pi / 4pi.
        conventional = rotation**4 / (3 * cgsconst.c**3)
        np.testing.assert_allclose(power / 4., conventional, rtol=2e-14, atol=0)

    def test_pure_axial_mode(self):
        for x in (-1., 0., 0.4, 1.):
            with self.subTest(x=x):
                np.testing.assert_allclose(self.spectrum(-0.3, [x], [1.], 0., 1.),
                                           4./3.*(1.-x*x), atol=1e-14)
        x, w = np.polynomial.legendre.leggauss(5)
        np.testing.assert_allclose(self.spectrum(-0.3, x, w, 0., 1.), 8./9.)

    def test_sphere_and_aligned_transverse_harmonics(self):
        for x in (-1., 0., 0.4, 1.):
            with self.subTest(x=x):
                # At beta=0 and isotropic dipole direction emission is axis-independent.
                np.testing.assert_allclose(self.spectrum(0., [x], [1.], 2./3., 1./3.), 8./9.)
                np.testing.assert_allclose(self.spectrum(0., [x], [1.], 1., 0.),
                                           (2./3.)*(1.+x*x))
        np.testing.assert_allclose(self.spectrum(-0.3, [1.], [1.], 1., 0.), 4./(3.*0.7))
        np.testing.assert_allclose(self.spectrum(0.1, [0.], [1.], 1., 0., SED_imp), 2./3.)

    def test_single_grain_public_endpoints_and_odd_counts(self):
        for beta, fraction in ((0.1, 0.), (-0.3, 1.), (-0.3, 0.)):
            with self.subTest(beta=beta, fraction=fraction):
                nu, sed = SpyDust_given_grain_size_shape(
                    dict(CNM, inplane=fraction), 4e-8, beta, tumbling=False, Ndipole=3,
                    Ncos_theta=7, N_angular_Omega=80, n_freq=16,
                )
                self.assertTrue(np.all(np.diff(nu) > 0.) and np.isfinite(sed).all())
                if beta < 0. and fraction == 0.:
                    np.testing.assert_array_equal(sed, 0.)
                else:
                    self.assertGreater(sed.max(), 0.)

    def test_population_entry_points(self):
        # Small dipole/angle counts exercise each public route at finite cost.
        opts = dict(Ndipole=1, Ncos_theta=4, n_freq=16, single_beta=True)
        expected = SpyDust(CNM, N_angular_Omega=1000, Nbeta=31, **opts)
        actual = SpyDust_imp(CNM, Nbeta=5, **opts)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.isfinite(actual).all() and actual[1].max() > 0.)
