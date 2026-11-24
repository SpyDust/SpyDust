import sys
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from SpyDust.SED import SED
from SpyDust.SED_fit import fit_sed_ensemble, lognormal_sed_fit_v2, measure_sed_peak_properties
from SpyDust.full_Stokes import full_Stokes_SED, mode4
from SpyDust.mom_exp import MC_moment_expansion
from SpyDust.util import cgsconst


def rotation_pdf(omega):
    return (1.0 + omega)**-4


class TestStokes(unittest.TestCase):
    def test_fixed_angle_and_stationary_harmonics(self):
        omega = np.array([1.0, 2.0])
        model = full_Stokes_SED(rot_dist_func=rotation_pdf)
        ip = 0.7
        mu_sq = 2.3
        for beta, x in [(0.0, 0.5), (-0.3, 0.0), (-0.3, 1.0), (-0.3, -1.0)]:
            with self.subTest(beta=beta, cos_theta=x):
                modes = 4/3 * (1-ip) * (1-x*x) * rotation_pdf(omega)
                for direction in [-1, 1]:
                    ratio = abs(1 + direction*beta*x)
                    modes += ip/3 * (1 + direction*x)**2 * rotation_pdf(omega/ratio) / ratio
                # In these geometries mode 4 has either zero frequency or zero power.
                result = model.generate_SED(omega, beta, ip, mu_sq, impulse=x)
                assert_allclose(result[0], mu_sq * omega**4 * modes, rtol=1e-12)
                assert_allclose(result[1:], 0.0, atol=1e-14)
        self.assertEqual(mode4.int_alignm_part(1.0), 0)
        self.assertEqual(mode4.int_alignm_part(-1.0), 0)

    def test_gridded_rotation_has_finite_support(self):
        rotation = np.geomspace(.1, 10., 81)
        model = full_Stokes_SED(rot_freq_grid=np.log(rotation),
                                rot_dist_grid=-np.log1p(rotation**2))
        def compact_pdf(omega):
            return np.where((omega >= rotation[0]) & (omega <= rotation[-1]),
                            1. / (1. + omega**2), 0.)

        omega = np.array([1., 2.])
        gapped = full_Stokes_SED(rot_freq_grid=[0, 1, 2, 3, 4],
                                  rot_dist_grid=[0, 0, -np.inf, -1, -1])
        assert_allclose(gapped.rot_dist_func(np.exp([0.5, 2., 3.5])), [1., 0., np.exp(-1)])
        reference = full_Stokes_SED(rot_dist_func=compact_pdf)
        # Small |beta cos(theta)| sends mode 4 far outside the tabulated grid.
        for impulse in (1e-8, None):
            result = model.generate_SED(omega, -.3, 2/3, 1., impulse=impulse)
            expected = reference.generate_SED(omega, -.3, 2/3, 1., impulse=impulse)
            assert_allclose(result, expected, rtol=2e-5, atol=1e-12)

    def test_isotropic_stokes_intensity_matches_scalar_spectrum(self):
        omega = np.array([1.0, 1.5, 3.0])
        beta = -0.3
        ip = 0.45
        mu_sq = 2.1
        rotation_grid = np.geomspace(0.01, 1e6, 1600)
        weighted_pdf = mu_sq * np.array([[ip, 1-ip]]).T * rotation_pdf(rotation_grid)
        cos_theta, weights = np.polynomial.legendre.leggauss(80)
        scalar = SED(omega/(2*np.pi), weighted_pdf[None, :, :], np.array([beta]),
                     rotation_grid, cos_theta, [weights])
        result = full_Stokes_SED(rot_dist_func=rotation_pdf).generate_SED(omega, beta, ip, mu_sq)
        assert_allclose(result[0], scalar * cgsconst.c**3, rtol=1e-7)
        assert_allclose(result[1:], 0.0, atol=1e-14)


class TestSpectralFits(unittest.TestCase):
    def test_frequency_order_preserves_peak_and_curve_alignment(self):
        freq = np.geomspace(1, 300, 101)[::-1]
        sed = np.exp(-0.5 * (np.log(freq/30.) / .6)**2)
        peak, _, sigma, _, curve, quality = lognormal_sed_fit_v2(
            freq, sed, brief_return=False, return_fit_curve=True)
        assert_allclose([peak, sigma, quality['r_squared']], [30., .6, 1.])
        use = np.isfinite(curve)
        assert_allclose(curve[use], sed[use], atol=0)

    def test_physical_peak_and_v2_legacy_tuple(self):
        freq = np.arange(5, 101)
        peak, width, height = 30.0, 0.45, 2.5e-17
        sed = height * np.exp(-0.5 * (np.log(freq / peak) / width)**2)
        result = measure_sed_peak_properties(freq, sed)
        self.assertAlmostEqual(result["f_peak"], peak)
        self.assertAlmostEqual(result["fwhm"], 2 * peak * np.sinh(width * np.sqrt(2*np.log(2))))
        assert_allclose(result["amplitude"], height, rtol=1e-9, atol=0)

        with patch.dict(sys.modules, {"tqdm": None}):
            assert_allclose(fit_sed_ensemble(freq, sed), [np.log(peak), width])
        for gamma in [-2.0, 2.0]:
            center = np.log(peak) - gamma * width**2
            scale = height * np.exp(-gamma * (np.log(peak) + center) / 2)
            initial = (scale, gamma, center, width)
            log_peak, fixed_gamma, sigma = lognormal_sed_fit_v2(freq, sed, initial_guess=initial)
            assert_allclose([np.exp(log_peak), fixed_gamma, sigma], [peak, 0, width])
            values = lognormal_sed_fit_v2(freq, sed, initial_guess=initial,
                                          brief_return=False, return_fit_curve=True)
            fitted, fixed_gamma, sigma, params, fitted_curve, quality = values
            assert_allclose(params, [height, 0, np.log(peak), width], atol=0)
            self.assertTrue(quality["gamma_fixed"])
            assert_array_equal(quality["covariance_matrix"][1], np.zeros(4))
            mask = sed >= 1e-3 * height
            assert_allclose(fitted_curve[mask], sed[mask], rtol=1e-10, atol=0)
            self.assertTrue(np.isnan(fitted_curve[~mask]).all())


class TestMomentCache(unittest.TestCase):
    def test_evaluation_refreshes_environment_and_explicit_basis_replays(self):
        pivots = [np.log(4e-8), -0.3, np.log(1e-4)]
        freqs = np.array([15., 30., 50.])
        model = MC_moment_expansion(pivots, order=0, default_freqs=freqs)
        original = model.generate_SED_basis(tumbling=False)
        model.T = 100.
        _, updated = model.evaluate_expansion(pivots)
        _, expected = model.generate_SED(*pivots, freq_coords=freqs, tumbling=False)
        assert_allclose(updated, expected, rtol=1e-12, atol=0.)
        self.assertGreater(np.max(abs(updated-original['zeroth'])), 1e-3*max(expected))
        assert_array_equal(model.evaluate_expansion(pivots, basis=original)[1], original['zeroth'])

    def test_changes_in_model_and_derivative_order(self):
        freqs = np.array([15., 30., 50.])
        model = MC_moment_expansion([0., -0.3, -4.], order=2, default_freqs=freqs)

        def spectrum(a, beta, xc, freq_coords, tumbling=True):
            f = np.asarray(freq_coords)
            return f, (1 + a + beta + xc) * f + model.nh * float(tumbling)

        with patch.object(model, 'generate_SED', side_effect=spectrum) as evaluate:
            basis = model.generate_SED_basis(order=0)
            model.fit_basis_functions(basis['zeroth'], order=0)
            self.assertEqual(evaluate.call_count, 1)
            model.fit_basis_functions(basis['zeroth'], order=0, tumbling=False)
            self.assertFalse(model.last_basis['tumbling'])
            self.assertEqual(evaluate.call_count, 2)
            model.fit_basis_functions(basis['zeroth'], order=1, tumbling=False)
            self.assertEqual(model.last_basis['order'], 1)
            self.assertEqual(evaluate.call_count, 9)
            model.T *= 2
            model.pivots[1] += 0.05
            model.step_sizes *= 1.1
            model.fit_basis_functions(basis['zeroth'], order=0, tumbling=False)
            self.assertEqual(evaluate.call_count, 10)
            assert_array_equal(model.last_basis['pivots'], model.pivots)
            self.assertEqual(model.last_basis['environment']['T'], model.T)
            model.default_freqs[0] = 10
            model.fit_basis_functions(basis['zeroth'], order=0, tumbling=False)
            self.assertEqual(evaluate.call_count, 11)
            assert_array_equal(model.last_basis['freqs'], model.default_freqs)


if __name__ == '__main__':
    unittest.main()
