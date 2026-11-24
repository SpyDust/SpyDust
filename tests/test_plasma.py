"""Regression checks for the plasma drag and photoelectric excitation rates."""
import unittest
from unittest.mock import patch

import numpy as np

from SpyDust import H2_photoemission as photo
from SpyDust import plasmadrag as plasma
from SpyDust.SPDUST_as_is import plasmadrag as legacy


class PlasmaRatesTest(unittest.TestCase):
    env = dict(nh=30, T=100., Chi=1., xh=1.2e-3, xC=3e-4)
    a = 4e-8
    charges = np.array([[0, -1, 1], [.7, .2, .1]])

    def reference_lookup(self, omega, beta):
        return plasma.Gp_per_mu2_averaged(
            self.env['T'], self.env['xh'], self.env['xC'], self.a,
            beta, self.charges, np.atleast_1d(omega),
        )

    def test_tumbling_quadrature(self):
        # The two dipoles isolate the in-plane and out-of-plane terms.
        omega = np.geomspace(1e9, 3e11, 35)
        dipole = 1e-18
        cos_theta = -1 + (np.arange(21) + .5) * 2 / 21
        for beta in (-.49, -.3, .1):
            with self.subTest(beta=beta):
                rates = plasma.FGp_averaged(
                    self.env, self.a, beta, self.charges, omega,
                    [dipole, 0.], [0., dipole],
                )
                expected_g = []
                expected_f = []
                for w in omega:
                    # Eqs. 43 and 46 of the SpyDust paper, evaluated one w at a time.
                    omega_ref = w / (1 + beta)
                    minus = self.reference_lookup(omega_ref * (1 - beta*cos_theta), beta)
                    plus = self.reference_lookup(omega_ref * (1 + beta*cos_theta), beta)
                    g_ip = np.mean((1 - cos_theta)**2 * minus / 2)
                    g_op = (2/3) * self.reference_lookup(omega_ref, beta)[0]
                    drift = np.mean(cos_theta * ((1 + cos_theta)**2 * plus
                                                 - (1 - cos_theta)**2 * minus) / 4)
                    expected_g.append([g_ip, g_op])
                    expected_f.append([(g_ip + beta * drift)/(1 + beta), g_op/(1 + beta)])
                np.testing.assert_allclose(rates['Gp'], np.array(expected_g)*dipole**2, rtol=3e-14, atol=0)
                np.testing.assert_allclose(rates['Fp'], np.array(expected_f)*dipole**2, rtol=3e-14, atol=0)

    def test_thin_disk_frequency_limit(self):
        # Hold the lookup fixed to isolate Eq. (45) from the singular
        # conversion between volume-equivalent radius and a zero-thickness disk.
        omega = np.array([.2, 1., 3.])
        def linear_lookup(temp, xh, xC, a, beta, fZ, w, a2):
            return w
        with patch.object(plasma, 'Gp_per_mu2_averaged', side_effect=linear_lookup):
            out = plasma.FGp_averaged(self.env, self.a, -.5, self.charges, omega, [0.], [1.])
        np.testing.assert_allclose(out['Gp'][:, 0], 2/3 * (2*omega))
        np.testing.assert_allclose(out['Fp'], 2*out['Gp'])

    def test_rank_local_kernels_do_not_enter_mpi_collectives(self):
        omega = np.array([.2, 1.])
        psi = np.array([-1., 1.])
        with patch.object(legacy, 'parallel_map', side_effect=AssertionError("nested MPI")):
            self.assertEqual(legacy.little_gp_charged_interpol(psi, omega).shape, (2, 2))
            self.assertGreater(legacy.little_gp_neutral(2., 1.), 0)
        with patch.object(plasma, 'parallel_map', side_effect=AssertionError("nested MPI")):
            self.assertGreater(plasma.little_gp_neutral(2., 1.), 0)

    def test_table_interpolation_preserves_legacy_values(self):
        # Signs, clamped table edges and non-contiguous frequency inputs.
        omega = np.geomspace(1e-12, 1e8, 52)[::2]
        psi = np.array([-1e8, -2.4, -1e-7, 1e-8, .23, 1e8])
        np.testing.assert_allclose(plasma.little_gp_charged_interpol(psi, omega),
                                   legacy.little_gp_charged_interpol(psi, omega), rtol=4e-14, atol=0)
        for phi in (1e-4, 1.2, 1e5):
            np.testing.assert_allclose(plasma.little_gp_neutral_interpol(phi, omega),
                                       legacy.little_gp_neutral_interpol(phi, omega), rtol=4e-14, atol=0)

    def test_photoelectric_rates_scale_with_radiation(self):
        for charge in (-1, 0, 1):
            ref = photo.FGpeZ(self.env, self.a, -.3, charge)
            for chi in (0., .1, 10.):
                out = photo.FGpeZ(dict(self.env, Chi=chi), self.a, -.3, charge)
                for rate in ('Fpe', 'Gpe'):
                    np.testing.assert_allclose(out[rate], chi * ref[rate], rtol=2e-14, atol=0)

    def test_photodetachment_heating_units(self):
        # Remove valence emission. At Z=-1 there is no electrostatic correction.
        beta = -.3
        z = -1
        threshold = photo.hnu_pdt(z, self.a)
        photons = photo.makelogtab(threshold, 13.6, 500)
        dlognu = photo.DX_over_X(threshold, 13.6, 500)
        photon_flux = photo.c * photo.nu_uisrf(photons) / (photons * photo.eV)
        rate_density = photon_flux * photo.sigma_pdt(photons, z, self.a)
        energy_erg = (photons - threshold + photo.E_min(z, self.a)) * photo.eV
        heating = np.sum(rate_density * energy_erg) * dlognu
        kT = photo.k * self.env['T']
        area_scale = photo.asurf(self.a, beta)**2
        expected = heating * photo.me / (4 * self.env['nh'] * np.sqrt(8*np.pi*photo.mp*kT) * area_scale * kT)
        with patch.object(photo, 'Qabs', return_value=np.zeros_like(photons)):
            actual = photo.FGpeZ(self.env, self.a, beta, z)
        np.testing.assert_allclose(actual['Gpe'], expected, rtol=2e-14, atol=0)
