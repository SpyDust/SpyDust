"""Physical rotation scales for aligned and tumbling axisymmetric grains."""
import numpy as np
import pytest

from SpyDust import AngMomDist as angular
from SpyDust import collisions, plasmadrag
from SpyDust.Grain import Inertia_ref, grainparams
from SpyDust.SED import SED
from SpyDust.util import cgsconst, makelogtab


def principal_moments(a, beta):
    ref = Inertia_ref(a, beta)
    axial = ref / (1 + beta)
    return ref, axial, max(ref, axial)


@pytest.mark.parametrize("beta", [-.3, 0., .25, .4])
def test_neutral_gas_thermal_spin_and_aligned_radiation(beta, monkeypatch):
    # At Tev=T with no other torques the real aligned neutral collision
    # coefficients satisfy detailed balance and L has a Maxwell distribution.
    temperature = 100.
    env = dict(T=temperature, Tev=temperature, nh=30., xh=0., xC=0., y=0.,
               Chi=0., gamma=0.)
    a = 1e-7
    ref, _, largest = principal_moments(a, beta)
    scale = np.sqrt(cgsconst.k * temperature * largest) / ref
    lo, hi, n = 1e-4 * scale, 12 * scale, 1200
    monkeypatch.setattr(angular, 'FGIR_averaged', lambda *a, **kw: {'FIR': 0., 'GIR': 0.})
    charges = np.array([[0.], [1.]])
    kwargs = dict(tumbling=False, omega_min=lo, omega_max=hi, Nomega=n)
    dipole = np.zeros(1)
    grid = makelogtab(lo, hi, n)
    adaptive = np.exp(angular.log_f_rot(env, a, beta, charges, dipole, dipole, **kwargs)[0])
    fixed_grid, fixed = angular.f_rot_old(env, a, beta, charges, dipole, dipole, **kwargs)
    np.testing.assert_allclose(fixed_grid, grid, rtol=3e-16)
    expected = np.sqrt(2/np.pi) * (grid/scale)**2 * np.exp(-0.5*(grid/scale)**2) / scale
    bright = expected > 0.01 * expected.max()
    for density in (adaptive, fixed[0]):
        np.testing.assert_allclose(density[bright], expected[bright], rtol=5e-4, atol=0.)
        rms = np.sqrt(np.sum(density * grid**3) / np.sum(density * grid))
        np.testing.assert_allclose(rms, np.sqrt(3.) * scale, rtol=4e-5)

    # Aligned rotation radiates at L/Imax. Select a fully transverse test
    # dipole in each geometry, then compare the observed (nu) spectrum.
    direction = float(beta <= 0)
    ip = adaptive * direction
    op = adaptive * (1 - direction)
    observed = np.linspace(.6, 4., 40) * np.sqrt(cgsconst.k * temperature / largest)
    sed = SED(observed/(2*np.pi), np.array([[ip, op]]), [beta], grid,
              [direction], [[1.]])
    shape = observed**6 * np.exp(-largest * observed**2 / (2*cgsconst.k*temperature))
    np.testing.assert_allclose(sed / sed.max(), shape / shape.max(), rtol=5e-4)


@pytest.mark.parametrize("beta", [-.3, 0., .4])
def test_ed_timescale_against_angular_momentum_torque(beta):
    # Test the transformation to the solver timescale against the torque at
    # fixed L, integrating the isotropic case without using its closed form.
    a, temperature = 4e-8, 100.
    ref, _, largest = principal_moments(a, beta)
    L = np.sqrt(largest * cgsconst.k * temperature)
    points, weights = np.polynomial.legendre.leggauss(10)
    for inplane in (0., .6, 1.):
        mu = 3e-18
        ip = mu * inplane
        op = mu * np.sqrt(1 - inplane**2)
        torques = angular.dissipation_vec_ed(L, np.arccos(points), beta, mu, inplane, ref)
        mean = np.dot(weights/2., torques)
        np.testing.assert_allclose(angular.dissipation_rate_ed_theta_avrg(L, beta, mu, inplane, ref),
                                   mean, rtol=2e-14, atol=0.)
        for tumbling in (False, True):
            theta = np.pi/2 if beta > 0 else 0.
            torque = mean if tumbling and beta != 0 else angular.dissipation_vec_ed(
                L, theta, beta, mu, inplane, ref)
            expected = -3 * largest * cgsconst.k * temperature * torque / L**3
            rate = angular.tau_ed_inv(temperature, a, beta, ip, op, tumbling=tumbling)
            np.testing.assert_allclose(rate, expected, rtol=2e-14, atol=0.)


@pytest.mark.parametrize("beta", [-.3, 0., .4])
@pytest.mark.parametrize("tumbling", [False, True])
def test_plasma_frequencies_and_dipole_projections(beta, tumbling, monkeypatch):
    # Impose a known frequency-dependent field spectrum. Average the two
    # circular polarizations independently in the angular momentum frame.
    def electric_power(w):
        return 0.5 + w + w**2

    monkeypatch.setattr(plasmadrag, 'Gp_per_mu2_averaged',
                        lambda temp, xh, xC, a, beta, fZ, w, a2: electric_power(w))
    env = dict(T=100., xh=.001, xC=.0003)
    a = 4e-8
    ref, _, largest = principal_moments(a, beta)
    omega = np.array([.1, .7, 1.8, 3.])
    # Columns isolate the body-equatorial and body-axial dipoles.
    rates = angular._solver_plasma(env, a, beta, None, omega,
                                   np.array([1., 0.]), np.array([0., 1.]),
                                   tumbling, grainparams.a2)
    if tumbling and beta != 0:
        x, weights = np.polynomial.legendre.leggauss(12)
        weights /= 2
        omega_ref = omega[:, None] * largest / ref
        plus = ((1+x)/2)**2 * electric_power(omega_ref*(1+beta*x))
        minus = ((1-x)/2)**2 * electric_power(omega_ref*(1-beta*x))
        g_ip = (plus + minus) @ weights
        g_op = ((1 - x**2) * electric_power(omega_ref)) @ weights
        cross = ((plus - minus) * x) @ weights
        expected_g = np.column_stack((g_ip, g_op))
        expected_f = (largest / ref) * np.column_stack((g_ip + beta*cross, g_op))
        # Production uses the historical 21-midpoint angular quadrature.
        tolerance = 4e-3
    else:
        transverse = np.array([.5, 1.]) if beta > 0 else np.array([1., 0.])
        expected_g = electric_power(omega)[:, None] * transverse
        expected_f = expected_g
        tolerance = 2e-15
    np.testing.assert_allclose(rates['Gp'], expected_g, rtol=tolerance, atol=0.)
    np.testing.assert_allclose(rates['Fp'], expected_f, rtol=tolerance, atol=0.)


def test_legacy_plasma_option_on_aligned_prolate(monkeypatch):
    # The comparison prescription accepts a dipole transverse to its spin
    # axis. Both body components contribute when the physical spin is equatorial.
    env = dict(T=100., xh=.001, xC=.0003)
    omega = np.array([1., 2., 3.])
    monkeypatch.setattr(angular.spd_plasmadrag, 'Gp_sphere_per_mu2_averaged',
                        lambda env, a, fZ, w: 1 + w)
    rates = angular._solver_plasma(env, 4e-8, .4, None, omega,
                                   np.array([2., 0.]), np.array([0., 3.]),
                                   False, grainparams.a2, use_spdust=True)
    expected = (1 + omega[:, None]) * np.array([2., 9.])
    for value in rates.values():
        np.testing.assert_allclose(value, expected, rtol=1e-15)


def test_zero_dipole_ion_collision_limits():
    from SpyDust.SPDUST_as_is import collisions as legacy

    positive = np.array([.5, 1.5, 5.])
    mixed = np.r_[positive[:1], 0., positive[1:]]
    with np.errstate(divide='raise', invalid='raise'):
        # AHD09 Eqs. (106–107) for neutral grains, including phi=0.
        for phi in (0., 1.5, 18.):
            limits = (1 + np.sqrt(np.pi)*phi/2,
                      1 + 3*np.sqrt(np.pi)*phi/4 + phi**2/2)
            for modern, archived, limit in zip((collisions.h1, collisions.h2),
                                               (legacy.h1, legacy.h2), limits):
                assert np.isscalar(modern(phi, 0.))
                assert modern(phi, 0.) == pytest.approx(limit)
                values = modern(phi, mixed)
                assert values[1] == pytest.approx(limit)
                np.testing.assert_allclose(values[[0, 2, 3]], archived(phi, positive), rtol=2e-14)

        # AHD09 Eqs. (90–91) for attractive and repulsive Coulomb potentials.
        for psi in (-2., 0., 2., 610.):
            limits = ((1-psi, 1-psi+psi**2/2) if psi < 0
                      else (np.exp(-psi), np.exp(-psi)))
            scalar = collisions.g1g2(psi, np.array(0.))
            values = collisions.g1g2(psi, mixed)
            previous = legacy.g1g2(psi, positive)
            for single, row, old, limit in zip(scalar, values, previous, limits):
                np.testing.assert_allclose(single, [limit], rtol=1e-15, atol=0.)
                np.testing.assert_allclose(row[1], limit, rtol=1e-15, atol=0.)
                if psi <= 600:
                    np.testing.assert_allclose(row[[0, 2, 3]], old, rtol=2e-15, atol=0.)


def test_zero_dipole_with_ionized_gas_and_mixed_grain_charges():
    env = dict(T=100., xh=.01, xC=.001)
    charges = np.array([[0., -1., 1.], [.4, .35, .25]])
    args = env, 1.e-7, .25, 80.
    with np.errstate(divide='raise', invalid='raise'):
        zero = collisions.FGi_averaged(*args, 0, charges)
        mixed = collisions.FGi_averaged(*args, np.array([2e-18, 0.]), charges)
    for coefficient in ('Fi', 'Gi'):
        assert np.isfinite(zero[coefficient]).all() and zero[coefficient][0] > 0
        np.testing.assert_array_equal(mixed[coefficient][1:], zero[coefficient])


def test_collision_limits_at_small_dipole_and_large_repulsion():
    for phi in (0., 1., 22.):
        mu = np.array([1e-18, 1e-12])
        np.testing.assert_allclose(collisions.h1(phi, mu), 1 + np.sqrt(np.pi)*phi/2, rtol=1e-14)
        np.testing.assert_allclose(collisions.h2(phi, mu), 1 + 3*np.sqrt(np.pi)*phi/4 + phi**2/2,
                                   rtol=1e-14)
    for rate in collisions.g1g2(610., [609., 610.]):
        np.testing.assert_allclose(rate, [np.exp(-1.)/1218., 1./1220.], rtol=2e-15, atol=0.)
    for rate in collisions.g1g2(0., np.array([1e-18])):
        np.testing.assert_allclose(rate, 1., rtol=1e-15)


def test_charge_average_uses_charge_labels():
    from SpyDust import infrared
    from SpyDust.SPDUST_as_is import infrared as legacy_ir, plasmadrag as legacy_plasma
    env = dict(T=100., nh=30., Chi=1., xh=.01, xC=.001)
    a, beta, w = 6e-8, -.2, np.array([1e10, 1e11])
    def rates(fz):
        return (plasmadrag.Gp_per_mu2_averaged(env['T'], env['xh'], env['xC'], a, beta, fz, w),
                legacy_plasma.Gp_sphere_per_mu2_averaged(env, a, fz, w),
                list(infrared.FGIR_averaged(env, a, beta, fz).values()),
                list(legacy_ir.FGIR_averaged(env, a, fz).values()))
    for neutral in (0., .3):
        charges = np.array([[0., 1., -1.], [neutral, .1, .9-neutral]])
        permuted = charges[:, [2, 1, 0]] if neutral else charges[:, [2, 1]]
        for actual, expected in zip(rates(permuted), rates(charges)):
            np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=0.)


def test_charge_distribution_truncation_keeps_one_neutral_state():
    from SpyDust import charge_dist as modern
    from SpyDust.SPDUST_as_is import charge_dist as legacy
    a = 2.8e-8
    env = dict(nh=30., T=100., Chi=31.6228, xh=1.2e-3, xC=3.e-4)
    for model, shape in ((modern, (-.3,)), (legacy, ())):
        labels, prob = model.charge_dist(env, a, *shape)
        assert list(labels).count(0.) == 1
        assert len(np.unique(labels)) == len(labels)
        assert prob.sum() == pytest.approx(1.)
        # Detailed balance between the neutral and singly positive charges.
        ion = model.J_ion_aux(env['nh'], env['T'], env['xh'], env['xC'], 0, a, *shape)
        electron = model.J_electron_aux(env['nh'], env['T'], env['xh'], env['xC'], 1, a, *shape)
        photo = env['Chi']*model.JPEisrf(a)[0][0]
        np.testing.assert_allclose(prob[labels == 1]/prob[labels == 0], (ion+photo)/electron)
