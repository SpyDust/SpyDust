"""Finite rotation support, continuous isotropic averaging and discrete orientations."""
import math

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.integrate import quad

from SpyDust.SED import SED, SED_imp
from SpyDust.full_Stokes import full_Stokes_SED, mode1, mode2and3, mode4
from SpyDust.util import cgsconst


@pytest.mark.parametrize("spectrum", [SED, SED_imp])
def test_scalar_frequency_and_disconnected_densities(spectrum):
    rotation = np.arange(1., 11.)
    # A full cubic segment followed by an isolated node and a two-point run.
    values = [1., 2., 3., 4., 0., 5., 0., 2., 2., 0.]
    moments = np.array([[values, np.zeros_like(rotation)]])
    for omega, expected in ((1., 1.), (2., 2.), (5., 0.), (5.5, 0.), (6., 5.),
                            (6.5, 0.), (7., 0.), (8.5, 2.), (10., 0.)):
        args = moments, [0.], rotation, [1.], [None]
        value = spectrum(omega/(2*np.pi), *args) * cgsconst.c**3
        assert_allclose(value, (4./3.) * omega**4 * expected, rtol=2e-14)
        assert_allclose(value, spectrum([omega/(2*np.pi)], *args)[0] * cgsconst.c**3)


@pytest.mark.parametrize("spectrum", [SED, SED_imp])
@pytest.mark.parametrize("shapes", [[0.], [-.3], [0., -.3]])
def test_historical_scalar_return_shape(spectrum, shapes):
    moments = np.ones((len(shapes), 2, 4))
    args = moments, shapes, np.arange(1., 5.), [1.], [None]*len(shapes)
    scalar = spectrum(.35, *args)
    vector = spectrum([.35], *args)
    assert np.shape(scalar) == ((1,) if any(shapes) else ())
    assert np.shape(vector) == (1,)
    assert_allclose(scalar, vector[0])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1.])
@pytest.mark.parametrize("component", [0, 1])
def test_invalid_density_is_not_an_empty_component(bad, component):
    moments = np.ones((1, 2, 3))
    moments[0, component] = bad
    with pytest.raises(ValueError, match="finite and nonnegative"):
        SED(1., moments, [0.], [1., 2., 3.], [1.], [None])


def test_vector_angle_integrands_and_stationary_modes():
    x = np.array([-.25, 0., .75])
    omega = np.array([1., 2., 3.])
    beta = -.3
    def rot(w):
        return np.exp(-w)
    def angles(x):
        return 1. + .2*x
    r = 1.+beta*x
    assert_allclose(mode1.int_integrand(x, omega, beta, angles, rot),
                    angles(x)*(1.-x*x)*rot(omega))
    assert_allclose(mode2and3.int_integrand(x, omega, beta, angles, rot),
                    (angles(x)+angles(-x))*(1.+x)**2*rot(omega/r)/(4.*r))
    expected = [mode4.int_integrand(a, w, beta, angles, rot) for a, w in zip(x, omega)]
    assert_allclose(mode4.int_integrand(x, omega, beta, angles, rot), expected)
    assert expected[1] == 0.


def _bands_reference(omega, beta, bands, angles):
    """Direct integration in Omega for constant density on each supplied band."""
    b = abs(beta)
    norm = quad(angles, -1., 1.)[0]
    total = 0.
    for low, high, density in bands:
        lower = max(low, omega/b)
        if lower < high:
            def fourth(Omega):
                x = omega/(b*Omega)
                return (2./3.) * (1.-x*x) * (angles(x)+angles(-x)) / Omega
            total += density * quad(fourth, lower, high)[0]
        lower, upper = max(low, omega/(1.+b)), min(high, omega/(1.-b))
        if lower < upper:
            def transverse(Omega):
                x = (omega/Omega-1.)/beta
                return (1.+x)**2 * (angles(x)+angles(-x)) / (3.*Omega)
            total += density * quad(transverse, lower, upper)[0]
    return omega**4 * total/(b*norm)


@pytest.mark.parametrize("width", [.02, .0002])
@pytest.mark.parametrize("anisotropic", [False, True])
def test_stokes_integration_resolves_each_narrow_band(width, anisotropic):
    bands = [(1.-width, 1.+width, .5/width), (1.7-width, 1.7+width, 3.)]
    grid = [*bands[0][:2], 1.3, *bands[1][:2]]
    density = [bands[0][2]]*2 + [0.] + [bands[1][2]]*2
    with np.errstate(divide="ignore"):
        log_density = np.log(density)
    model = full_Stokes_SED(rot_freq_grid=np.log(grid), rot_dist_grid=log_density)
    # math.exp also exercises angular functions accepting only scalar values.
    angles = (lambda x: math.exp(.4*x)) if anisotropic else (lambda x: 1.)
    omega = np.array([.008, .1123, .2999, .4, .99, 1.4, 1.7, 2.4])
    result = model.generate_SED(omega, -.3, 1., 1., internal_dist=angles, full_Stokes=False)
    expected = [_bands_reference(w, -.3, bands, angles) for w in omega]
    assert_allclose(result, expected, rtol=2e-11, atol=0.)
    assert_array_equal(model.rot_dist_func([1.3, 1.4]), 0.)


def test_continuous_mode4_and_discrete_orientations():
    beta = -.3
    bands = [(.98, 1.02, 25.), (1.65, 1.75, 2.)]
    rotation = np.array([.98, 1.02, 1.3, 1.65, 1.75])
    moments = np.array([[[25., 25., 0., 2., 2.], np.zeros(5)]])
    omega = np.array([.003, .04, .1123, .21, .299, .48])
    nu = omega / (2.*np.pi)
    x, weights = np.polynomial.legendre.leggauss(6)
    args = moments, [beta], rotation, x, [weights]
    result = SED(nu, *args, isotropic=[True]) * cgsconst.c**3
    expected = [_bands_reference(w, beta, bands, lambda x: 1.) for w in omega]
    assert_allclose(result, expected, rtol=4e-12, atol=0.)
    # The six discrete orientations contain no emitter at the lowest frequency.
    assert SED(nu, *args)[0] == 0.


def test_isotropic_mode4_resolves_steep_density_between_coarse_knots():
    rotation = np.array([1., 2.])
    moments = np.array([[[1., 1e-299], [0., 0.]]])
    beta, omega = -.3, .1
    result = SED(omega/(2*np.pi), moments, [beta], rotation, [0.], [None], isotropic=[True])
    # Linear interpolation of log density in log rotation gives f(Omega)=Omega**s.
    slope = np.log(1e-299) / np.log(2.)
    def moment(power):
        return np.expm1((slope-power)*np.log(2.)) / (slope-power)
    expected = (2./(3.*abs(beta))) * (moment(0.) - (omega/beta)**2*moment(2.))
    assert_allclose(result*cgsconst.c**3/omega**4, expected, rtol=2e-12, atol=0.)


@pytest.mark.parametrize('log_density', [False, True])
def test_positive_background_narrow_peak(log_density):
    from scipy.interpolate import PchipInterpolator
    grid = np.array([1., 1.23436, 1.23446, 1.23466, 1.23476, 3.])
    values = np.array([1e-6, 1e-6, 1., 1., 1e-6, 1e-6])
    y = np.log(values) if log_density else values
    interp = PchipInterpolator(np.log(grid), y)
    model = full_Stokes_SED(rot_freq_grid=np.log(grid), rot_dist_grid=y, log_dist=log_density)
    beta = -.3
    omega = np.array([.1123, 1.05, 1.39])
    def density(spin):
        v = interp(np.log(spin))
        return np.exp(v) if log_density else v
    expected = []
    for w in omega:
        def radiation(spin):
            fourth = w / (abs(beta)*spin)
            x = (w/spin-1.)/beta
            power = 2*(1.-fourth**2) if fourth <= 1. else 0.
            if abs(x) <= 1.:
                power += (1.+x)**2
            return w**4 * density(spin) * power / (3.*abs(beta)*spin)
        bounds = np.unique(np.r_[grid, np.clip([w/abs(beta), w/(1+beta), w/(1-beta)], grid[0], grid[-1])])
        expected.append(sum(quad(radiation, l, r, epsabs=1e-21)[0] for l, r in zip(bounds[:-1], bounds[1:])))
    observed = model.generate_SED(omega, beta, 1., 1., full_Stokes=False)
    assert_allclose(observed, expected, rtol=2e-11, atol=0.)
    if log_density:
        moments = np.array([[values, np.zeros_like(values)]])
        observed = SED(omega/(2*np.pi), moments, [beta], grid, [0.], [None], isotropic=[True])
        assert_allclose(observed*cgsconst.c**3, expected, rtol=2e-11, atol=0.)


def test_batch_frequencies_support_scalar_angular_density():
    model = full_Stokes_SED(rot_freq_grid=[.98, 1.02], rot_dist_grid=[25., 25.],
                            log_freq=False, log_dist=False)
    omega = np.array([.01, .1, .2, .28])
    def scalar(x):
        return 1. + 4.*float(np.mean(np.square(x)))
    observed = model.generate_SED(omega, -.3, 1., 1., internal_dist=scalar, full_Stokes=False)
    expected = model.generate_SED(omega, -.3, 1., 1., internal_dist=lambda x: 1.+4.*x**2, full_Stokes=False)
    assert_allclose(observed, expected, rtol=1e-12, atol=0.)


def test_log_density_underflow_inside_single_cell():
    # A normalized power-law PDF whose support is concentrated at the first knot.
    slope = 1e6
    grid = np.log([1., 2.])
    model = full_Stokes_SED(rot_freq_grid=grid, rot_dist_grid=np.log(slope-1.) - slope*grid)
    omega, beta = .1123, -.3
    actual = model.generate_SED([omega], beta, 1., 1., full_Stokes=False)
    expected = omega**4 * (2./(3*abs(beta))) * (slope-1.) * (1./slope - (omega/beta)**2/(slope+2.))
    assert_allclose(actual, expected, rtol=1e-12, atol=0.)
