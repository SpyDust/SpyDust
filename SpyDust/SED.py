from .util import cgsconst, DX_over_X, maketab, makelogtab 
from .mpiutil import *
from .Grain import grainparams, rms_dipole
from .charge_dist import charge_dist
from .AngMomDist import log_f_rot
import numpy as np
from .full_Stokes import _TabulatedRotationPDF, mode2and3, mode4
from .util import homogeneous_dist


pi = np.pi
c = cgsconst.c
h = cgsconst.h
k = cgsconst.k
eV = cgsconst.eV
me = cgsconst.me
q = cgsconst.q
mp = cgsconst.mp

a2_default = grainparams.a2


def _check_dipole_sampling(ip, Ndipole):
    if not isinstance(Ndipole, (int, np.integer)) or Ndipole < 1:
        raise ValueError("Ndipole must be a positive integer.")
    if not 0. <= ip <= 1.:
        raise ValueError("inplane must be in [0, 1].")


def mu2_f_cond(env, a, beta, fZ, mu_rms, ip, Ndipole, tumbling=True, omega_min=1e7, omega_max=1e12, Nomega=1000, spdust_plasma=False, a2=a2_default):
    """Dipole averages [<mu_ip² f>, <mu_op² f>] on the rotation grid."""
    _check_dipole_sampling(ip, Ndipole)

    if Ndipole == 1:
        mu_ip = np.array([np.sqrt(ip) * mu_rms])
        mu_op = np.array([np.sqrt(1. - ip) * mu_rms])
        weights = np.ones(1)
    else:
        # Half the bins sample the low-moment tail on a logarithmic grid.
        xmin, xmed, xmax = 0.005, 0.5, 5.
        n_log = Ndipole // 2
        n_linear = Ndipole - n_log
        low = makelogtab(xmin, xmed, n_log)
        x = np.concatenate([low, maketab(xmed, xmax, n_linear)])
        dx = np.concatenate([DX_over_X(xmin, xmed, n_log) * low,
                             np.full(n_linear, (xmax - xmed) / n_linear)])

        if a >= a2:
            # The large-grain model first averages over dipole direction.
            mu_ip = np.sqrt(2./3.) * mu_rms * x
            mu_op = np.sqrt(1./3.) * mu_rms * x
            weights = x**2 * np.exp(-1.5*x**2) * dx
        else:
            # Independent Rayleigh (in-plane) and half-normal (axial) moments.
            # At either endpoint the absent component is a delta mass at zero.
            x_ip = x if ip > 0. else np.zeros(1)
            x_op = x if ip < 1. else np.zeros(1)
            w_ip = x * np.exp(-x**2) * dx if ip > 0. else np.ones(1)
            w_op = np.exp(-0.5*x**2) * dx if ip < 1. else np.ones(1)
            mu_ip = np.sqrt(ip) * mu_rms * np.repeat(x_ip, len(x_op))
            mu_op = np.sqrt(1. - ip) * mu_rms * np.tile(x_op, len(x_ip))
            weights = np.outer(w_ip, w_op).ravel()
        weights /= weights.sum()

    log_f = log_f_rot(env, a, beta, fZ, mu_ip, mu_op, tumbling=tumbling,
                      omega_min=omega_min, omega_max=omega_max, Nomega=Nomega,
                      use_spdust_plasma=spdust_plasma, a2=a2)
    f = np.exp(log_f)
    return np.array([np.sum((weights * moment**2)[:, None] * f, axis=0)
                     for moment in (mu_ip, mu_op)])


def mu2_f(env, a_tab, beta_tab, f_a_beta, mole_dipole, ip, Ndipole, tumbling=True, parallel=True, contract_a=True, omega_min=1e7, omega_max=1e15, Nomega=1000, spdust_plasma=False, a2=a2_default):
    '''
    Calculate mu^2 f(omega, a, beta, mu) and marginalize over "mu" and "a" (if contract_a is True).
    '''
    _check_dipole_sampling(ip, Ndipole)
    num_a = np.size(a_tab)
    num_beta = np.size(beta_tab)
    
    def aux_func(a_ind):
        result = []
        a = a_tab[a_ind]
        for beta_ind in range(num_beta):
            beta = beta_tab[beta_ind]
            if f_a_beta[a_ind, beta_ind] == 0.:
                result.append(np.zeros((2, Nomega)))
                continue
            fZ = charge_dist(env, a, beta, a2=a2)
            Z2 = np.sum(fZ[0, :]**2 * fZ[1, :])
            mu_rms = rms_dipole(a, beta, Z2, mole_dipole, a2=a2)
            mu2_fa_aux  = mu2_f_cond(env, a, beta, fZ, mu_rms, ip, Ndipole, tumbling=tumbling, omega_min=omega_min, omega_max=omega_max, Nomega=Nomega, spdust_plasma=spdust_plasma, a2=a2)
            result.append(mu2_fa_aux)
        return np.array(result)
    
    if parallel:
        aux = np.array(parallel_map(aux_func, np.arange(num_a)))
    else:
        aux=[]
        for ai in np.arange(num_a):
            aux.append(aux_func(ai))
        aux = np.array(aux)
    result = aux * f_a_beta[:,:,np.newaxis,np.newaxis]
    if contract_a:
        # Only when f_a_beta incorporate the differential element da 
        # we can simply sum over a as the integral over a.
        # This is the case when the keyword "a_weighted" is set to True in the function size_and_shape_dist. (See main.Grain)
        return np.sum(result, axis=0) # shape (Nbeta, 2, Nomega)
    else:
        return result # shape (Na, Nbeta, 2, Nomega)
    
def _rotation_pdf(log_rotation, values, cutoff):
    """Construct the same finite-support log-log PCHIP density used by Stokes."""
    if not np.isfinite(values).all() or np.any(values < 0.):
        raise ValueError("Dipole-weighted rotation densities must be finite and nonnegative.")
    active = values > cutoff
    if not np.any(active):
        return None
    log_values = np.full(values.shape, -np.inf)
    log_values[active] = np.log(values[active])
    return _TabulatedRotationPDF(log_rotation, log_values, True, True)


def SED(nu_tab, mu2_f_arr, beta_tab, angular_Omega_tab, cos_theta_list, cos_theta_weights,
        *, isotropic=None):
    """Sum the four radiation modes over shape and internal alignment.

    Frequencies nu_tab are in Hz and angular_Omega_tab is in rad/s.
    mu2_f_arr has shape (Nbeta, 2, Nomega) for the in-plane and axial
    dipole powers. Each row of cos_theta_weights gives probability masses
    on cos_theta_list for that beta, or None for equally weighted nodes.
    Set isotropic[i] for continuous isotropic orientations. The continuous
    integrals use rotation frequency to resolve narrow angular features.
    """
    cosines = np.asarray(cos_theta_list, dtype=float)
    if cosines.ndim != 1 or not np.all(np.abs(cosines) <= 1.):
        raise ValueError("cos_theta_list must contain direction cosines in [-1, 1].")
    angular_Omega_tab = np.asarray(angular_Omega_tab, dtype=float)
    if (angular_Omega_tab.ndim != 1 or not angular_Omega_tab.size or
            not np.isfinite(angular_Omega_tab).all() or np.any(angular_Omega_tab <= 0.) or
            np.any(np.diff(angular_Omega_tab) <= 0.)):
        raise ValueError("Rotation frequencies must be positive, finite and increasing.")
    log_rotation = np.log(angular_Omega_tab)
    nu_tab = np.asarray(nu_tab, dtype=float)
    omega = 2 * pi * np.atleast_1d(nu_tab)
    if omega.ndim != 1 or np.any(omega < 0.) or not np.isfinite(omega).all():
        raise ValueError("Observing frequencies must be finite and nonnegative.")
    if isotropic is None:
        isotropic = np.zeros(len(beta_tab), dtype=bool)
    isotropic = np.asarray(isotropic, dtype=bool)
    if isotropic.shape != (len(beta_tab),):
        raise ValueError("Expected one isotropic alignment flag per beta.")
    mu2_f_arr = np.asarray(mu2_f_arr)
    positive_freq = omega > 0.
    log_omega = np.log(omega[positive_freq])

    if len(cos_theta_weights) != len(beta_tab):
        raise ValueError("Expected one alignment weight row per beta.")
    orientations = []
    for weights in cos_theta_weights:
        weights = np.ones_like(cosines) if weights is None else np.asarray(weights, dtype=float)
        if (weights.shape != cosines.shape or not np.all(np.isfinite(weights))
                or np.any(weights < 0.) or not np.any(weights > 0.)):
            raise ValueError("Alignment weights must be nonnegative with positive total mass.")
        active = weights > 0.
        orientations.append((cosines[active], weights[active] / weights[active].sum()))

    def emiss_beta(index):
        beta = beta_tab[index]
        x, w = orientations[index]
        ip = _rotation_pdf(log_rotation, mu2_f_arr[index, 0], 1e-300)
        op = _rotation_pdf(log_rotation, mu2_f_arr[index, 1], 0.)
        emiss = np.zeros_like(log_omega)
        if isotropic[index]:
            frequencies = omega[positive_freq]
            # <1-x²> = <(1+x)²>/2 = 2/3 for uniform x = cos(theta).
            if op is not None:
                emiss += (8./9.) * op(frequencies)
            if ip is not None:
                if beta == 0.:
                    emiss += (8./9.) * ip(frequencies)
                else:
                    # Uniform orientation gives density 1/2 in x. The external
                    # factors for the transverse and axial harmonics are 4/3 and 2/3.
                    emiss += (2./3.) * ip.angle_integral(mode2and3, frequencies, beta, homogeneous_dist, 1)
                    emiss += (1./3.) * ip.angle_integral(mode4, frequencies, beta, homogeneous_dist, 1)
            return emiss
        # Mode 1, after averaging over external orientations (paper Eq. 74).
        if op is not None:
            emiss += 4./3. * np.sum(w * (1. - x**2)) * op(omega[positive_freq])
        if ip is not None:
            # Modes 2--4. A zero mapping factor emits only at zero frequency.
            # At beta=0 the two coincident transverse modes add at the same omega.
            modes = ((1. + beta*x, (1. + x)**2 / 3.),
                     (1. - beta*x, (1. - x)**2 / 3.),
                     (beta*x, 2./3. * (1. - x**2)))
            for mapping, power in modes:
                factors = np.abs(mapping)
                use = (factors > 0.) & (power > 0.)
                if np.any(use):
                    sampled = ip(omega[None, positive_freq] / factors[use, None])
                    emiss += np.sum((w[use]*power[use]/factors[use])[:, None] * sampled, axis=0)
        return emiss

    shapes = parallel_map(emiss_beta, np.arange(len(beta_tab)))
    spectrum = np.zeros_like(omega)
    spectrum[positive_freq] = np.sum(shapes, axis=0) * omega[positive_freq]**4 / c**3
    # The historical spherical shortcut returns a scalar. With nonspherical
    # grains the angular average retains its length-one frequency axis.
    return spectrum[0] if nu_tab.ndim == 0 and np.all(np.asarray(beta_tab) == 0.) else spectrum


def SED_imp(nu_tab, mu2_f_arr, beta_tab, angular_Omega_tab, cos_theta_list, cos_theta_weights,
            *, isotropic=None):
    """Compatibility entry point for the vectorized SED calculation."""
    return SED(nu_tab, mu2_f_arr, beta_tab, angular_Omega_tab, cos_theta_list, cos_theta_weights,
               isotropic=isotropic)
