import numpy as np
from .util import cgsconst, makelogtab, DX_over_X
from .Grain import acx, Inertia_largest, grainparams, inertia_ref_over_largest
from .infrared import FGIR_averaged
from .collisions import Tev_effective, FGn_averaged, FGi_averaged
from .plasmadrag import FGp_averaged
from .H2_photoemission import GH2, FGpe_averaged

# To compare with the old plasma drag calculation in SpDust
from .SPDUST_as_is import plasmadrag as spd_plasmadrag
from scipy.interpolate import PchipInterpolator
from scipy.special import logsumexp


c = cgsconst.c
mp = cgsconst.mp
k = cgsconst.k
pi = np.pi

a2_default = grainparams.a2


# Function to calculate the characteristic damping time through collisions with neutral H atoms
# @njit
def tau_H(temp, nh, a, beta, a2=a2_default):
    """
    Returns the characteristic damping time through collisions with neutral H atoms.
    This corresponds to Eq. (24) in AHD09.

    Parameters:
    - env: Environment parameters (temperature T, hydrogen density nh).
    - a: Grain size.
    - temp:   temperature in K
    - nh:  hydrogen density in cm^-3

    Returns:
    - Damping time tau_H (in seconds).
    """

    acx_val = acx(a, beta, a2=a2)  # characteristic length scale
    Inertia_val = Inertia_largest(a, beta, a2=a2)  # grain's moment of inertia

    # Calculate tau_H using the given equation
    return 1.0 / (nh * mp * np.sqrt(2 * k * temp / (pi * mp)) * 4 * pi * acx_val**4 / (3 * Inertia_val))

# @njit
def dissipation_vec_ed( L_val, theta_b, beta, mu_abs, ip, I_ref):
    '''
    Calculate the angular momentum dissipation rate for the back-reaction of the electric dipole radiation.

    Parameters:
    -----------
    L_val: float
        Magnitude of the angular momentum.
    theta_b: float
        The internal angle between the grain body z-axis and the angular momentum.
    beta: float
        The grain parameter, describing the shape of the grain.
    mu_abs: float
        The absolute value of the dipole moment.
    ip: float
        The in-plane ratio of the dipole moment. ip = mu_perp/mu_abs.
    I_ref: float
        The reference moment of inertia.

    Returns:
    --------
        The angular momentum dissipation rate.
    '''

    mu_parallel_2 = mu_abs**2 * (1 - ip**2)
    mu_perp_2 = mu_abs**2 * ip**2

    aux_perp = 1 + ( 1 + 3 * beta * (2 + beta) ) * np.cos(theta_b)**2 + beta**2 * ( 3 + 2 * beta ) * np.cos(theta_b)**4
    aux_parallel = 2 * np.sin(theta_b)**2 

    result = - (1/3) * ( mu_perp_2 * aux_perp + mu_parallel_2 * aux_parallel ) * (L_val / (I_ref * c)) ** 3
    return result


def _isotropic_ed_inplane(beta):
    # Eq. (38), using <cos²(theta)> = 1/3 and <cos⁴(theta)> = 1/5.
    return (2/5) * beta**3 + (8/5) * beta**2 + 2 * beta + 4/3


# @njit
def dissipation_rate_ed_theta_avrg( L_val, beta, mu_abs, ip, I_ref):
    '''
    Calculate the ed angular momentum dissipation rate, averaging over nutation angle.
    '''

    mu_perp_2 = mu_abs**2 * ip**2
    mu_parallel_2 = mu_abs**2 * (1 - ip**2)

    aux_perp = _isotropic_ed_inplane(beta)
    aux_parallel = 4 / 3 

    result = - (1/3) * ( mu_perp_2 * aux_perp + mu_parallel_2 * aux_parallel ) * (L_val / (I_ref * c)) ** 3
    return result


# Function to calculate the inverse of the characteristic damping time through electric dipole radiation
# @njit
def tau_ed_inv(temp, a, beta, mu_ip, mu_op, tumbling=True, a2=a2_default):
    """
    Returns the inverse of the characteristic damping time through electric dipole radiation.
    Uses the torque in SpyDust Eqs. (37–39), normalized with Imax.

    Parameters:
    - temp: Environment temperature T in K.
    - a: Grain size.
    - mu_ip: Electric dipole moment in-plane.
    - mu_op: Electric dipole moment out-of-plane.
    - tumbling: Average over isotropic nutation angles for non-spherical grains.

    Returns:
    - Inverse of the damping time (in seconds^-1).
    """  

    # Eq. (39) in solver units: -3 Imax kT (dL/dt)/L³. The physical
    # torque of Eq. (37) is proportional to (L/Iref)³.
    inertia = Inertia_largest(a, beta, a2=a2)
    ratio = inertia_ref_over_largest(beta)
    prefactor = k * temp / (inertia**2 * c**3 * ratio**3)

    if tumbling and beta != 0:
        return prefactor * (_isotropic_ed_inplane(beta) * mu_ip**2 + (4/3) * mu_op**2)
    if beta <= 0:
        # Rotation around the body symmetry axis.
        return 2 * k * temp * mu_ip**2 / (inertia**2 * c**3)
    # The aligned prolate grain rotates about an equatorial principal axis.
    return prefactor * (mu_ip**2 + 2 * mu_op**2)


# Manually implemented cumulative sum function
# @njit
def cumsum_axis_0(arr):
    result = np.zeros_like(arr)
    result[0] = arr[0]
    for i in range(1, arr.shape[0]):
        result[i] = result[i-1] + arr[i]
    return result


def aux_int(Tval, F, G, omega, Nomega, Nmu, tau_H_val, tau_ed_inv_val, Inertia_val, Dln_omega):

    """Solve AHD09 Eq. (33) on logarithmic cell centres, one column per dipole."""
    log_omega = np.log(omega)
    x = Inertia_val * omega[:, None]**2 / (k * Tval)
    drift = (F * x + tau_H_val / 3. * tau_ed_inv_val * x**2) / G

    # Cumulative trapezoid in ln(omega). The arbitrary integration constant
    # cancels on normalization. Dln_omega is the bin width d(omega)/omega.
    increments = 0.5 * (drift[:-1] + drift[1:]) * np.diff(log_omega)[:, None]
    exponent = np.vstack([np.zeros(drift.shape[1]), np.cumsum(increments, axis=0)])
    log_density = 2. * log_omega[:, None] - exponent
    log_norm = logsumexp(log_density + log_omega[:, None], axis=0) + np.log(Dln_omega)
    return (log_density - log_norm).T


# @njit
def rescale_f_rot(omega, f_a, beta, log=True):
    """Transform a density per solver frequency L/Imax to one per Omega=L/Iref."""
    ratio = inertia_ref_over_largest(beta)
    omega_new = omega / ratio
    if log:
        return np.log(omega_new), f_a + np.log(ratio)
    return omega_new, f_a * ratio


def _solver_plasma(env, a, beta, fZ, omega, mu_ip, mu_op, tumbling, a2, use_spdust=False):
    """Plasma drift and diffusion for omega=L/Imax and tau_H proportional to Imax."""
    if use_spdust:
        # The comparison prescription models a legacy disk or sphere. Its
        # input is the rotation rate about that model grain's largest axis.
        if not tumbling and beta > 0:
            mu_ip = np.sqrt(0.5 * mu_ip**2 + mu_op**2)
        return spd_plasmadrag.FGp_averaged(env, a, fZ, omega, mu_ip, mu_op, tumbling=tumbling)

    # The low-level plasma API uses the paper's L/I3 and tau_H proportional
    # to I3. F scales with tau_H, whereas G depends only on tau_H/I.
    largest_over_z = max(1.0, 1.0 + beta)
    rates = FGp_averaged(env, a, beta, fZ, omega * largest_over_z,
                         mu_ip, mu_op, tumbling=tumbling, a2=a2)
    return {'Fp': rates['Fp'] * largest_over_z, 'Gp': rates['Gp']}


def log_f_rot(env, a, beta, fZ, mu_ip, mu_op, tumbling=True, omega_min=1e8, omega_max=1e16, Nomega=1000, use_spdust_plasma=False, a2=a2_default):
    """Log probability density per unit Omega on the requested output grid.

    Each dipole has its own normalized density. The rates and normalization
    use an intrinsic grid in L/Imax. The output grid uses Omega=L/Iref.
    omega_min and omega_max only select where that density is sampled.
    """
    if not isinstance(Nomega, (int, np.integer)) or Nomega < 4:
        raise ValueError("Nomega must be an integer of at least four.")
    if not 0. < omega_min < omega_max or not np.isfinite(omega_max):
        raise ValueError("The rotation-frequency bounds must be finite, positive and increasing.")
    Nmu = np.size(mu_ip)  # Number of dipole moments

    Inertia_val = Inertia_largest(a, beta, a2=a2)  # Grain's moment of inertia

    Tval, nH = env['T'], env['nh']

    if beta==0: # spherical grain
        tumbling = False

    # Characteristic timescales
    tau_H_val = tau_H(Tval, nH, a, beta, a2=a2)
    tau_ed_inv_val = tau_ed_inv(Tval, a, beta, mu_ip, mu_op, tumbling=tumbling, a2=a2)

    # Evaporation temperature
    Tev_val = Tev_effective(env, a, beta, a2=a2)

    # Gas, IR and photoelectric coefficients already use the AHD collision
    # reference (tau_H proportional to Imax) or the invariant tau_H/Imax.
    FGn = FGn_averaged(env, a, beta, Tev_val, fZ, tumbling=tumbling, a2=a2)
    Fn = FGn['Fn']
    Gn = FGn['Gn']

    mu_tot = np.sqrt(mu_ip**2 + mu_op**2)
    FGi = FGi_averaged(env, a, beta, Tev_val, mu_tot, fZ, a2=a2)
    Fi = FGi['Fi']
    Gi = FGi['Gi']

    FGIR = FGIR_averaged(env, a, beta, fZ, a2=a2)
    FIR = FGIR['FIR']
    GIR = FGIR['GIR']

    FGpe = FGpe_averaged(env, a, beta, fZ, a2=a2)
    Fpe = FGpe['Fpe']
    Gpe = FGpe['Gpe']

    GH2_val = GH2(env, a, beta, a2=a2)

    def rates(omega):
        plasma = _solver_plasma(env, a, beta, fZ, omega, mu_ip, mu_op, tumbling, a2,
                                use_spdust=use_spdust_plasma)
        return (Fn + FIR + Fpe + Fi + plasma['Fp'],
                Gn + GIR + Gpe + GH2_val + Gi + plasma['Gp'])

    # Estimate the emission peak for each actual dipole from its rates at the
    # thermal rotation frequency. This only seeds the integration bounds.
    thermal = np.sqrt(6. * k * Tval / Inertia_val)
    F_th, G_th = rates(np.array([thermal]))
    xi = 8. * G_th / F_th**2 * tau_H_val * tau_ed_inv_val
    peaks = thermal * np.sqrt(2. * G_th / F_th / (1. + np.sqrt(1. + xi)))
    low, high = 1e-4 * np.min(peaks), 8. * np.max(peaks)
    if not (0. < low < high and np.isfinite(high)):
        raise ValueError("The rotation rates do not give finite positive peak frequencies.")

    for _ in range(12):
        omega = makelogtab(low, high, Nomega)
        F, G = rates(omega)
        log_f = aux_int(Tval, F, G, omega, Nomega, Nmu, tau_H_val, tau_ed_inv_val,
                        Inertia_val, DX_over_X(low, high, Nomega))
        # dP/dln(omega) sets the omitted low-frequency mass. The emitted
        # spectrum is proportional to omega^4 f and falls later at high omega.
        log_w = np.log(omega)
        log_mass = log_f + log_w
        log_emission = log_f + 4. * log_w
        extend_low = np.any(log_mass[:, 0] > log_mass.max(axis=1) - 22.)
        extend_high = np.any(log_emission[:, -1] > log_emission.max(axis=1) - 40.)
        if not extend_low and not extend_high:
            break
        if extend_low:
            low /= 4.
        if extend_high:
            high *= 2.
    else:
        raise ValueError("Could not bracket the tails of every rotation distribution.")

    log_Omega, log_f = rescale_f_rot(omega, log_f, beta)
    requested = np.log(makelogtab(omega_min, omega_max, Nomega))
    interior = (requested >= log_Omega[0]) & (requested <= log_Omega[-1])
    result = np.full((Nmu, Nomega), -np.inf)
    if interior.any():
        result[:, interior] = PchipInterpolator(log_Omega, log_f, axis=1)(requested[interior])
    # As Omega -> 0 the drift in the exponent vanishes and f scales as Omega².
    below = requested < log_Omega[0]
    result[:, below] = log_f[:, :1] + 2. * (requested[below] - log_Omega[0])
    return result


def f_rot_old(env, a, beta, fZ, mu_ip, mu_op, tumbling=True, omega_min=1e7, omega_max=1e11, Nomega=1000, a2=a2_default):
    """Fixed-grid solver returning (Omega, probability density per unit Omega).

    The grid has Nomega bin centres and the density has shape (Nmu, Nomega).
    This diagnostic solver normalizes within the supplied interval.
    """

    if beta==0: # spherical grain
        tumbling = False

    Nmu = np.size(mu_ip)  # Number of dipole moments

    Inertia_val = Inertia_largest(a, beta, a2=a2)  # Grain's moment of inertia

    Tval, nH = env['T'], env['nh']

    # Characteristic timescales
    tau_H_val = tau_H(Tval, nH, a, beta, a2=a2)
    tau_ed_inv_val = tau_ed_inv(Tval, a, beta, mu_ip, mu_op, tumbling=tumbling, a2=a2)

    # Evaporation temperature
    Tev_val = Tev_effective(env, a, beta, a2=a2)

    # F's and G's (except for plasma drag)
    FGn = FGn_averaged(env, a, beta, Tev_val, fZ, tumbling=tumbling, a2=a2)
    Fn = FGn['Fn']
    Gn = FGn['Gn']

    mu_tot = np.sqrt(mu_ip**2 + mu_op**2)
    FGi = FGi_averaged(env, a, beta, Tev_val, mu_tot, fZ, a2=a2)
    Fi = FGi['Fi']
    Gi = FGi['Gi']

    FGIR = FGIR_averaged(env, a, beta, fZ, a2=a2)
    FIR = FGIR['FIR']
    GIR = FGIR['GIR']

    FGpe = FGpe_averaged(env, a, beta, fZ, a2=a2)
    Fpe = FGpe['Fpe']
    Gpe = FGpe['Gpe']

    GH2_val = GH2(env, a, beta, a2=a2)

    omega = makelogtab(omega_min, omega_max, Nomega) * inertia_ref_over_largest(beta)
    Dln_omega = DX_over_X(omega_min, omega_max, Nomega)

    # Fp(omega), Gp(omega)
    FGp = _solver_plasma(env, a, beta, fZ, omega, mu_ip, mu_op, tumbling, a2)
    Fp = FGp['Fp'] # shape (Nomega, Nmu)
    Gp = FGp['Gp']

    F = Fn + FIR + Fpe + np.matmul(np.ones((Nomega, 1)), Fi.reshape(1, -1)) + Fp
    G = Gn + GIR + Gpe + GH2_val + np.matmul(np.ones((Nomega, 1)), Gi.reshape(1, -1)) + Gp
    log_f = aux_int(Tval, F, G, omega, Nomega, Nmu, tau_H_val, tau_ed_inv_val, Inertia_val, Dln_omega)
    return rescale_f_rot(omega, np.exp(log_f), beta, log=False)
   
