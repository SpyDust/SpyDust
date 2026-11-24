from .Grain import N_C, N_H, grain_distribution, grainparams
from .util import cgsconst, makelogtab
from .mpiutil import *
import numpy as np

from .SED import mu2_f, SED, SED_imp

debye = cgsconst.debye
a2_default = grainparams.a2
d = grainparams.d

def generate_ang_Omega_limits(numin, numax, cos_theta_list, beta_list, cos_theta_weights=None):
    """Return padded rotation-frequency bounds for the active emission modes."""
    x = np.asarray(cos_theta_list)
    if not np.all(np.abs(x) <= 1.):
        raise ValueError("cos_theta_list must contain direction cosines in [-1, 1].")
    beta = np.asarray(beta_list)
    active = np.ones((len(beta), len(x)), dtype=bool)
    if cos_theta_weights is not None:
        for i, row in enumerate(cos_theta_weights):
            if row is not None:
                active[i] = np.asarray(row) > 0.
    bx = beta[:, None] * x
    factors = [np.array([1.])]  # Mode 1.
    for mapping, radiates in ((1. + bx, x != -1.), (1. - bx, x != 1.),
                              (bx, np.abs(x) != 1.)):
        factors.append(np.abs(mapping[active & radiates & (mapping != 0.)]))
    factors = np.concatenate(factors)
    return 2*np.pi*numin / factors.max() / 5., 2*np.pi*numax / factors.min() * 5.


def generate_cos_theta_list(num_points=20):
    """Midpoints of equally sized intervals in cos(theta)."""
    if not isinstance(num_points, (int, np.integer)) or num_points < 1:
        raise ValueError("num_points must be a positive integer.")
    return (2*np.arange(num_points) + 1 - num_points) / num_points


def _alignment_grid(beta_tab, ncos, tumbling, population):
    """Represent isotropic orientations and exact alignment impulses on a shared grid.

    Non-tumbling grains rotate about the largest-inertia axis. The population
    also aligns nearly spherical grains when tumbling is enabled.
    """
    if not isinstance(ncos, (int, np.integer)) or ncos < 1:
        raise ValueError("Ncos_theta must be a positive integer.")
    quad_x, quad_w = np.polynomial.legendre.leggauss(ncos)
    cosines = np.unique(np.r_[quad_x, 0., 1.])
    isotropic = np.zeros_like(cosines)
    isotropic[np.searchsorted(cosines, quad_x)] = quad_w / 2.
    rows = []
    beta_tab = np.asarray(beta_tab)
    isotropic_shapes = (beta_tab != 0.) & bool(tumbling)
    if population:
        isotropic_shapes &= beta_tab <= -0.1
    for beta, random_orientation in zip(beta_tab, isotropic_shapes):
        if random_orientation:
            rows.append(isotropic)
        else:
            # Rotation about the largest-inertia axis: perpendicular to the
            # symmetry axis for prolate grains and along it for oblate grains.
            rows.append((cosines == (0. if beta > 0. else 1.)).astype(float))
    return cosines, rows, isotropic_shapes


def _grain_spectrum(environment, a_tab, beta_tab, weights, dipole, ip, ndip, tumbling,
                    a2, spdust_plasma, nspin, ncos, population, nu_tab, numin, numax):
    cosines, orientations, isotropic = _alignment_grid(beta_tab, ncos, tumbling, population)
    low, high = generate_ang_Omega_limits(numin, numax, cosines, beta_tab, orientations)
    for _ in range(16):
        moments = mu2_f(environment, a_tab, beta_tab, weights, dipole, ip, ndip,
                        tumbling=tumbling, omega_min=low, omega_max=high, Nomega=nspin,
                        spdust_plasma=spdust_plasma, a2=a2)
        # The isotropic fourth mode samples spins up to infinity as cos(theta)
        # approaches zero. A narrow observing band can end below the spin peak.
        transverse = moments[isotropic, 0, :]
        if not np.any(transverse[:, -1] > 1e-12 * transverse.max(axis=1)):
            rotation = makelogtab(low, high, nspin)
            return SED(nu_tab, moments, beta_tab, rotation, cosines, orientations,
                       isotropic=isotropic)
        high *= 16.
    raise ValueError("Could not bracket the spin distribution for isotropic radiation.")


def SpyDust(environment, 
            tumbling=True, 
            output_file=None, 
            min_freq=None, 
            max_freq=None, 
            n_freq=None, 
            Ndipole=None, 
            Ncos_theta=20,
            single_beta=False, 
            spdust_plasma=False, 
            N_angular_Omega=500,
            a2=a2_default,
            d=d, Nbeta=grain_distribution.default_Nbeta):

    """Population spectrum [frequency in GHz, historical mode spectrum per H].

    Divide row 1 by four for absolute emissivity in Jy sr^-1 cm² per H.

    Nbeta controls the shape grid of the ensemble model. With single_beta=True
    each size uses its exact fixed-thickness shape. Ncos_theta sets the angular grid used to estimate the initial spin-frequency
    range. Continuous isotropic emission is integrated over the spin grid.
    """
    # Check the environment structure for required parameters
    if 'dipole' not in environment and 'dipole_per_atom' not in environment:
        if rank0:
            print("Please specify the dipole moment (of a=1e-7 cm) or dipole_per_atom.")
        return
    
    # Determine the dipole moment
    if 'dipole' in environment:
        mu_1d_7 = environment['dipole']
        dipole_per_atom = mu_1d_7 / np.sqrt(N_C(1e-7) + N_H(1e-7)) * debye
    else:
        dipole_per_atom = environment['dipole_per_atom'] * debye
        mu_1d_7 = np.sqrt(N_C(1e-7) + N_H(1e-7)) * environment['dipole_per_atom']
        

    # Check for grain size distribution parameters
    if 'line' not in environment:
        if rank0:
            print("Please specify the grain size distribution parameters (Weingartner & Draine, 2001a).")
        return

    line = environment['line']-1

    # Number of dipole moments
    Ndip = 20
    if Ndipole is not None:
        Ndip = Ndipole

    # Set in-plane moment ratio
    ip = 2 / 3
    if 'inplane' in environment:
        if rank0:
            print(f"Assuming that <mu_ip^2>/<mu^2> = {environment['inplane']} for disklike grains")
        ip = environment['inplane']

    # Frequency settings
    GHz = 1e9
    numin = 1 * GHz
    numax = 100 * GHz
    Nnu = 200
    if min_freq is not None:
        numin = min_freq * GHz
    if max_freq is not None:
        numax = max_freq * GHz
    if n_freq is not None:
        Nnu = n_freq

    nu_tab = makelogtab(numin, numax, Nnu)

    grain_obj = grain_distribution()
    f_a_beta = grain_obj.shape_and_size_dist(line, normalize=False, fixed_thickness=single_beta, a2=a2, d=d, Nbeta=Nbeta)
    a_tab = grain_obj.a_tab
    beta_tab = grain_obj.beta_tab

    resultSED = _grain_spectrum(environment, a_tab, beta_tab, f_a_beta,
                               dipole_per_atom, ip, Ndip, tumbling, a2, spdust_plasma,
                               N_angular_Omega, Ncos_theta, True, nu_tab, numin, numax)

    Jy = 1e-23
    result = np.zeros((2, Nnu))

    result[0, :] = nu_tab / GHz
    result[1, :] = resultSED / Jy

    # Only the root writes. Every participating rank observes a write failure.
    write_error = None
    if rank0 and output_file is not None:
        try:
            with open(output_file, 'w') as f:
                f.write('#=========================== SPYDUST ===============================\n')
                f.write(f'#    nH = {environment["nh"]} cm^-3\n')
                f.write(f'#    T = {environment["T"]} K\n')
                f.write(f'#    Chi = {environment["Chi"]}\n')
                f.write(f'#    xh = {environment["xh"]}\n')
                f.write(f'#    xC = {environment["xC"]}\n')
                f.write(f'#    mu(1E-7 cm) = {mu_1d_7} debye (beta = {dipole_per_atom / debye} debye)\n')
                if tumbling:
                    f.write('#    Disklike grains are randomly oriented with respect to angular momentum.\n')
                else:
                    f.write('#    Disklike grains spin around their axis of greatest inertia\n')
                f.write('#=====================================================================\n')

                f.write('# Frequency (GHz), historical mode spectrum (divide by 4 for Jy sr^-1 cm^2/H)\n')
                np.savetxt(f, result.T, fmt='%12.6e')
        except Exception as exc:
            if size == 1:
                raise
            write_error = f"{type(exc).__name__}: {exc}"
    if size > 1:
        write_error = world.bcast(write_error, root=0)
        if write_error is not None:
            raise OSError(f"SpyDust could not write the spectrum on rank 0: {write_error}")
    
    return result



def SpyDust_imp(environment, tumbling=True, output_file=None, min_freq=None, max_freq=None, n_freq=None,
                Ndipole=None, Ncos_theta=20, single_beta=False, spdust_plasma=False, a2=a2_default, d=d,
                Nbeta=grain_distribution.default_Nbeta):
    """Population spectrum with the historical 1000-point rotation grid."""
    return SpyDust(environment, tumbling=tumbling, output_file=output_file, min_freq=min_freq,
                   max_freq=max_freq, n_freq=n_freq, Ndipole=Ndipole, Ncos_theta=Ncos_theta,
                   single_beta=single_beta, spdust_plasma=spdust_plasma, N_angular_Omega=1000,
                   a2=a2, d=d, Nbeta=Nbeta)


def SpyDust_given_grain_size_shape(environment, a, beta, tumbling=True, min_freq=None, max_freq=None, n_freq=None, Ndipole=None,
                                    N_angular_Omega=500,
                                    Ncos_theta=20,
                                    a2=a2_default
                                    ):
    """
    Parameters:
    environment (dict): A dictionary containing the environment parameters.
    a (float): The grain size in cm.
    beta (float): The grain shape parameter.
    tumbling (bool, optional): Whether the grains are tumbling. Default is True.
    min_freq (float, optional): The minimum frequency in GHz. Default is None.
    max_freq (float, optional): The maximum frequency in GHz. Default is None.
    n_freq (int, optional): The number of frequency points. Default is None.
    Ndipole (int, optional): The number of dipole moments. Default is None.
    N_angular_Omega (int, optional): The number of angular frequency points. Default is 500.

    Returns the frequency in GHz and the historical single-grain mode spectrum.
    Divide the spectrum by four for the spectral emission per Hz per steradian.
    """
    


    # Check the environment structure for required parameters
    if 'dipole' not in environment and 'dipole_per_atom' not in environment:
        if rank0:
            print("Please specify the dipole moment (of a=1e-7 cm) or dipole_per_atom.")
        return
    
    # Determine the dipole moment
    if 'dipole' in environment:
        mu_1d_7 = environment['dipole']
        dipole_per_atom = mu_1d_7 / np.sqrt(N_C(1e-7) + N_H(1e-7)) * debye
    else:
        dipole_per_atom = environment['dipole_per_atom'] * debye
        mu_1d_7 = np.sqrt(N_C(1e-7) + N_H(1e-7)) * environment['dipole_per_atom']
        

    # Number of dipole moments
    Ndip = 20
    if Ndipole is not None:
        Ndip = Ndipole

    # Set in-plane moment ratio
    ip = 2 / 3
    if 'inplane' in environment:
        if rank0:
            print(f"Assuming that <mu_ip^2>/<mu^2> = {environment['inplane']} for disklike grains")
        ip = environment['inplane']

    # Frequency settings
    GHz = 1e9
    numin = 0.1 * GHz
    numax = 100 * GHz
    Nnu = 200
    if min_freq is not None:
        numin = min_freq * GHz
    if max_freq is not None:
        numax = max_freq * GHz
    if n_freq is not None:
        Nnu = n_freq

    nu_tab = makelogtab(numin, numax, Nnu)

    f_a_beta = np.array([1]).reshape(1, 1)
    a_tab = np.array([a])
    beta_tab = np.array([beta])

    resultSED = _grain_spectrum(environment, a_tab, beta_tab, f_a_beta,
                               dipole_per_atom, ip, Ndip, tumbling, a2, False,
                               N_angular_Omega, Ncos_theta, False, nu_tab, numin, numax)

    Jy = 1e-23
    result = np.zeros((2, Nnu))

    result[0, :] = nu_tab / GHz
    result[1, :] = resultSED / Jy
    
    return result
