from utils.util import cgsconst, makelogtab
from spdust.grain_properties import N_C, N_H
from spdust.emissivity import emissivity, free_free
import numpy as np

pi = np.pi
c = cgsconst.c
q = cgsconst.q
k = cgsconst.k
mp = cgsconst.mp
debye = 1e-18
eV = cgsconst.eV


def SPDUST(environment, tumbling=True, output_file=None, min_freq=None, max_freq=None, n_freq=None, Ndipole=None, freefree=False):
    
    # Check the environment structure for required parameters
    if 'dipole' not in environment and 'beta' not in environment:
        print("Please specify the dipole moment or beta.")
        return
    
    # Determine the dipole moment
    if 'beta' in environment:
        beta0 = environment['beta'] * debye
        mu_1d_7 = np.sqrt(N_C(1e-7) + N_H(1e-7)) * environment['beta']
    else:
        mu_1d_7 = environment['dipole']
        beta0 = mu_1d_7 / np.sqrt(N_C(1e-7) + N_H(1e-7)) * debye

    # Check for grain size distribution parameters
    if 'line' not in environment:
        print("Please specify the grain size distribution parameters (Weingartner & Draine, 2001a).")
        return

    # Number of dipole moments
    Ndip = 20
    if Ndipole is not None:
        Ndip = Ndipole

    # Set in-plane moment ratio
    ip = 2 / 3
    if 'inplane' in environment:
        print(f"Assuming that <mu_ip^2>/<mu^2> = {environment['inplane']} for disklike grains")
        ip = environment['inplane']

    # Frequency settings
    GHz = 1e9
    numin = 0.5 * GHz
    numax = 500 * GHz
    Nnu = 200
    if min_freq is not None:
        numin = min_freq * GHz
    if max_freq is not None:
        numax = max_freq * GHz
    if n_freq is not None:
        Nnu = n_freq

    nu_tab = makelogtab(numin, numax, Nnu)

    # Calculate emissivity
    if tumbling:
        jnu_per_H = emissivity(environment, beta0, ip, Ndip, nu_tab, tumbling=True)
    else:
        print("Assuming that disklike grains spin around their axis of greatest inertia")
        jnu_per_H = emissivity(environment, beta0, ip, Ndip, nu_tab, tumbling=False)
        

    # Handle free-free emission if requested
    Jy = 1e-23
    result = np.zeros((3 if freefree else 2, Nnu))
    if freefree:
        result[2, :] = free_free(environment, nu_tab) / Jy
    result[0, :] = nu_tab / GHz
    result[1, :] = jnu_per_H / Jy

    # Write output to file
    if output_file is not None:        
        with open(output_file, 'w') as f:
            f.write('#=========================== SPDUST.2 ===============================\n')
            f.write('#    Rotational emission from a population of spinning dust grains,\n')
            f.write('#    as described by Ali-Haimoud, Hirata & Dickinson, 2009, MNRAS, 395, 1055\n')
            f.write('#    and in Silsbee, Ali-Haimoud & Hirata, 2010\n')
            f.write(f'#    nH = {environment["nh"]} cm^-3\n')
            f.write(f'#    T = {environment["T"]} K\n')
            f.write(f'#    Chi = {environment["Chi"]}\n')
            f.write(f'#    xh = {environment["xh"]}\n')
            f.write(f'#    xC = {environment["xC"]}\n')
            f.write(f'#    mu(1E-7 cm) = {mu_1d_7} debye (beta = {beta0 / debye} debye)\n')
            if tumbling:
                f.write('#    Disklike grains are randomly oriented with respect to angular momentum.\n')
            else:
                f.write('#    Disklike grains spin around their axis of greatest inertia\n')
            f.write('#=====================================================================\n')
            if freefree:
                f.write('#nu(GHz)       j_nu/nH(Jy sr-1 cm2/H)     j_nu/nH (free-free) (Jy sr-1 cm2/H)\n')
            else:
                f.write('#nu(GHz)       j_nu/nH(Jy sr-1 cm2/H)\n')
            np.savetxt(f, result.T, fmt='%12.6e') # Columns are nu, jnu_per_H, jnu_per_H_freefree
    
    return result # shape (2, Nnu) or (3, Nnu) if freefree is True; rows are nu, jnu_per_H, jnu_per_H_freefree


