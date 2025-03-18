from .Grain import N_C, N_H, grain_distribution
from .util import cgsconst, makelogtab
from .mpiutil import *
import numpy as np

from .SED import mu2_f, SED, SED_imp

debye = cgsconst.debye

def SpyDust(environment, tumbling=True, output_file=None, min_freq=None, max_freq=None, n_freq=None, Ndipole=None, single_beta=False, spdust_plasma=False):

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
    f_a_beta = grain_obj.shape_and_size_dist(line, normalize=False, fixed_thickness=single_beta)
    a_tab = grain_obj.a_tab
    beta_tab = grain_obj.beta_tab

    ang_Omega_min = 1e7
    ang_Omega_max = 1e15
    N_angular_Omega = 1000
    angular_Omega_tab = makelogtab(ang_Omega_min, ang_Omega_max, N_angular_Omega)

    mu2_f_arr = mu2_f(environment, a_tab, beta_tab, f_a_beta, 
                      dipole_per_atom, ip, Ndip, 
                      tumbling=tumbling, 
                      parallel=True, 
                      contract_a =True, 
                      omega_min=ang_Omega_min, 
                      omega_max=ang_Omega_max, 
                      Nomega=N_angular_Omega,
                      spdust_plasma=spdust_plasma)
    
    cos_theta_list = np.linspace(-1, 1, 20)

    # We mimic spdust and give sort of "ad-hoc" treatment to the angular distribution of the grain rotation.
    # The user can provide a more sophisticated treatment of the distribution of internal alignment.
    cos_theta_weights = []
    for beta in beta_tab:
        if beta > 0: # rotation around the axis of largest inertia (prolate grain, theta = pi/2)
            aux = np.zeros_like(cos_theta_list)
            aux[15] = 1
            cos_theta_weights.append(aux)
        elif beta > -0.1: # rotation around the axis of largest inertia (nearly spherical grains; theta=0)
            aux = np.zeros_like(cos_theta_list)
            aux[-1] = 1
            cos_theta_weights.append(aux)
        elif tumbling:
            cos_theta_weights.append(None) # rotation with isotropic distribution of theta
        else:  # rotation around the axis of largest inertia (oblate grains; theta=0)
            aux = np.ones_like(cos_theta_list)
            aux[-1] = 1
            cos_theta_weights.append(aux)

    resultSED = SED(nu_tab, mu2_f_arr, beta_tab, angular_Omega_tab, cos_theta_list, cos_theta_weights)

     # Handle free-free emission if requested
    Jy = 1e-23
    result = np.zeros((2, Nnu))

    result[0, :] = nu_tab / GHz
    result[1, :] = resultSED / Jy

    # Write output to file
    if rank0:
        if output_file is not None:        
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

                np.savetxt(f, result.T, fmt='%12.6e') # Columns are nu, jnu_per_H, jnu_per_H_freefree
    
    return result # shape (2, Nnu); rows are nu, SED(nu) in Jy

    
    
def SpyDust_imp(environment, tumbling=True, output_file=None, min_freq=None, max_freq=None, n_freq=None, Ndipole=None, single_beta=False, spdust_plasma=False):

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
    f_a_beta = grain_obj.shape_and_size_dist(line, normalize=False, fixed_thickness=single_beta)
    a_tab = grain_obj.a_tab
    beta_tab = grain_obj.beta_tab

    ang_Omega_min = 1e7
    ang_Omega_max = 1e15
    N_angular_Omega = 1000
    angular_Omega_tab = makelogtab(ang_Omega_min, ang_Omega_max, N_angular_Omega)

    mu2_f_arr = mu2_f(environment, a_tab, beta_tab, f_a_beta, 
                      dipole_per_atom, ip, Ndip, 
                      tumbling=tumbling, 
                      parallel=True, 
                      contract_a =True, 
                      omega_min=ang_Omega_min, 
                      omega_max=ang_Omega_max, 
                      Nomega=N_angular_Omega,
                      spdust_plasma=spdust_plasma)
    
    cos_theta_list = np.linspace(-1, 1, 20)

    # We mimic spdust and give sort of "ad-hoc" treatment to the angular distribution of the grain rotation.
    # The user can provide a more sophisticated treatment of the distribution of internal alignment.
    cos_theta_weights = []
    for beta in beta_tab:
        if beta > 0: # rotation around the axis of largest inertia (prolate grain, theta = pi/2)
            aux = np.zeros_like(cos_theta_list)
            aux[15] = 1
            cos_theta_weights.append(aux)
        elif beta > -0.1: # rotation around the axis of largest inertia (nearly spherical grains; theta=0)
            aux = np.zeros_like(cos_theta_list)
            aux[-1] = 1
            cos_theta_weights.append(aux)
        elif tumbling:
            cos_theta_weights.append(None) # rotation with isotropic distribution of theta
        else:  # rotation around the axis of largest inertia (oblate grains; theta=0)
            aux = np.ones_like(cos_theta_list)
            aux[-1] = 1
            cos_theta_weights.append(aux)

    resultSED = SED_imp(nu_tab, mu2_f_arr, beta_tab, angular_Omega_tab, cos_theta_list, cos_theta_weights)

     # Handle free-free emission if requested
    Jy = 1e-23
    result = np.zeros((2, Nnu))

    result[0, :] = nu_tab / GHz
    result[1, :] = resultSED / Jy

    # Write output to file
    if rank0:
        if output_file is not None:        
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

                np.savetxt(f, result.T, fmt='%12.6e') # Columns are nu, jnu_per_H, jnu_per_H_freefree
    
    return result # shape (2, Nnu); rows are nu, SED(nu) in Jy

    
 

def SpyDust_single_grain(environment, a, beta, tumbling=True, min_freq=None, max_freq=None, n_freq=None, Ndipole=None):

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

    f_a_beta = np.array([1]).reshape(1, 1)
    a_tab = np.array([a])
    beta_tab = np.array([beta])

    ang_Omega_min = 1e7
    ang_Omega_max = 1e15
    N_angular_Omega = 1000
    angular_Omega_tab = makelogtab(ang_Omega_min, ang_Omega_max, N_angular_Omega)

    mu2_f_arr = mu2_f(environment, a_tab, beta_tab, f_a_beta, 
                      dipole_per_atom, ip, Ndip, 
                      tumbling=tumbling, 
                      parallel=True, 
                      contract_a =True, 
                      omega_min=ang_Omega_min, 
                      omega_max=ang_Omega_max, 
                      Nomega=N_angular_Omega,
                      spdust_plasma=False)
    
    cos_theta_list = np.linspace(-1, 1, 20)

    # We mimic spdust and give a quite ad-hoc treatment to the angular distribution of the grain rotation.
    # The user can provide a more sophisticated treatment of the distribution of internal alignment.
    cos_theta_weights = []
    for beta in beta_tab:
        if beta > 0: # rotation around the axis of largest inertia (prolate grain, theta = pi/2)
            aux = np.zeros_like(cos_theta_list)
            aux[15] = 1
            cos_theta_weights.append(aux)
        elif beta > -0.1: # rotation around the axis of largest inertia (nearly spherical grains; theta=0)
            aux = np.zeros_like(cos_theta_list)
            aux[-1] = 1
            cos_theta_weights.append(aux)
        elif tumbling:
            cos_theta_weights.append(None) # rotation with isotropic distribution of theta
        else:  # rotation around the axis of largest inertia (oblate grains; theta=0)
            aux = np.ones_like(cos_theta_list)
            aux[-1] = 1
            cos_theta_weights.append(aux)

    resultSED = SED(nu_tab, mu2_f_arr, beta_tab, angular_Omega_tab, cos_theta_list, cos_theta_weights)

     # Handle free-free emission if requested
    Jy = 1e-23
    result = np.zeros((2, Nnu))

    result[0, :] = nu_tab / GHz
    result[1, :] = resultSED / Jy
    
    return result # shape (2, Nnu); rows are nu, SED(nu) in Jy

   


    
