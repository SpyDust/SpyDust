from .Grain import N_C, N_H, grain_distribution, grainparams
from .util import cgsconst, makelogtab
from .mpiutil import *
import numpy as np

from .SED import mu2_f, SED, SED_imp

debye = cgsconst.debye
a2_default = grainparams.a2
d = grainparams.d

def generate_ang_Omega_limits(numin, numax, cos_theta_list, beta_list):
    assert np.all((cos_theta_list >= -1) & (cos_theta_list <= 1)), "cos_theta_list must be in [-1, 1]"
    assert np.all(cos_theta_list != 0), "cos_theta_list must not contain 0 to avoid division by zero."

    beta_tab = beta_list[beta_list != 0]
    if beta_tab.size > 0:
        beta_cos_theta = beta_tab[:, np.newaxis] * cos_theta_list[np.newaxis, :]
        min_beta_cos_beta = np.min(np.abs(beta_cos_theta))
        min_one_minus_beta_cos = np.min(np.abs(1. - beta_cos_theta))
        max_one_minus_beta_cos = np.max(np.abs(1. - beta_cos_theta))
        min_one_plus_beta_cos = np.min(np.abs(1. + beta_cos_theta))
        max_one_plus_beta_cos = np.max(np.abs(1. + beta_cos_theta))
        min_denom = min(min_one_minus_beta_cos, min_one_plus_beta_cos, min_beta_cos_beta)
        max_denom = max(max_one_minus_beta_cos, max_one_plus_beta_cos)
        ang_Omega_min = numin*2*np.pi / max_denom / 5.0
        ang_Omega_max = numax*2*np.pi / min_denom * 5.0
        return ang_Omega_min, ang_Omega_max
    else:
        return 1e6, 1e17

def generate_cos_theta_list(num_points=20):
    edges = np.linspace(-1, 1, num_points+1)  # 21 edges define 20 intervals
    cos_theta_list = 0.5 * (edges[:-1] + edges[1:])
    return cos_theta_list

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
            d=d):

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
    f_a_beta = grain_obj.shape_and_size_dist(line, normalize=False, fixed_thickness=single_beta, a2=a2, d=d)
    a_tab = grain_obj.a_tab
    beta_tab = grain_obj.beta_tab

    cos_theta_list = generate_cos_theta_list(num_points=Ncos_theta)

    ang_Omega_min, ang_Omega_max = generate_ang_Omega_limits(numin, numax, cos_theta_list, beta_tab)
    angular_Omega_tab = makelogtab(ang_Omega_min, ang_Omega_max, N_angular_Omega)

    mu2_f_arr = mu2_f(environment, a_tab, beta_tab, f_a_beta, 
                      dipole_per_atom, ip, Ndip, 
                      tumbling=tumbling, 
                      parallel=True, 
                      contract_a =True, 
                      omega_min=ang_Omega_min, 
                      omega_max=ang_Omega_max, 
                      Nomega=N_angular_Omega,
                      spdust_plasma=spdust_plasma,
                      a2=a2
                      )

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



def SpyDust_imp(environment, tumbling=True, output_file=None, min_freq=None, max_freq=None, n_freq=None, 
                Ndipole=None, Ncos_theta=20, single_beta=False, spdust_plasma=False, a2=a2_default, d=d):

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
    f_a_beta = grain_obj.shape_and_size_dist(line, normalize=False, fixed_thickness=single_beta, a2=a2, d=d)
    a_tab = grain_obj.a_tab
    beta_tab = grain_obj.beta_tab

    cos_theta_list = generate_cos_theta_list(num_points=Ncos_theta)

    ang_Omega_min, ang_Omega_max = generate_ang_Omega_limits(numin, numax, cos_theta_list, beta_tab)

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
                      spdust_plasma=spdust_plasma,
                      a2=a2
                      )

    

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
    # ang_Omega_min (float, optional): The minimum angular frequency in Hz. Default is 1e7.
    # ang_Omega_max (float, optional): The maximum angular frequency in Hz. Default is 1e15.
    N_angular_Omega (int, optional): The number of angular frequency points. Default is 500.
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

    cos_theta_list = generate_cos_theta_list(num_points=Ncos_theta)

    ang_Omega_min, ang_Omega_max = generate_ang_Omega_limits(numin, numax, cos_theta_list, beta_tab)

    angular_Omega_tab = makelogtab(ang_Omega_min, ang_Omega_max, N_angular_Omega)

    mu2_f_arr = mu2_f(environment, a_tab, beta_tab, f_a_beta, 
                      dipole_per_atom, ip, Ndip, 
                      tumbling=tumbling, 
                      parallel=True, 
                      contract_a =True, 
                      omega_min=ang_Omega_min, 
                      omega_max=ang_Omega_max, 
                      Nomega=N_angular_Omega,
                      spdust_plasma=False,
                      a2=a2)
    
    # We mimic spdust and give a quite ad-hoc treatment to the angular distribution of the grain rotation.
    # The user can provide a more sophisticated treatment of the distribution of internal alignment.
    cos_theta_weights = []

    if beta == 0:
        tumbling = False

    if not tumbling: # if not tumbling, all grains spin around the axis of largest inertia
        if beta > 0: # rotation around the axis of largest inertia (prolate grain, theta = pi/2)
            aux = np.zeros_like(cos_theta_list)
            aux[15] = 1
            cos_theta_weights.append(aux)
        else:  # rotation around the axis of largest inertia (oblate grains; theta=0)
            aux = np.ones_like(cos_theta_list)
            aux[-1] = 1
            cos_theta_weights.append(aux)
    else:  # tumbling grains
        cos_theta_weights.append(None) # rotation with isotropic distribution of theta

    resultSED = SED_imp(nu_tab, mu2_f_arr, beta_tab, angular_Omega_tab, cos_theta_list, cos_theta_weights)

     # Handle free-free emission if requested
    Jy = 1e-23
    result = np.zeros((2, Nnu))

    result[0, :] = nu_tab / GHz
    result[1, :] = resultSED / Jy
    
    return result # shape (2, Nnu); rows are nu, SED(nu) in Jy

   


    
