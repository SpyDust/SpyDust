from .util import cgsconst, makelogtab, DX_over_X
from scipy.special import erf
from . import SpDust_data_dir
#from numba import njit
import numpy as np
import os

class grainparams:
    # Stores some grain geometrical properties
    a2 = 6e-8          # radius under which grains are clyindrical, above which grains are ellipsoidal
    d = 3.35e-8        # assumed minimal disk thickness (graphite interlayer separation)
    rho = 2.24         # carbon density in g/cm^3
    epsilon = 0.01     # charge centroid displacement

rho = grainparams.rho # carbon density in g/cm^3
a2 = grainparams.a2
d = grainparams.d
epsilon = grainparams.epsilon

pi = np.pi
mp = cgsconst.mp
q = cgsconst.q

# Number of carbon atoms in a grain
#@njit
def N_C(a):
    return int(np.floor(4 * pi / 3 * a**3 * rho / (12 * mp)) + 1)

# Number of hydrogen atoms in a grain
#@njit
def N_H(a):
    Nc = N_C(a)
    if Nc < 25:
        result = np.floor(0.5 * Nc + 0.5)
    elif Nc < 100:
        result = np.floor(2.5 * np.sqrt(Nc) + 0.5)
    else:
        result = np.floor(0.25 * Nc + 0.5)
    return int(result)
    
#@njit
def Inertia_z(I_ref, beta):
    """
    Calculate the moment of inertia around the z-axis. Also notated as I_3.
    """
    return I_ref / (1 + beta)
    
#@njit    
def grain_mass(a):
    """"
    Calculate the mass of the grain given the effective radius. (Functions below will define the effective radius)

    Parameters:
    -----------
    a: float
        The effective radius of the grain.
    """
    return (12 * N_C(a) + N_H(a)) * mp

########################################### Functions for cylindrical type grains ###########################################
#@njit
def effective_radius_cylindrical(I_ref, beta):
    '''
    Calculate the effective radius of an elliptical cylinder grain.
    '''
    result = (9 / 4*pi*rho) * I_ref / (1 + beta) * (beta + 0.5)**(1/3)
    return result**(1/5)

#@njit
def Inertia_ref_cylindrical(a, beta):
    '''
    Calculate the reference moment of inertia.
    '''
    aux = 9 * (beta + 0.5)**(1/3) / (1 + beta)
    mass = grain_mass(a)
    real_rho = mass / (4/3 * pi * a**3)
    return a**5 * 4 * pi * real_rho / aux 

#@njit
def cylindrical_radius(a, beta):
    '''
    Calculate the radius of the cylindrical grain in the plane spaned by grain body axes 1 and 2.
    '''
    # a = effective_radius_cylindrical(I_ref, beta)
    mass = grain_mass(a)
    I_ref = Inertia_ref_cylindrical(a, beta)
    return np.sqrt( 2 * Inertia_z(I_ref, beta) / mass )

#@njit
def cylindrical_thickness(a, beta):
    """
    Calculate the thickness of the cylindrical grain.
    """

    mass = grain_mass(a)
    I_ref = Inertia_ref_cylindrical(a, beta)
    result = (2 * I_ref - Inertia_z(I_ref, beta)) * 6 / mass
    return np.sqrt(result)

#@njit
def cylindrical_params(a, d_val):
    '''
    Given the effective radius (a) and thickness (d), calculate the I_ref and beta
    '''
    mass = grain_mass(a)
    b_squared = 4/3 * a**3 / d_val
    I_ref = (b_squared / 4 + d_val**2 / 12) * mass
    I_z = 0.5 * b_squared  * mass
    beta = I_ref / I_z - 1
    return I_ref, beta

#@njit
def beta_min(a):
    '''
    Calculate the minimum value of beta for a cylindrical grain, assuming the thinest disk-like grain - single .
    '''
    return cylindrical_params(a, d)[1]

#@njit
def asurf_cylindrical(a, beta):
    ''''
    Calculate the surface-equivalent radius for cylindrical grains.
    '''
    b_val = cylindrical_radius(a, beta)
    d_val = cylindrical_thickness(a, beta)
    return np.sqrt(b_val**2 / 2 + b_val * d_val / 2)

#@njit
def acx_cylindrical(a, beta):
    '''
    Calculate the cylindrical excitation equivalent radius for cylindrical grains. Assume thin disk-like grains.
    '''
    return (3 / 8)**0.25 * cylindrical_radius(a, beta)


########################################### Functions for ellipsoid grains ###########################################
#@njit
def effective_radius_ellipsoidal(I_ref, beta):
    '''
    Calculate the effective radius of an ellipsoidal grain.
    '''
    result = (15 * I_ref / 8*pi*rho) / (1 + beta) * (2 * beta + 1)**(1/3)
    return result**(1/5)

#@njit
def Inertia_ref_ellipsoidal(a, beta):
    '''
    Calculate the reference moment of inertia.
    '''
    aux = 15 * (2 * beta + 1)**(1/3) / (1 + beta)
    mass = grain_mass(a)
    real_rho = mass / (4/3 * pi * a**3)
    return a**5 * 8 * pi * real_rho / aux

#@njit
def radii_ellipsoidal(a, beta):
    '''
    Calculate the radii of the ellipsoidal grain.
    '''
    I_ref = Inertia_ref_ellipsoidal(a, beta)
    I1 = I2 = I_ref
    I3 = Inertia_z(I_ref, beta)

    mass = grain_mass(a)

    r_1 = r_2 = np.sqrt(5 * I3 / (2 * mass))
    r_3 = np.sqrt(5 * (I1 + I2 - I3) / (2 * mass))

    return r_1, r_2, r_3

#@njit
def asurf_ellipsoidal(a, beta):
    ''''
    Calculate the surface-equivalent radius for ellipsoidal grains.
    '''
    if beta == 0: # Spherical grain
        return a
    elif beta < 0: # Oblate grain
        r_1, r_2, r_3 = radii_ellipsoidal(a, beta)
        e = np.sqrt(1 - r_3**2 / r_1**2)
        surface = 2 * pi * r_1**2 * (1 + (1 - e**2) / e * np.arctanh(e))
        asurf = np.sqrt(surface / (4 * pi))
        return asurf
    else: # Prolate grain
        r_1, r_2, r_3 = radii_ellipsoidal(a, beta)
        e = np.sqrt(1 - r_1**2 / r_3**2)
        surface = 2 * pi * r_1**2 * (1 + r_3 / r_1 / e * np.arcsin(e))
        asurf = np.sqrt(surface / (4 * pi))
        return asurf

#@njit
def acx_ellipsoidal(a, beta):
    '''
    Calculate the excitation equivalent radius for ellipsoidal grains.

    Assume nearly spherical grains; use asurf as the excitation equivalent radius.
    '''
    return asurf_ellipsoidal(a, beta) 

########################################### Functions for general grains ###########################################
# Assume cylindrical grains for a < a2 and ellipsoidal grains for a > a2

#@njit
def Inertia_ref(a, beta):
    '''
    Calculate the reference moment of inertia.
    '''
    if a <= a2:
        return Inertia_ref_cylindrical(a, beta)
    else:
        return Inertia_ref_ellipsoidal(a, beta)
    
#@njit
def asurf(a, beta):
    '''
    Calculate the surface-equivalent radius for general grains.
    '''
    if a <= a2:
        return asurf_cylindrical(a, beta)
    else:
        return asurf_ellipsoidal(a, beta)
    
#@njit
def acx(a, beta):
    '''
    Calculate the excitation equivalent radius for general grains.
    '''
    if a <= a2:
        return acx_cylindrical(a, beta)
    else:
        return acx_ellipsoidal(a, beta)
    
#@njit
def Inertia_largest(a, beta):
    '''
    Calculate the largest moment of inertia.
    '''

    I_ref = Inertia_ref(a, beta)

    if beta < 0:
        return Inertia_z(I_ref, beta)
    else:
        return I_ref
    
#@njit
def rms_dipole(a, beta, Z2, mole_dipole):
    '''
    Calculate the root mean square dipole moment.

    Parameters:
    -----------
    a: float
        The effective radius of the grain.
    Z2: float
        The square of the charge.
    mole_dipole: float
        The typical molecular dipole moment. (This is equivalent to the "beta" in spdust (AHD09;SAH11). Don't confuse with the grain parameter beta in SpyDust.)
    '''
    muZ = epsilon * np.sqrt(Z2) * q * acx(a, beta)
    N_at = N_C(a) + N_H(a)
    return np.sqrt(N_at * mole_dipole**2 + muZ**2)

########################################### Functions for grain size and shape distributions ###########################################

# Define the path to the size distribution file
size_dist_file = os.path.join(SpDust_data_dir, 'sizedists_table1.out')

class size_dist_arrays:
    bc1e5_tab, alpha_tab, beta_tab, at_tab, ac_tab, C_tab = \
    np.loadtxt(size_dist_file, usecols=(1, 3, 4, 5, 6, 7), unpack=True, comments=';')

#@njit
def apodized_Gaussian_dist(x, centre, sigma, right_half = False):
    '''
    An auxiliary Gaussian distribution apodized by a cosine function.
    This is a finite distribution with range [centre - 2 * sigma, centre + 2 * sigma].
    '''
    if x >= centre + 2 * sigma:
        return 0
    elif x>= centre:
        result = np.exp(-0.5 * (x - centre)**2 / sigma**2) / (sigma * np.sqrt(2 * np.pi)) # Gaussian part
        result *= np.cos(np.pi * (x - centre) / 4 / sigma) # Apodization at 2 sigma
        return result   
    elif right_half or x <= centre - 2 * sigma:
        return 0
    else:
        result = np.exp(-0.5 * (x - centre)**2 / sigma**2) / (sigma * np.sqrt(2 * np.pi))
        result *= np.cos(np.pi * (x - centre) / 4 / sigma)
        return result

class grain_distribution():
    bc, alpha_g, beta_g, at_g, ac_g, C_g = None, None, None, None, None, None

    def __init__(self, a_min=3.5e-8, a_max=3.5e-7, Na=30): # Default settings for grain size distribution
        self.a_min = a_min
        self.a_max = a_max
        self.Na = Na
        self.a_tab = makelogtab(a_min, a_max, Na)
        self.Da_over_a = DX_over_X(a_min, a_max, Na)

    def size_dist_func(self, line): 
        # Unpack size distribution parameters
        bc = size_dist_arrays.bc1e5_tab[line] * 1e-5
        alpha_g = size_dist_arrays.alpha_tab[line]
        beta_g = size_dist_arrays.beta_tab[line]
        at_g = size_dist_arrays.at_tab[line] * 1e-4
        ac_g = size_dist_arrays.ac_tab[line] * 1e-4
        C_g = size_dist_arrays.C_tab[line]

        amin = self.a_min
        # Lognormal populations parameters
        mc = 12 * mp
        bci = np.array([0.75, 0.25]) * bc
        a0i = np.array([3.5, 30.]) * 1e-8
        sigma = 0.4

        Bi = 3 / (2 * np.pi)**1.5 * np.exp(-4.5 * sigma**2) / (
                rho * a0i**3 * sigma) * bci * mc / (1 + erf(3 * sigma / np.sqrt(2) + np.log(a0i / amin) / (sigma * np.sqrt(2))))
        
        #@njit
        def size_dist(a):
            '''
            Grain size distribution, using Weingartner & Draine, 2001a prescription. 
            The line of their table 1 is given by the user in param_file.
            '''
            
            D_a = np.sum(Bi / a * np.exp(-0.5 * (np.log(a / a0i) / sigma)**2))

            if beta_g >= 0:
                F_a = 1 + beta_g * a / at_g
            else:
                F_a = 1 / (1 - beta_g * a / at_g)

            cutoff = 1
            if a > at_g:
                cutoff = np.exp(-((a - at_g) / ac_g)**3)

            return D_a + C_g / a * (a / at_g)**alpha_g * F_a * cutoff
        
        return size_dist

    @staticmethod
    #@njit
    def shape_dist(a, beta_tab):
        '''
        A toy model for the distribution of the grain shape parameter, beta. 
        Note that this is a conditional distribution, given the grain size.
        We assume a Gaussian distribution apodized by a cosine function.
        '''
        if a <= a2:
            beta_min = cylindrical_params(a, d)[1]
            sigma = 0.1
            # Calculate the distribution
            dist = np.zeros_like(beta_tab)
            for i, beta in enumerate(beta_tab):
                dist[i] = apodized_Gaussian_dist(beta, beta_min, sigma, right_half=True)
            # Normalize the distribution
            dist /= np.sum(dist)
            return dist
        else:
            sigma = 0.025
            dist = np.zeros_like(beta_tab)
            for i, beta in enumerate(beta_tab):
                dist[i] = apodized_Gaussian_dist(beta, 0, sigma)
            dist /= np.sum(dist)
            return dist

    
    @staticmethod
    #@njit 
    def shape_dist_fixed_thickness(a, beta_tab):
        '''
        The distribution of the grain shape parameter, beta, if we fix the thickness of the disk as d. 
        Note that this is a conditional distribution, given the grain size.
        We assume a Gaussian distribution apodized by a cosine function.
        '''
        if a <= a2:
            beta = cylindrical_params(a, d)[1]
            dist = np.zeros_like(beta_tab)
            # find the index of the element closest to beta in beta_tab
            idx = np.abs(beta_tab - beta).argmin()
            dist[idx] = 1
            return dist
        else:
            dist = np.zeros_like(beta_tab)
            # find the index of the element closest to 0 in beta_tab
            idx = np.abs(beta_tab).argmin()
            dist[idx] = 1
            return dist

    def shape_and_size_dist(self, line, a_weighted=True, normalize=False, Nbeta=5, fixed_thickness=False):
        '''
        Calculate the joint distribution of the grain shape parameter, beta, and the grain size, a.

        Parameters:
        -----------
        line: int
            The line of the size distribution table.
        
        a_weighted: bool
            If True, the grain size distribution is weighted and we get f(a)da;
            otherwise, we get f(a).

        Returns:
        --------
        A 2D array with the joint distribution.
        '''
        size_func = self.size_dist_func(line)
        beta_tab = np.linspace(-0.49, 0.1, Nbeta)
        result = np.zeros((len(self.a_tab), len(beta_tab)))
        a_dist = np.zeros_like(self.a_tab)
        for ai, a in enumerate(self.a_tab):
            a_dist[ai] = size_func(a)
        if a_weighted:
            a_dist = a_dist * self.Da_over_a * self.a_tab
        if normalize:
            a_dist /= np.sum(a_dist)
        self.a_dist = a_dist

        if fixed_thickness:
            for ai, a in enumerate(self.a_tab):
                result[ai,:] = self.shape_dist_fixed_thickness(a, beta_tab) * a_dist[ai]
        else:
            for ai, a in enumerate(self.a_tab):
                result[ai,:] = a_dist[ai] * self.shape_dist(a, beta_tab)

        aux = np.sum(result, axis=0)
        ind = aux > 1e-500 # Avoid beta values with very small probability

        self.beta_tab = beta_tab[ind]
        return result[:, ind] 

        
