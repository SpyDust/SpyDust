from ast import arg
import numpy as np
from scipy import integrate
from .util import ParallelInterpolator, ParallelBase, interp_func_1d
# Use functools.partial to pass extra arguments
from functools import partial


class ModeBase():
    @staticmethod
    def Omega_mapping(omega, beta, cos_theta_b):
        '''
        The mapping function from the observational frequency to the rotation frequency.
        Parameters
        ----------
        omega: float
            The observational frequency.

        Returns
        -------
        Omega: float
            The rotation frequency.
        '''
        pass

    @staticmethod
    def _dOmega_over_domega(beta, cos_theta_b):
        pass

    @staticmethod
    def in_plane_part(u):
        ''' u = mu_perp^2 / mu^2 is the ratio of the in-plane part of mu^2 '''
        pass

    @staticmethod
    def int_alignm_part(cos_theta_b):
        pass
    
    @staticmethod
    def _stokes_integrand(cos_theta_L, phi_L):
        pass

    @classmethod
    def int_integrand(cls, cos_theta_b, omegas, beta, int_dist_func, rot_dis_func):
        rot_Omegas = cls.Omega_mapping(omegas, beta, cos_theta_b)
        rot_dis_func_values = rot_dis_func(rot_Omegas)
        factor = cls.int_alignm_part(cos_theta_b) * cls._dOmega_over_domega(beta, cos_theta_b) * int_dist_func(cos_theta_b) 
        return factor * rot_dis_func_values

    @classmethod
    def int_integral(cls, omegas, beta, int_dist_func, rot_dis_func, impulse=None, normalise_int_dist=False, nworkers=None):
        '''
        The auxiliary function for the integral with respect to the internal alignment, theta_b.
        Parameters
        ----------
        beta: float
            The grain shape parameter.
        dist_func: function
            The distribution function of cos_theta_b.
        Returns
        -------
        float
            The value of the integral.
        '''

        if impulse is None:

            # Integrate disc_func over cos_theta_b from -1 to 1 to get the normalization factor;
            if normalise_int_dist:
                norm, _ = integrate.quad(int_dist_func, -1, 1, epsabs=1e-6, epsrel=1e-6)
            else:
                norm = 1.

            # Adaptive integration with error control
            # result, _ = integrate.quad_vec(cls.int_integrand, -1, 1, # epsabs=1e-6, epsrel=1e-6, 
            #                                workers=nworkers,
            #                                args=(omegas, beta, int_dist_func, rot_dis_func))

            # Handle singularity at x=0 using weighted quadrature
            result_part1, _ = integrate.quad_vec(
                cls.int_integrand, -1, 0,
                #epsabs=1e-6, epsrel=1e-6,
                workers=nworkers,
                args=(omegas, beta, int_dist_func, rot_dis_func),
                #weight='alg', wvar=(0, -1)  # Right endpoint singularity (x=0) with 1/x behavior
            )
            
            result_part2, _ = integrate.quad_vec(
                cls.int_integrand, 0, 1,
                #epsabs=1e-6, epsrel=1e-6,
                workers=nworkers,
                args=(omegas, beta, int_dist_func, rot_dis_func),
                #weight='alg', wvar=(-1, 0)  # Left endpoint singularity (x=0) with 1/x behavior
            )
            
            result = result_part1 + result_part2
            return result / norm

        else:
            # Evaluste cls.int_integrand at the impulse point
            # The int_dist_func is not used is this case.
            def func(x):
                return 1.
            return cls.int_integrand(impulse, omegas, beta, func, rot_dis_func)


    @classmethod
    def vector_integrand_1(cls, phi_L, cos_theta_L, dist_func):
        return cls._stokes_integrand(cos_theta_L, phi_L) * dist_func(cos_theta_L, phi_L) 
    
    @classmethod
    def vector_integrand_2(cls, cos_theta_L, dist_func):
        # Integrate the Stokes parts over phi_L (from 0 to 2pi) ;
        res, _ = integrate.quad_vec(
                    cls.vector_integrand_1,
                    0, 2*np.pi,
                    workers=1,  #if -1, Use all available cores
                    args=(cos_theta_L, dist_func)
                    )
        return res 

    @classmethod
    def ext_integral(cls, dist_func, normalised=False, full_Stokes=True, nworkers=-1):
        '''
        The auxiliary function for the integral with respect to the external alignment, (cos_theta_L, phi_L).
        Parameters
        ----------
        dist_func: function
            The distribution function: f(cos_theta_L, phi_L).
        Returns
        -------
        float
            The value of the integral.
        '''
        # Integrate disc_func over cos_theta_L (from -1 to 1) and phi_L (from 0 to 2pi) to get the normalization factor;
        if normalised:
            norm = 1.
        else:
            # norm, _ = integrate.dblquad(dist_func, -1, 1, 0, 2*np.pi)
            norm, _ = integrate.nquad(
                                    dist_func,
                                    [[-1, 1], [0, 2*np.pi]],  # (cos_theta_L, phi_L) bounds
                                    # opts={'epsabs': 1e-6, 'epsrel': 1e-6}
                                    )

        if full_Stokes:

            # Outer integral aggregates all components
            result, _ = integrate.quad_vec(
                cls.vector_integrand_2,
                -1, 1,
                workers=nworkers,
                args=(dist_func,)
                # epsabs=1e-6, 
                # epsrel=1e-6
            )
            
            # Reorder components to I, Q, U, V
            return tuple(result / norm)
        else:
            # Only get Stokes I
            result, _ = integrate.nquad(
                                lambda cos_theta_L, phi_L: (
                                    cls._stokes_integrand(cos_theta_L, phi_L)[0] * dist_func(cos_theta_L, phi_L) / norm
                                ),
                                [[-1, 1], [0, 2*np.pi]],  # Integration bounds (cos_theta_L, phi_L)
                                #opts={'epsabs': 1e-6, 'epsrel': 1e-6}
                            )
            return result

    # def _mapping_rot_dist_interp(self, obs_omegas, grid_Omega_points, grid_rot_dist, max_workers=None):
    #     """This function generates the corresponding rotation distribution function values given observational frequencies using interpolation."""
    #     Omegas = self.Omega_mapping(obs_omegas).reshape(-1, 1) # Reshape to 2D array
    #     with ParallelInterpolator(grid_rot_dist, (grid_Omega_points,), max_workers=max_workers) as interpolator:
    #         results = interpolator(Omegas)
    #     return np.array(results).reshape(-1)

    # def _mapping_rot_dist(self, obs_omegas, rot_dist_func, max_workers=None):
    #     """This function generates the corresponding rotation distribution function values given observational frequencies with a given function."""
    #     Omegas = self.Omega_mapping(obs_omegas, beta, cos_theta_b)
    #     with ParallelBase(max_workers=max_workers) as parallel:
    #         results = parallel._parallel_execute(rot_dist_func, Omegas)
    #     return np.array(results)

    def generate_SED(self, omegas, beta, ip, 
                     internal_dist=None, 
                     impulse=None,
                     external_dist=None, 
                     rot_dist_func=None, 
                     max_workers=None, 
                     full_Stokes=True, ):
        """
        This function generates the average (expected) SED (divided by mu^2) of the single grain.
        """
        
        internal_part = self.int_integral(omegas, beta, internal_dist, rot_dist_func, impulse=impulse,
                                        normalise_int_dist=True, nworkers=max_workers)
        
        aux = omegas**4 * self.in_plane_part(ip) * internal_part 

        polarisation_part = np.array(self.ext_integral(external_dist, normalised=False, full_Stokes=full_Stokes, nworkers=max_workers) ).reshape(-1)

        if full_Stokes:
            result = polarisation_part[:, np.newaxis] * aux[np.newaxis, :]
        else:
            result = polarisation_part * aux

        return result


class mode1(ModeBase):
    @staticmethod
    def Omega_mapping(omega, beta, cos_theta_b):
        return omega

    @staticmethod
    def _dOmega_over_domega(beta, cos_theta_b):
        return 1.

    @staticmethod
    def in_plane_part(u):
        return 1. - u

    @staticmethod
    def int_alignm_part(cos_theta_b):
        return 1. - cos_theta_b**2

    @staticmethod
    def _stokes_integrand(cos_theta_L, phi_L):
        I = 2. * cos_theta_L**2 + 2.
        Q = -2. * (1 - cos_theta_L**2) * np.cos(2*phi_L)
        U = 2. * (1 - cos_theta_L**2) * np.sin(2*phi_L)
        V = -4. * cos_theta_L
        return np.array([I, Q, U, V])


class mode2and3(ModeBase):
    @staticmethod
    def Omega_mapping(omega, beta, cos_theta_b):
        return omega / np.abs(1. + beta * cos_theta_b)

    @staticmethod
    def _dOmega_over_domega(beta, cos_theta_b):
        return 1. / np.abs(1. + beta * cos_theta_b) 
    
    @staticmethod
    def in_plane_part(u):
        ''' u = mu_perp^2 / mu^2 is the ratio of the in-plane part of mu^2 '''
        return u

    @staticmethod
    def int_alignm_part(cos_theta_b):
        return (1. + cos_theta_b)**2 / 4.

    @staticmethod
    def _stokes_integrand(cos_theta_L, phi_L):
        I = 2. * cos_theta_L**2 + 2.
        Q = -2. * (1 - cos_theta_L**2) * np.cos(2*phi_L)
        U = 2. * (1 - cos_theta_L**2) * np.sin(2*phi_L)
        V = -4. * cos_theta_L
        return np.array([I, Q, U, V]) 

    @classmethod
    def int_integrand(cls, cos_theta_b, omegas, beta, int_dist_func, rot_dis_func):
        rot_Omegas = cls.Omega_mapping(omegas, beta, cos_theta_b)
        rot_dis_func_values = rot_dis_func(rot_Omegas)
        factor = cls.int_alignm_part(cos_theta_b) * cls._dOmega_over_domega(beta, cos_theta_b) * (int_dist_func(cos_theta_b) + int_dist_func(-cos_theta_b)) 
        return factor * rot_dis_func_values

    
class mode4(ModeBase):
    @staticmethod
    def Omega_mapping(omega, beta, cos_theta_b):
        return omega / np.abs(beta * cos_theta_b)

    @staticmethod
    def _dOmega_over_domega(beta, cos_theta_b):
        return 1. / np.abs(beta * cos_theta_b) 
    
    @staticmethod
    def in_plane_part(u):
        ''' u = mu_perp^2 / mu^2 is the ratio of the in-plane part of mu^2 '''
        return u

    @staticmethod
    def int_alignm_part(cos_theta_b):
        return (1. + cos_theta_b)**2 / 4.

    @staticmethod
    def _stokes_integrand(cos_theta_L, phi_L):
        I = 1. - cos_theta_L**2 
        Q = (1. - cos_theta_L**2) * np.cos(2*phi_L)
        U = - (1. - cos_theta_L**2) * np.sin(2*phi_L)
        V = 0.
        return np.array([I, Q, U, V]) 

class full_Stokes_SED():
    def __init__(self, rot_dist_func=None, rot_freq_grid=None, rot_dist_grid=None, log_freq=True, log_dist=True):
        self.mode1 = mode1()
        self.mode2and3 = mode2and3()
        self.mode4 = mode4()
        if rot_dist_func is not None:
            self.rot_dist_func = rot_dist_func
        else:
            assert rot_freq_grid is not None and rot_dist_grid is not None, "Please provide the rotation distribution function or the grid points."
            self.rot_dist_func = interp_func_1d(rot_freq_grid, rot_dist_grid, log_coord=log_freq, log_data=log_dist)

    def generate_SED(self, omegas, beta, ip, mu_sq,
                     internal_dist=None, 
                     impulse=None,
                     external_dist=None, 
                     max_workers=None, 
                     full_Stokes=True):
        """This function generates the average (expected) SED of the single grain"""
        # Prevent nested multiprocessing by setting inner workers to 1
        # inner_workers = 1 if max_workers and max_workers > 1 else max_workers
        inner_workers = max_workers 
        
        result = (self.mode1.generate_SED(omegas, beta, ip, 
                 internal_dist=internal_dist, 
                 impulse=impulse,
                 external_dist=external_dist, 
                 rot_dist_func=self.rot_dist_func,
                 max_workers=inner_workers,  
                 full_Stokes=full_Stokes) 
                + self.mode2and3.generate_SED(omegas, beta, ip, 
                    internal_dist=internal_dist, 
                    impulse=impulse,
                    external_dist=external_dist, 
                    rot_dist_func=self.rot_dist_func,
                    max_workers=inner_workers,  
                    full_Stokes=full_Stokes) 
                + self.mode4.generate_SED(omegas, beta, ip, 
                    internal_dist=internal_dist,
                    impulse=impulse,
                    external_dist=external_dist,
                    rot_dist_func=self.rot_dist_func,
                    max_workers=inner_workers,  
                    full_Stokes=full_Stokes))
        result *= mu_sq
        return result


