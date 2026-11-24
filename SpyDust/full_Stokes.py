from functools import partial

import numpy as np
from scipy import integrate
from scipy.interpolate import PchipInterpolator
from .util import homogeneous_dist


def _transverse_stokes(cos_theta_L, phi_L):
    """Directional factors of the rotating dipole (SpyDust Eqs. 7–9)."""
    sin_sq = 1.0 - cos_theta_L**2
    return np.array([1.0 + cos_theta_L**2, -sin_sq * np.cos(2*phi_L),
                     sin_sq * np.sin(2*phi_L), -2.0*cos_theta_L])


class ModeBase():
    @classmethod
    def Omega_mapping(cls, omega, beta, cos_theta_b):
        """Rotation frequency associated with a nonzero radiation harmonic."""
        return omega / cls._frequency_ratio(beta, cos_theta_b)

    @classmethod
    def _dOmega_over_domega(cls, beta, cos_theta_b):
        return 1.0 / cls._frequency_ratio(beta, cos_theta_b)

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
    def _harmonic(cls, cos_theta_b, omegas, beta, rot_dis_func):
        ratio = cls._frequency_ratio(beta, cos_theta_b)
        weight = cls.int_alignm_part(cos_theta_b)
        if np.ndim(ratio) == 0 and np.ndim(weight) == 0:
            if ratio == 0. or weight == 0.:
                return np.zeros_like(omegas, dtype=float)
            return weight * rot_dis_func(omegas / ratio) / ratio
        # Public integrands support broadcast angles and observing frequencies.
        ratio, weight, omega = np.broadcast_arrays(ratio, weight, omegas)
        active = (ratio != 0.) & (weight != 0.)
        value = np.zeros_like(omega, dtype=float)
        if np.any(active):
            value[active] = weight[active] * rot_dis_func(omega[active]/ratio[active]) / ratio[active]
        return value

    @classmethod
    def int_integrand(cls, cos_theta_b, omegas, beta, int_dist_func, rot_dis_func):
        return int_dist_func(cos_theta_b) * cls._harmonic(cos_theta_b, omegas, beta, rot_dis_func)

    @classmethod
    def int_integral(cls, omegas, beta, int_dist_func, rot_dis_func, impulse=None,
                     normalise_int_dist=False, nworkers=None):
        """Average over cos(theta_b), or evaluate a unit point mass at impulse."""
        if impulse is not None:
            if not -1 <= impulse <= 1:
                raise ValueError("impulse must lie in [-1, 1].")
            return cls._harmonic(impulse, omegas, beta, rot_dis_func)

        if int_dist_func is None:
            int_dist_func = homogeneous_dist
        norm = 1.0
        if normalise_int_dist:
            norm, _ = integrate.quad(int_dist_func, -1, 1)

        if isinstance(rot_dis_func, _TabulatedRotationPDF) and beta != 0 and cls is not mode1:
            return rot_dis_func.angle_integral(cls, omegas, beta, int_dist_func, nworkers) / norm

        result, _ = integrate.quad_vec(
            partial(cls.int_integrand, omegas=omegas, beta=beta, int_dist_func=int_dist_func,
                    rot_dis_func=rot_dis_func), -1, 1, points=[0.0],
            workers=1 if nworkers is None else nworkers,
        )
        return result / norm

    @classmethod
    def vector_integrand_1(cls, phi_L, cos_theta_L, dist_func):
        return cls._stokes_integrand(cos_theta_L, phi_L) * dist_func(cos_theta_L, phi_L) 
    
    @classmethod
    def vector_integrand_2(cls, cos_theta_L, dist_func):
        # Integrate the Stokes parts over phi_L (from 0 to 2pi) ;
        res, _ = integrate.quad_vec(
                    partial(cls.vector_integrand_1, cos_theta_L=cos_theta_L, dist_func=dist_func),
                    0, 2*np.pi, workers=1
                    )
        return res 

    @classmethod
    def ext_integral(cls, dist_func, normalised=False, full_Stokes=True, nworkers=None):
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
        if dist_func is None:
            dist_func = homogeneous_dist
        nworkers = 1 if nworkers is None else nworkers
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
                partial(cls.vector_integrand_2, dist_func=dist_func),
                -1, 1, workers=nworkers
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
        
        omegas = np.atleast_1d(np.asarray(omegas, dtype=float))
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
    def _frequency_ratio(beta, cos_theta_b):
        return 1.0

    @staticmethod
    def in_plane_part(u):
        return 1. - u

    @staticmethod
    def int_alignm_part(cos_theta_b):
        return 1. - cos_theta_b**2

    _stokes_integrand = staticmethod(_transverse_stokes)


class mode2and3(ModeBase):
    @staticmethod
    def _frequency_ratio(beta, cos_theta_b):
        return abs(1.0 + beta * cos_theta_b)

    @staticmethod
    def in_plane_part(u):
        ''' u = mu_perp^2 / mu^2 is the ratio of the in-plane part of mu^2 '''
        return u

    @staticmethod
    def int_alignm_part(cos_theta_b):
        return (1. + cos_theta_b)**2 / 4.

    _stokes_integrand = staticmethod(_transverse_stokes)

    @classmethod
    def _harmonic(cls, cos_theta_b, omegas, beta, rot_dis_func):
        # Mode 3 equals mode 2 reflected at cos(theta_b) -> -cos(theta_b).
        mode2 = super()._harmonic(cos_theta_b, omegas, beta, rot_dis_func)
        mode3 = super()._harmonic(-cos_theta_b, omegas, beta, rot_dis_func)
        return mode2 + mode3

    @classmethod
    def int_integrand(cls, cos_theta_b, omegas, beta, int_dist_func, rot_dis_func):
        # Reflect the integration variable in mode 3 so both PDFs are sampled
        # at the same rotation frequency, as in the historical integrand.
        angular_pdf = int_dist_func(cos_theta_b) + int_dist_func(-cos_theta_b)
        return angular_pdf * super()._harmonic(cos_theta_b, omegas, beta, rot_dis_func)


class mode4(ModeBase):
    @staticmethod
    def _frequency_ratio(beta, cos_theta_b):
        return abs(beta * cos_theta_b)

    @staticmethod
    def in_plane_part(u):
        ''' u = mu_perp^2 / mu^2 is the ratio of the in-plane part of mu^2 '''
        return u

    @staticmethod
    def int_alignm_part(cos_theta_b):
        return 1.0 - cos_theta_b**2

    @staticmethod
    def _stokes_integrand(cos_theta_L, phi_L):
        I = 1. - cos_theta_L**2 
        Q = (1. - cos_theta_L**2) * np.cos(2*phi_L)
        U = - (1. - cos_theta_L**2) * np.sin(2*phi_L)
        V = 0.
        return np.array([I, Q, U, V]) 

class _TabulatedRotationPDF:
    """Interpolate within the supplied support, including zero-density gaps."""

    def __init__(self, freq, density, log_freq, log_density):
        x = np.asarray(freq, dtype=float)
        y = np.asarray(density, dtype=float)
        if (x.ndim != 1 or not x.size or x.shape != y.shape or not np.isfinite(x).all()
                or np.any(np.diff(x) <= 0.) or (not log_freq and np.any(x <= 0.))):
            raise ValueError("Provide matching rotation arrays with increasing finite frequencies.")
        if np.isnan(y).any() or np.isposinf(y).any() or (not log_density and np.any(y < 0.)):
            raise ValueError("The rotation probability density must be finite and nonnegative.")
        self.log_freq = log_freq
        self.log_density = log_density
        # log(0) is a valid boundary or gap on a log-density grid.
        active = np.isfinite(y) if log_density else y > 0.
        edges = np.flatnonzero(np.diff(np.r_[False, active, False]))
        self.segments = []
        # Linear densities interpolate through their zeros. Use the same PCHIP
        # on each band so derivatives at a zero match the full input table.
        linear_interp = PchipInterpolator(x, y) if not log_density and x.size > 1 else None
        for start, end in edges.reshape(-1, 2):
            if not log_density:
                # Linear densities interpolate to the first zero on each side.
                start, end = max(0, start-1), min(len(x), end+1)
                interp = linear_interp
            else:
                interp = PchipInterpolator(x[start:end], y[start:end]) if end - start > 1 else None
            self.segments.append((x[start], x[end-1], interp, y[start]))

    def __call__(self, omega):
        with np.errstate(divide="ignore"):
            q = np.log(omega) if self.log_freq else np.asarray(omega)
        result = np.zeros_like(q, dtype=float)
        for low, high, interp, first_value in self.segments:
            # Unit conversions can move a query off a tabulated endpoint by an ulp.
            eps = 4 * np.finfo(float).eps
            use = (q >= low - eps * max(1., abs(low))) & (q <= high + eps * max(1., abs(high)))
            if np.any(use):
                values = first_value if interp is None else interp(np.clip(q[use], low, high))
                result[use] = np.exp(values) if self.log_density else values
        return result

    def _density_at_log_frequency(self, z, band):
        low, high, interp, _ = band
        coord = z if self.log_freq else np.exp(z)
        values = interp(np.clip(coord, low, high))
        return np.exp(values) if self.log_density else values

    @staticmethod
    def _paired_alignment(angular_pdf, x):
        if angular_pdf is homogeneous_dist:
            return 2.
        # User-supplied angular densities historically accept scalar arguments.
        if x.size == 1:
            return angular_pdf(x[0]) + angular_pdf(-x[0])
        try:
            paired = angular_pdf(x) + angular_pdf(-x)
            if np.shape(paired) == x.shape:
                return paired
        except (TypeError, ValueError):
            pass
        return np.array([angular_pdf(v) + angular_pdf(-v) for v in x])

    def _integrand4(self, t, *, start, span, threshold, beta, band, angular_pdf):
        z = start + span*t
        log_x = np.minimum(threshold - z, 0.)
        angle = self._paired_alignment(angular_pdf, np.exp(log_x))
        return (span/abs(beta)) * -np.expm1(2.*log_x) * angle * self._density_at_log_frequency(z, band)

    def _integrand23(self, t, *, start, span, log_omega, sign, beta, band, angular_pdf):
        # u = log(Omega/omega) and |dx/du| = |1 + beta*x| / |beta|.
        # expm1 keeps the internal angle accurate in the spherical limit.
        u = start + span*t
        x = np.expm1(-u)/beta if sign > 0 else (-np.exp(-u)-1.)/beta
        x = np.clip(x, -1., 1.)
        angle = self._paired_alignment(angular_pdf, x)
        return (span / abs(beta) * mode2and3.int_alignm_part(x) * angle
                * self._density_at_log_frequency(log_omega+u, band))

    def angle_integral(self, mode, omegas, beta, angular_pdf, nworkers):
        """Integrate over each tabulated interpolation cell and its angular support."""
        omegas = np.atleast_1d(omegas)
        result = np.zeros_like(omegas, dtype=float)
        workers = 1 if nworkers is None else nworkers
        positive = np.flatnonzero(omegas > 0.)
        b = abs(beta)
        for band in self.segments:
            low, high, interp, _ = band
            if low == high:
                continue
            knots = interp.x[(interp.x >= low) & (interp.x <= high)]
            if self.log_density:
                # A finite log-PDF can fall below floating-point density range
                # inside one cell. Expose that edge so its thin representable
                # part is sampled even when every full-cell node underflows.
                floor = np.log(np.finfo(float).tiny)
                heights = interp(knots)
                if heights.min() < floor < heights.max():
                    crossings = interp.solve(floor, extrapolate=False)
                    knots = np.union1d(knots, crossings[(crossings > low) & (crossings < high)])
            log_knots = knots if self.log_freq else np.log(knots)
            # Vectorize over a bounded number of (observing frequency, cell) pairs.
            # Exposing the cell edges prevents quadrature from skipping a thin
            # peak even when a positive background connects it to the whole PDF.
            batch = max(1, 16384 // (len(knots)-1))
            for pos in range(0, positive.size, batch):
                indices = positive[pos:pos+batch]
                log_w = np.log(omegas[indices])
                if mode is mode4:
                    threshold = log_w - np.log(b)
                    left = np.maximum(log_knots[:-1], threshold[:, None])
                    span = log_knots[1:] - left
                    rows, cols = np.nonzero(span > 0.)
                    if not rows.size:
                        continue
                    integrand = partial(self._integrand4, start=left[rows, cols],
                                        span=span[rows, cols], threshold=threshold[rows],
                                        beta=beta, band=band, angular_pdf=angular_pdf)
                    values = integrate.quad_vec(integrand, 0., 1., workers=workers,
                                                norm="max", epsabs=np.finfo(float).tiny)[0]
                    result[indices] += np.bincount(rows, weights=values, minlength=len(indices))
                    continue

                # The +/- branches correspond to the sign of 1+beta*x.
                edges = log_knots - log_w[:, None]
                branches = [(1, -np.log1p(b), -np.log1p(-b) if b < 1. else np.inf)]
                if b > 1.:
                    branches.append((-1, -np.log(b-1.), np.inf))
                for sign, u_min, u_max in branches:
                    left = np.maximum(edges[:, :-1], u_min)
                    span = np.minimum(edges[:, 1:], u_max) - left
                    rows, cols = np.nonzero(span > 0.)
                    if not rows.size:
                        continue
                    integrand = partial(self._integrand23, start=left[rows, cols],
                                        span=span[rows, cols], log_omega=log_w[rows],
                                        sign=sign, beta=beta, band=band, angular_pdf=angular_pdf)
                    values = integrate.quad_vec(integrand, 0., 1., workers=workers,
                                                norm="max", epsabs=np.finfo(float).tiny)[0]
                    result[indices] += np.bincount(rows, weights=values, minlength=len(indices))
        return result


class full_Stokes_SED():
    def __init__(self, rot_dist_func=None, rot_freq_grid=None, rot_dist_grid=None, log_freq=True, log_dist=True):
        self.mode1 = mode1()
        self.mode2and3 = mode2and3()
        self.mode4 = mode4()
        if rot_dist_func is not None:
            self.rot_dist_func = rot_dist_func
        else:
            if rot_freq_grid is None or rot_dist_grid is None:
                raise ValueError("Provide a rotation distribution function or both grid arrays.")
            # A gridded PDF has finite support. A callable can supply a physical tail.
            self.rot_dist_func = _TabulatedRotationPDF(rot_freq_grid, rot_dist_grid, log_freq, log_dist)

    def generate_SED(self, omegas, beta, ip, mu_sq,
                     internal_dist=None, 
                     impulse=None,
                     external_dist=None, 
                     max_workers=None, 
                     full_Stokes=True):
        """Return I or [I, Q, U, V] for the supplied angular frequencies.

        Omitted angular distributions are isotropic. An impulse selects a
        fixed cos(theta_b). As in the original Stokes interface the overall
        radiation factor c**(-3) is omitted.
        """
        result = sum(
            mode.generate_SED(omegas, beta, ip, internal_dist=internal_dist,
                              impulse=impulse, external_dist=external_dist,
                              rot_dist_func=self.rot_dist_func, max_workers=max_workers,
                              full_Stokes=full_Stokes)
            for mode in (self.mode1, self.mode2and3, self.mode4)
        )
        return mu_sq * result
