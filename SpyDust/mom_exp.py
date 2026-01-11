import numpy as np
from .SpyDust import SpyDust_given_grain_size_shape


class MC_moment_expansion:
    nh = 3e2
    T = 20
    chi = 0.01
    xh = 0.0
    y = 0.99

    def __init__(self, pivots, order=2, step_sizes=None, default_freqs=None):
        pivots = np.asarray(pivots, dtype=float)
        if pivots.shape != (3,):
            raise ValueError("pivots must contain three values: (log_a, beta, log_xC).")
        self.param_names = ("log_a", "beta", "log_xC")
        self.pivot_a, self.pivot_beta, self.pivot_xC = pivots
        self.pivots = pivots
        self.order = self._validate_order(order)
        if step_sizes is None:
            step_sizes = self._default_step_sizes()
        else:
            step_sizes = np.asarray(step_sizes, dtype=float)
            if step_sizes.shape != (3,):
                raise ValueError("step_sizes must contain three positive values.")
            if np.any(step_sizes <= 0):
                raise ValueError("step_sizes must be strictly positive.")
        self.step_sizes = step_sizes
        if default_freqs is None:
            self.default_freqs = None
        else:
            default_freqs = np.asarray(default_freqs, dtype=float)
            if default_freqs.ndim != 1 or default_freqs.size == 0:
                raise ValueError("default_freqs must be a non-empty one-dimensional array.")
            self.default_freqs = default_freqs
        self.last_basis = None

    def _validate_order(self, order):
        if not isinstance(order, (int, np.integer)):
            raise ValueError("order must be an integer.")
        if order < 0 or order > 2:
            raise ValueError("order must be between 0 and 2.")
        return int(order)

    def _default_step_sizes(self):
        beta_scale = max(5e-3, 5e-3 * abs(self.pivot_beta))
        return np.array([5e-3, beta_scale, 5e-3], dtype=float)

    def _resolve_basis(self, freqs, basis, tumbling):
        """Return a basis compatible with the requested frequencies."""
        if basis is not None:
            return basis
        regenerate = False
        if self.last_basis is None:
            regenerate = True
        elif freqs is not None:
            freq_vec = np.asarray(freqs, dtype=float)
            if freq_vec.ndim != 1 or freq_vec.size == 0:
                raise ValueError("freqs must be a non-empty one-dimensional array.")
            if not np.array_equal(freq_vec, np.asarray(self.last_basis["freqs"], dtype=float)):
                regenerate = True
        if regenerate:
            basis = self.generate_SED_basis(freqs=freqs, tumbling=tumbling)
        else:
            basis = self.last_basis
        return basis

    def _assemble_design_matrix(self, basis, max_order, include_zeroth):
        """Construct a design matrix up to max_order from the cached basis."""
        max_order = int(max_order)
        columns = []
        labels = []
        if include_zeroth:
            if "zeroth" not in basis:
                raise ValueError("Basis is missing the zeroth-order SED.")
            columns.append(np.asarray(basis["zeroth"], dtype=float))
            labels.append("zeroth")
        if max_order >= 1:
            first = basis.get("first")
            if first is None:
                raise ValueError("First-order derivatives are not available in the basis.")
            for name in self.param_names:
                if name not in first:
                    raise KeyError(f"Missing first-order derivative for parameter '{name}'.")
                columns.append(np.asarray(first[name], dtype=float))
                labels.append(f"first:{name}")
        if max_order >= 2:
            second = basis.get("second")
            if second is None or "diag" not in second or "cross" not in second:
                raise ValueError("Second-order derivatives are not available in the basis.")
            diag = second["diag"]
            cross = second["cross"]
            for name in self.param_names:
                if name not in diag:
                    raise KeyError(f"Missing second-order diagonal term for parameter '{name}'.")
                columns.append(np.asarray(diag[name], dtype=float))
                labels.append(f"second_diag:{name}")
            for i in range(len(self.param_names)):
                for j in range(i + 1, len(self.param_names)):
                    key = (self.param_names[i], self.param_names[j])
                    if key not in cross:
                        raise KeyError(f"Missing second-order cross term for parameters {key}.")
                    columns.append(np.asarray(cross[key], dtype=float))
                    labels.append(f"second_cross:{key[0]}:{key[1]}")
        if not columns:
            raise ValueError("Requested basis does not add any columns. Adjust order or include_zeroth.")
        design_matrix = np.column_stack(columns)
        return design_matrix, labels

    def generate_SED(self, log_a, beta, log_xC, freq_coords=None, tumbling=True):
        """Generate an SED for a single grain configuration using SpyDust."""
        log_a = float(log_a)
        beta = float(beta)
        log_xC = float(log_xC)

        if freq_coords is not None:
            target_freqs = np.asarray(freq_coords, dtype=float)
            if target_freqs.ndim != 1:
                raise ValueError("freq_coords must be a one-dimensional array of frequencies.")
            if target_freqs.size == 0:
                raise ValueError("freq_coords cannot be empty.")
            span = max(target_freqs.max() - target_freqs.min(), 1.0)
            pad = max(0.01 * span, 0.05)
            min_freq = max(target_freqs.min() - pad, 1e-3)
            max_freq = target_freqs.max() + pad
            n_freq = max(300, int(np.ceil(target_freqs.size * 1.5)))
        else:
            target_freqs = None
            min_freq = None
            max_freq = None
            n_freq = 300

        a = np.exp(log_a)
        xC = np.exp(log_xC)
        if not np.isfinite(a) or not np.isfinite(xC):
            raise ValueError("Provided log parameters lead to non-finite grain properties.")

        a2_aux = 2.0 * a if tumbling else a / 2.0
        env = {
            'nh': self.nh,
            'T': self.T,
            'Chi': self.chi,
            'xh': self.xh,
            'xC': xC,
            'y': self.y,
            'gamma': 0,
            'dipole': 9.3,
        }
        freqs_native, sed_native = SpyDust_given_grain_size_shape(
            env,
            a,
            beta,
            tumbling=tumbling,
            min_freq=min_freq,
            max_freq=max_freq,
            n_freq=n_freq,
            N_angular_Omega=500,
            a2=a2_aux,
        )
        freqs_native = np.asarray(freqs_native, dtype=float)
        sed_native = np.asarray(sed_native, dtype=float)

        if target_freqs is None:
            return freqs_native, sed_native

        if not np.all(np.diff(freqs_native) >= 0):
            sort_idx = np.argsort(freqs_native)
            freqs_native = freqs_native[sort_idx]
            sed_native = sed_native[sort_idx]

        tol = 5e-4 * max(1.0, freqs_native.max())
        if target_freqs.min() < freqs_native.min() - tol or target_freqs.max() > freqs_native.max() + tol:
            raise ValueError("freq_coords must lie within the native SpyDust frequency grid.")

        if not np.all(np.diff(target_freqs) >= 0):
            order_idx = np.argsort(target_freqs)
            sorted_target = target_freqs[order_idx]
            interpolated = np.interp(sorted_target, freqs_native, sed_native)
            reverse_idx = np.argsort(order_idx)
            interpolated = interpolated[reverse_idx]
            return target_freqs, interpolated

        interpolated = np.interp(target_freqs, freqs_native, sed_native)
        return target_freqs, interpolated

    def generate_SED_basis(self, pivots=None, freqs=None, order=2, tumbling=True):
        if pivots is None:
            pivot_vec = self.pivots
        else:
            pivot_vec = np.asarray(pivots, dtype=float)
            if pivot_vec.shape != (3,):
                raise ValueError("pivots must contain three values: (log_a, beta, log_xC).")
        if freqs is None:
            if self.default_freqs is None:
                raise ValueError("freqs cannot be None when constructing the SED basis.")
            freq_coords = self.default_freqs
        else:
            freq_coords = np.asarray(freqs, dtype=float)
            if freq_coords.ndim != 1 or freq_coords.size == 0:
                raise ValueError("freqs must be a one-dimensional array of frequencies.")

        target_order = self.order
        if order is not None:
            target_order = min(self._validate_order(order), target_order)

        freq_grid, sed0 = self.generate_SED(*pivot_vec, freq_coords=freq_coords, tumbling=tumbling)
        freq_grid = np.asarray(freq_grid, dtype=float)
        sed0 = np.asarray(sed0, dtype=float)

        step_sizes = self.step_sizes
        unit_vectors = np.eye(3, dtype=int)
        cache = {(0, 0, 0): sed0.copy()}

        def evaluate(multipliers):
            # Cache SED evaluations keyed by finite-difference multipliers.
            key = tuple(int(m) for m in multipliers)
            if key in cache:
                return cache[key]
            delta = step_sizes * np.array(key, dtype=float)
            params = pivot_vec + delta
            freq_eval, sed_eval = self.generate_SED(*params, freq_coords=freq_coords, tumbling=tumbling)
            if not np.allclose(freq_eval, freq_grid, rtol=0, atol=1e-6):
                raise ValueError("Frequency grid mismatch encountered while constructing the basis.")
            cache[key] = sed_eval
            return sed_eval

        basis = {
            "freqs": freq_grid,
            "zeroth": sed0,
            "pivots": pivot_vec,
            "step_sizes": step_sizes.copy(),
            "order": target_order,
        }

        if target_order >= 1:
            first_derivs = {}
            for idx, name in enumerate(self.param_names):
                e = unit_vectors[idx]
                sed_plus = evaluate(e)
                sed_minus = evaluate(-e)
                first_derivs[name] = (sed_plus - sed_minus) / (2.0 * step_sizes[idx])
            basis["first"] = first_derivs

        if target_order >= 2:
            second_diag = {}
            second_cross = {}
            for idx, name in enumerate(self.param_names):
                e = unit_vectors[idx]
                sed_plus = evaluate(e)
                sed_minus = evaluate(-e)
                second_diag[name] = (sed_plus - 2.0 * sed0 + sed_minus) / (step_sizes[idx] ** 2)
            for i in range(len(self.param_names)):
                for j in range(i + 1, len(self.param_names)):
                    key = (self.param_names[i], self.param_names[j])
                    e_i = unit_vectors[i]
                    e_j = unit_vectors[j]
                    sed_pp = evaluate(e_i + e_j)
                    sed_pm = evaluate(e_i - e_j)
                    sed_mp = evaluate(-e_i + e_j)
                    sed_mm = evaluate(-e_i - e_j)
                    second_cross[key] = (sed_pp - sed_pm - sed_mp + sed_mm) / (4.0 * step_sizes[i] * step_sizes[j])
            basis["second"] = {"diag": second_diag, "cross": second_cross}

        self.last_basis = basis
        return basis

    def evaluate_expansion(self, samples, basis=None, order=None):
        if basis is None:
            if self.last_basis is None:
                raise ValueError("No cached basis available. Call generate_SED_basis first.")
            basis = self.last_basis

        samples = np.asarray(samples, dtype=float)
        squeeze = False
        if samples.ndim == 1:
            if samples.size != 3:
                raise ValueError("samples must have three entries: (log_a, beta, log_xC).")
            samples = samples[None, :]
            squeeze = True
        elif samples.shape[1] != 3:
            raise ValueError("samples must have shape (N, 3).")

        if order is None:
            resolved_order = min(self.order, basis["order"])
        else:
            resolved_order = self._validate_order(order)
            resolved_order = min(resolved_order, self.order, basis["order"])

        freqs = basis["freqs"]
        pivot_vec = np.asarray(basis["pivots"], dtype=float)

        approximations = []
        for sample in samples:
            delta = sample - pivot_vec
            sed = np.asarray(basis["zeroth"], dtype=float).copy()
            if resolved_order >= 1:
                for idx, name in enumerate(self.param_names):
                    sed = sed + delta[idx] * basis["first"][name]
            if resolved_order >= 2:
                diag_terms = basis["second"]["diag"]
                cross_terms = basis["second"]["cross"]
                for idx, name in enumerate(self.param_names):
                    sed = sed + 0.5 * (delta[idx] ** 2) * diag_terms[name]
                for i in range(len(self.param_names)):
                    for j in range(i + 1, len(self.param_names)):
                        key = (self.param_names[i], self.param_names[j])
                        sed = sed + delta[i] * delta[j] * cross_terms[key]
            approximations.append(sed)

        approximations = np.asarray(approximations)
        if squeeze:
            approximations = approximations[0]

        return freqs, approximations

    def fit_basis_functions(self, target_sed, freqs=None, basis=None, order=None, include_zeroth=True, weights=None, rcond=None, return_fitted=False, tumbling=True):
        """Fit the target SED using all available basis functions up to the requested order."""
        target = np.asarray(target_sed, dtype=float)
        if target.ndim != 1 or target.size == 0:
            raise ValueError("target_sed must be a non-empty one-dimensional array.")

        basis = self._resolve_basis(freqs, basis, tumbling)
        if order is None:
            resolved_order = min(self.order, basis["order"])
        else:
            resolved_order = self._validate_order(order)
            resolved_order = min(resolved_order, self.order, basis["order"])

        freqs_basis = np.asarray(basis["freqs"], dtype=float)
        if target.shape != freqs_basis.shape:
            raise ValueError("target_sed must have the same shape as the basis frequency grid.")

        design_matrix, column_labels = self._assemble_design_matrix(basis, resolved_order, include_zeroth)
        design_matrix = np.asarray(design_matrix, dtype=float)

        weighted_matrix = design_matrix
        weighted_target = target
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != target.shape:
                raise ValueError("weights must match the SED shape.")
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative.")
            sqrt_w = np.sqrt(weights)
            weighted_matrix = design_matrix * sqrt_w[:, None]
            weighted_target = target * sqrt_w

        coeffs, _, _, _ = np.linalg.lstsq(weighted_matrix, weighted_target, rcond=rcond)
        fitted_sed = design_matrix @ coeffs

        result = {
            "freqs": freqs_basis,
            "coefficients": coeffs,
            "coefficient_labels": column_labels,
            "coefficients_map": dict(zip(column_labels, coeffs)),
            "order": resolved_order,
        }
        if return_fitted:
            result["fitted_sed"] = fitted_sed
            result["residual"] = target - fitted_sed

        return result

    def fit_linear_basis(self, target_sed, freqs=None, basis=None, weights=None, rcond=None, return_fitted=False, tumbling=True):
        """Fit a target SED with the linear (first-order) basis functions."""
        target = np.asarray(target_sed, dtype=float)
        if target.ndim != 1 or target.size == 0:
            raise ValueError("target_sed must be a non-empty one-dimensional array.")

        basis = self._resolve_basis(freqs, basis, tumbling)
        if basis.get("order", 0) < 1 or "first" not in basis:
            raise ValueError("The provided basis does not contain first-order derivatives.")

        freqs_basis = np.asarray(basis["freqs"], dtype=float)
        if target.shape != freqs_basis.shape:
            raise ValueError("target_sed must have the same shape as the basis frequency grid.")

        zeroth = np.asarray(basis["zeroth"], dtype=float)
        residual = target - zeroth

        design_matrix, _ = self._assemble_design_matrix(basis, max_order=1, include_zeroth=False)
        design_matrix = np.asarray(design_matrix, dtype=float)

        weighted_matrix = design_matrix
        weighted_residual = residual
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape != target.shape:
                raise ValueError("weights must match the SED shape.")
            if np.any(weights < 0):
                raise ValueError("weights must be non-negative.")
            sqrt_w = np.sqrt(weights)
            weighted_matrix = design_matrix * sqrt_w[:, None]
            weighted_residual = residual * sqrt_w

        coeffs, _, _, _ = np.linalg.lstsq(weighted_matrix, weighted_residual, rcond=rcond)
        fitted_delta = coeffs
        fitted_sample = np.asarray(basis["pivots"], dtype=float) + fitted_delta
        fitted_sed = zeroth + (design_matrix @ coeffs)

        result = {
            "freqs": freqs_basis,
            "delta_params": dict(zip(self.param_names, fitted_delta)),
            "sample": fitted_sample,
            "coefficients": coeffs,
            "coefficients_map": dict(zip(self.param_names, coeffs)),
        }

        if return_fitted:
            result["fitted_sed"] = fitted_sed
            result["residual"] = target - fitted_sed

        return result