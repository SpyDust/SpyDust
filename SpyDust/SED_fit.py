from scipy.optimize import curve_fit
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt


def lognormal_sed_fit_v1(freqs, sed, initial_guess=None, thres=1e-3, brief_return=True, return_fit_curve=False):
    """
    Fit a log-normal model to SED data to extract peak frequency and width.
    
    The log-normal model used is:
    SED(f) = A * exp(-0.5 * ((ln(f) - ln(f_peak)) / sigma)^2)
    
    where:
    - f_peak is the peak frequency
    - sigma is the width parameter in log space
    - A is the amplitude
    
    Parameters:
    -----------
    freqs : array_like
        1D array of frequencies
    sed : array_like
        1D array of SED values corresponding to freqs
    initial_guess : tuple, optional
        Initial guess for (amplitude, log_f_peak, sigma)
        If None, will estimate from data
    thres : float, optional
        Threshold parameter. Points with SED < thres * max(SED) are masked as invalid.
        Default is 1e-6.
    return_fit_curve : bool, optional
        If True, also return the fitted curve values
    
    Returns:
    --------
    f_peak : float
        Peak frequency of the fitted log-normal
    sigma : float
        Width parameter in log space
    fit_params : tuple
        Full fitting parameters (amplitude, log_f_peak, sigma)
    fit_curve : array_like (only if return_fit_curve=True)
        Fitted SED values at input frequencies
    fit_quality : dict
        Dictionary containing R-squared and other fit quality metrics
    """

    if return_fit_curve and brief_return:
        raise ValueError("Cannot return fit curve with brief_return=True")
    
    # Convert to numpy arrays
    freqs = np.asarray(freqs)
    sed = np.asarray(sed)
    
    # Calculate threshold value
    sed_max = np.max(sed)
    threshold_value = thres * sed_max
    
    # Remove any non-positive values and points below threshold
    valid_mask = (freqs > 0) & (sed > 0) & np.isfinite(freqs) & np.isfinite(sed) & (sed >= threshold_value)
    freqs_clean = freqs[valid_mask]
    sed_clean = sed[valid_mask]
    
    if len(freqs_clean) < 3:
        raise ValueError(f"Need at least 3 valid data points for fitting. Only {len(freqs_clean)} points above threshold {threshold_value:.2e}")
    
    # Define log-normal function in linear space
    def lognormal_func(log_f, amplitude, log_f_peak, sigma):
        return amplitude * np.exp(-0.5 * ((log_f - log_f_peak) / sigma)**2)
    
    # Estimate initial parameters if not provided
    if initial_guess is None:
        # Find approximate peak
        peak_idx = np.argmax(sed_clean)
        f_peak_est = freqs_clean[peak_idx]
        amplitude_est = sed_clean[peak_idx]
        
        # Estimate width from FWHM
        half_max = amplitude_est / 2
        # Find points closest to half maximum
        left_idx = np.argmin(np.abs(sed_clean[:peak_idx] - half_max)) if peak_idx > 0 else 0
        right_idx = peak_idx + np.argmin(np.abs(sed_clean[peak_idx:] - half_max))
        
        if right_idx > left_idx:
            fwhm_freq = freqs_clean[right_idx] - freqs_clean[left_idx]
            # Convert FWHM to sigma for log-normal (approximate)
            sigma_est = np.log(1 + fwhm_freq / f_peak_est) / (2 * np.sqrt(2 * np.log(2)))
        else:
            sigma_est = 0.5  # Default fallback
        
        initial_guess = (amplitude_est, np.log(f_peak_est), sigma_est)
    
    try:
        # Perform the fit
        popt, pcov = curve_fit(lognormal_func, np.log(freqs_clean), sed_clean, 
                              p0=initial_guess, maxfev=5000)
        
        amplitude_fit, log_f_peak_fit, sigma_fit = popt

        if brief_return:
            return log_f_peak_fit, sigma_fit
        else:
            f_peak = np.exp(log_f_peak_fit)
            
            # Calculate fit quality metrics
            sed_fit = lognormal_func(np.log(freqs_clean), *popt)
            ss_res = np.sum((sed_clean - sed_fit)**2)
            ss_tot = np.sum((sed_clean - np.mean(sed_clean))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            f_peak_error = f_peak * param_errors[1]  # Error propagation for exp(log_f_peak)
            
            fit_quality = {
                'r_squared': r_squared,
                'residual_sum_squares': ss_res,
                'f_peak_error': f_peak_error,
                'sigma_error': param_errors[2],
                'amplitude_error': param_errors[0],
                'covariance_matrix': pcov,
                'n_points_used': len(freqs_clean),
                'threshold_value': threshold_value
            }
            
            if return_fit_curve:
                # Return fit curve at original frequency points
                fit_curve_full = np.full_like(freqs, np.nan)
                fit_curve_full[valid_mask] = sed_fit
                return f_peak, sigma_fit, popt, fit_curve_full, fit_quality
            else:
                return f_peak, sigma_fit, popt, fit_quality
            
    except Exception as e:
        raise RuntimeError(f"Fitting failed: {str(e)}")

def lognormal_sed_fit_v2(freqs, sed, initial_guess=None, thres=1e-3, brief_return=True, return_fit_curve=False):
    """
    Fit a log-normal model to SED data to extract peak frequency and width.
    
    The log-normal model used is:
    SED(f) = A * exp(-0.5 * ((ln(f) - ln(f_peak)) / sigma)^2)
    
    where:
    - f_peak is the peak frequency
    - sigma is the width parameter in log space
    - A is the amplitude
    
    Parameters:
    -----------
    freqs : array_like
        1D array of frequencies
    sed : array_like
        1D array of SED values corresponding to freqs
    initial_guess : tuple, optional
        Initial guess for (amplitude, log_f_peak, sigma)
        If None, will estimate from data
    thres : float, optional
        Threshold parameter. Points with SED < thres * max(SED) are masked as invalid.
        Default is 1e-6.
    return_fit_curve : bool, optional
        If True, also return the fitted curve values
    
    Returns:
    --------
    f_peak : float
        Peak frequency of the fitted log-normal
    sigma : float
        Width parameter in log space
    fit_params : tuple
        Full fitting parameters (amplitude, log_f_peak, sigma)
    fit_curve : array_like (only if return_fit_curve=True)
        Fitted SED values at input frequencies
    fit_quality : dict
        Dictionary containing R-squared and other fit quality metrics
    """

    if return_fit_curve and brief_return:
        raise ValueError("Cannot return fit curve with brief_return=True")
    
    # Convert to numpy arrays
    freqs = np.asarray(freqs)
    sed = np.asarray(sed)
    
    # Calculate threshold value
    sed_max = np.max(sed)
    threshold_value = thres * sed_max
    
    # Remove any non-positive values and points below threshold
    valid_mask = (freqs > 0) & (sed > 0) & np.isfinite(freqs) & np.isfinite(sed) & (sed >= threshold_value)
    freqs_clean = freqs[valid_mask]
    sed_clean = sed[valid_mask]
    
    if len(freqs_clean) < 3:
        raise ValueError(f"Need at least 3 valid data points for fitting. Only {len(freqs_clean)} points above threshold {threshold_value:.2e}")
    
    # Define log-normal function in linear space
    def lognormal_func(log_f, amplitude, gamma, log_f_peak, sigma):
        return amplitude * np.exp( gamma * log_f -0.5 * ((log_f - log_f_peak) / sigma)**2)
    
    # Estimate initial parameters if not provided
    if initial_guess is None:
        # Find approximate peak
        peak_idx = np.argmax(sed_clean)
        f_peak_est = freqs_clean[peak_idx]
        amplitude_est = sed_clean[peak_idx]
        
        # Estimate width from FWHM
        half_max = amplitude_est / 2
        # Find points closest to half maximum
        left_idx = np.argmin(np.abs(sed_clean[:peak_idx] - half_max)) if peak_idx > 0 else 0
        right_idx = peak_idx + np.argmin(np.abs(sed_clean[peak_idx:] - half_max))
        
        if right_idx > left_idx:
            fwhm_freq = freqs_clean[right_idx] - freqs_clean[left_idx]
            # Convert FWHM to sigma for log-normal (approximate)
            sigma_est = np.log(1 + fwhm_freq / f_peak_est) / (2 * np.sqrt(2 * np.log(2)))
        else:
            sigma_est = 0.5  # Default fallback
        
        initial_guess = (amplitude_est, 0, np.log(f_peak_est), sigma_est)
    
    try:
        # Perform the fit
        popt, pcov = curve_fit(lognormal_func, np.log(freqs_clean), sed_clean, 
                              p0=initial_guess, maxfev=5000)
        
        amplitude_fit, gamma_fit, log_f_peak_fit, sigma_fit = popt

        if brief_return:
            return log_f_peak_fit, gamma_fit, sigma_fit
        else:
            f_peak = np.exp(log_f_peak_fit)
            
            # Calculate fit quality metrics
            sed_fit = lognormal_func(np.log(freqs_clean), *popt)
            ss_res = np.sum((sed_clean - sed_fit)**2)
            ss_tot = np.sum((sed_clean - np.mean(sed_clean))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate parameter uncertainties
            param_errors = np.sqrt(np.diag(pcov))
            f_peak_error = f_peak * param_errors[2]  # Error propagation for exp(log_f_peak)
            
            fit_quality = {
                'r_squared': r_squared,
                'residual_sum_squares': ss_res,
                'f_peak_error': f_peak_error,
                'sigma_error': param_errors[3],
                'amplitude_error': param_errors[0],
                'gamma_error': param_errors[1], 
                'covariance_matrix': pcov,
                'n_points_used': len(freqs_clean),
                'threshold_value': threshold_value
            }
            
            if return_fit_curve:
                # Return fit curve at original frequency points
                fit_curve_full = np.full_like(freqs, np.nan)
                fit_curve_full[valid_mask] = sed_fit
                return f_peak, gamma_fit, sigma_fit, popt, fit_curve_full, fit_quality
            else:
                return f_peak, gamma_fit, sigma_fit, popt, fit_quality
            
    except Exception as e:
        raise RuntimeError(f"Fitting failed: {str(e)}")
  

def fit_sed_ensemble(freqs, sed_ensemble, thres=1e-3, parameter_list=None):
    """
    Fit SED ensemble to log-normal model.
    
    Parameters:
    -----------
    freqs : array_like
        Frequency array
    sed_ensemble : array_like
        2D array of SED values (rows are different realizations)
    thres : float, optional
        Threshold parameter for fitting. Default is 1e-3.
    """
    if sed_ensemble.ndim == 1:
        sed_ensemble = sed_ensemble.reshape(1, -1)
    n_sed = sed_ensemble.shape[0]

    lognormal_sed_fit = lognormal_sed_fit_v1
    fit_params = np.zeros((n_sed, 2))

    from tqdm import tqdm
    for i in tqdm(range(n_sed)):
        try:
            fit_params[i] = lognormal_sed_fit(freqs, sed_ensemble[i], thres=thres, brief_return=True)
        except:
            if parameter_list is not None:
                print(parameter_list[i])
            raise ValueError(f"Failed to fit SED {i}")
    if sed_ensemble.shape[0] == 1:
        fit_params = fit_params[0]
    return fit_params

def measure_sed_peak_properties(freqs, sed, plot=False, save_path=None, thres=1e-3, title='SED fit:', v2=True):
    """
    Convenience function to measure peak properties of an SED using log-normal fitting.
    
    Parameters:
    -----------
    freqs : array_like
        Frequency array
    sed : array_like  
        SED values
    plot : bool, optional
        If True, create a plot showing the fit
    save_path : str, optional
        Path to save the plot
    thres : float, optional
        Threshold parameter. Points with SED < thres * max(SED) are masked as invalid.
        Default is 1e-6.
        
    Returns:
    --------
    results : dict
        Dictionary containing:
        - 'f_peak': peak frequency
        - 'sigma': width parameter
        - 'fwhm': full width at half maximum
        - 'r_squared': fit quality
    """
    if v2:
        lognormal_sed_fit = lognormal_sed_fit_v2
        f_peak, gamma, sigma, fit_params, fit_quality = lognormal_sed_fit(freqs, sed, thres=thres, brief_return=False)
    else:
        lognormal_sed_fit = lognormal_sed_fit_v1
        f_peak, sigma, fit_params, fit_quality = lognormal_sed_fit(freqs, sed, thres=thres, brief_return=False)
    
    # Calculate FWHM from sigma
    # For log-normal: FWHM ≈ f_peak * (exp(sigma*sqrt(2*ln(2))) - exp(-sigma*sqrt(2*ln(2))))
    fwhm_factor = np.sqrt(2 * np.log(2))
    fwhm = f_peak * (np.exp(sigma * fwhm_factor) - np.exp(-sigma * fwhm_factor))
    
    results = {
        'f_peak': f_peak,
        'sigma': sigma,
        'fwhm': fwhm,
        'r_squared': fit_quality['r_squared'],
        'amplitude': fit_params[0],
        'f_peak_error': fit_quality['f_peak_error'],
        'sigma_error': fit_quality['sigma_error'],
        'n_points_used': fit_quality['n_points_used'],
        'threshold_value': fit_quality['threshold_value']
    }
    ft_size = 14
    
    if plot:
        fig = plt.figure(figsize=(5, 3))
        ax1 = fig.add_subplot(111)

        # Main plot
        f_peak_plot = f_peak
        norm = max(sed)
        ax1.loglog(freqs, sed/norm, 'o-', label='Simulation', alpha=0.7)
        
        # Show threshold line
        # ax1.axhline(y=fit_quality['threshold_value'], color='gray', linestyle=':', 
        #            alpha=0.5, label=f'Threshold ({thres:.0e}×max)')
        # ax1.set_ylim(fit_quality['threshold_value'], max(sed) * 10)
        ax1.set_ylim(thres, 10)
        
        # Generate smooth fit curve for plotting
        freq_smooth = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), 200)
        if v2:
            _, _, _, _, fit_smooth, _ = lognormal_sed_fit(freqs, sed, return_fit_curve=True, thres=thres, brief_return=False)
        else:
            _, _, _, fit_smooth, _ = lognormal_sed_fit(freqs, sed, return_fit_curve=True, thres=thres, brief_return=False)
        
        # Interpolate fit for smooth curve
        from scipy.interpolate import interp1d
        valid_mask = np.isfinite(fit_smooth)
        if np.sum(valid_mask) > 1:
            interp_func = interp1d(freqs[valid_mask], fit_smooth[valid_mask], 
                                 kind='linear', bounds_error=False, fill_value=np.nan)
            fit_smooth_plot = interp_func(freq_smooth)
            ax1.loglog(freq_smooth, fit_smooth_plot/norm, '--', label='Log-Gaussian fit', linewidth=2)
        
        ax1.axvline(f_peak_plot, color='red', linestyle=':', alpha=0.8, 
                   label=rf'Fitted peak')
        ax1.set_xlabel('Frequency [GHz]', fontsize=ft_size)
        ax1.set_xlim(1.5, freqs.max())
        ax1.set_ylabel('SED (normalized)', fontsize=ft_size)
        # No frame legend
        ax1.legend(fontsize=ft_size, loc='upper left', frameon=False)
        # ax1.grid(True, alpha=0.3)
        ax1.set_title(rf'{title}', fontsize=ft_size+1)
        ax1.tick_params(axis='both', which='major', labelsize=ft_size-2)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    return results


def measure_sed_log_moments(freqs, sed, thres=1e-3, excess_kurtosis=True):
    """Measure the first four moments of an SED profile in logarithmic frequency.

    We interpret the SED as a weighted distribution over logarithmic frequency,
    keeping only samples with amplitude above ``thres * max(sed)``.  Introducing
    :math:`x_i = \ln \nu_i` and local spacings :math:`\Delta x_i`, we define
    weights :math:`w_i = S_i \Delta x_i` and normalized probabilities
    :math:`p_i = w_i / \sum_j w_j`.  The central moments are then

    .. math::

        \mu_1 = \sum_i p_i x_i,\quad
        \mu_2 = \sum_i p_i (x_i - \mu_1)^2,\quad
        \mu_3 = \sum_i p_i (x_i - \mu_1)^3,\quad
        \mu_4 = \sum_i p_i (x_i - \mu_1)^4.

    We report the mean ``mu_1``, variance ``mu_2``, skewness
    :math:`\gamma_1 = \mu_3 / \mu_2^{3/2}`, and kurtosis
    :math:`\kappa = \mu_4 / \mu_2^2` (optionally excess kurtosis ``kappa - 3``).

    Parameters
    ----------
    freqs : array_like
        Frequency grid of the SED (positive values required).
    sed : array_like
        Spectral energy density values (non-negative).
    thres : float, optional
        Fractional amplitude threshold relative to ``max(sed)`` for selecting
        samples.  Default is :math:`10^{-3}`.
    excess_kurtosis : bool, optional
        If True, subtract 3 from the kurtosis (i.e. return excess kurtosis).

    Returns
    -------
    dict
        Dictionary with keys ``mean``, ``variance``, ``skewness``, ``kurtosis``.
        The ``kurtosis`` entry respects ``excess_kurtosis``.

    Raises
    ------
    ValueError
        If fewer than four valid samples remain after thresholding or if the
        second central moment is non-positive.
    """

    freqs = np.asarray(freqs, dtype=float)
    sed = np.asarray(sed, dtype=float)

    if freqs.shape != sed.shape:
        raise ValueError("freqs and sed must have the same shape")

    if np.any(freqs <= 0):
        raise ValueError("Frequencies must be strictly positive for log-space analysis")

    sed_max = np.max(sed)
    if sed_max <= 0:
        raise ValueError("SED must contain positive values")

    threshold = thres * sed_max
    mask = (sed >= threshold) & np.isfinite(freqs) & np.isfinite(sed)
    if not np.any(mask):
        raise ValueError("No samples exceed the requested threshold")

    freqs_sel = freqs[mask]
    sed_sel = sed[mask]

    if freqs_sel.size < 4:
        raise ValueError("At least four samples above threshold are required to compute moments")

    log_freq = np.log(freqs_sel)

    # Local spacing in log-frequency (forward/backward difference at the ends).
    delta = np.empty_like(log_freq)
    if log_freq.size == 1:
        delta.fill(1.0)
    else:
        delta[1:-1] = 0.5 * (log_freq[2:] - log_freq[:-2])
        delta[0] = log_freq[1] - log_freq[0]
        delta[-1] = log_freq[-1] - log_freq[-2]

    weights = sed_sel * np.abs(delta)
    total_weight = np.sum(weights)
    if total_weight == 0 or not np.isfinite(total_weight):
        raise ValueError("Degenerate weights encountered while computing kurtosis")

    probs = weights / total_weight

    mean_log_freq = np.sum(probs * log_freq)
    centered = log_freq - mean_log_freq
    m2 = np.sum(probs * centered**2)

    if m2 <= 0:
        raise ValueError("Second central moment is non-positive; moments undefined")

    m3 = np.sum(probs * centered**3)
    m4 = np.sum(probs * centered**4)

    skewness = m3 / (m2 ** 1.5)
    kurtosis = m4 / (m2 ** 2)
    if excess_kurtosis:
        kurtosis -= 3.0

    return {
        'mean': mean_log_freq,
        'variance': m2,
        'skewness': skewness,
        'kurtosis': kurtosis,
    }


def measure_sed_log_moments_batch(freqs, sed_matrix, thres=1e-3, excess_kurtosis=True):
    """Compute log-space moments for a collection of SEDs.

    Parameters
    ----------
    freqs : array_like
        Shared frequency grid ``(N_freqs,)``. Must contain positive values.
    sed_matrix : array_like
        Array of SEDs with shape ``(N_samples, N_freqs)``.
    thres : float, optional
        Fractional amplitude threshold used for each SED (default ``1e-3``).
    excess_kurtosis : bool, optional
        Pass-through flag to control the kurtosis definition. When ``True``
        (default) the output reports excess kurtosis.

    Returns
    -------
    numpy.ndarray
        Array with shape ``(N_samples, 4)`` containing ``mean``, ``variance``,
        ``skewness``, and ``kurtosis`` (or excess kurtosis).

    Notes
    -----
    This simply applies :func:`measure_sed_log_moments` row-by-row; any
    individual failure propagates to the caller.
    """

    freqs = np.asarray(freqs, dtype=float)
    sed_matrix = np.asarray(sed_matrix, dtype=float)

    if sed_matrix.ndim != 2:
        raise ValueError("sed_matrix must be 2D with shape (N_samples, N_freqs)")

    n_samples, n_freqs = sed_matrix.shape
    if freqs.shape != (n_freqs,):
        raise ValueError("freqs must have shape (N_freqs,) matching sed_matrix")

    moments = np.empty((n_samples, 4), dtype=float)
    for idx in range(n_samples):
        stats = measure_sed_log_moments(
            freqs,
            sed_matrix[idx],
            thres=thres,
            excess_kurtosis=excess_kurtosis,
        )
        moments[idx, 0] = stats['mean']
        moments[idx, 1] = stats['variance']
        moments[idx, 2] = stats['skewness']
        moments[idx, 3] = stats['kurtosis']

    return moments

