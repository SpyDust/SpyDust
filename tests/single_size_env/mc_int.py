import numpy as np
from scipy.stats import gaussian_kde

def monte_carlo_integral(params, data, p_target, q_sample=None):
    """
    Estimate integral of f(theta) over parameter space with target distribution p_target,
    using samples and importance sampling. If the sampling density q_sample is not provided,
    it is estimated from the samples using KDE.

    Parameters
    ----------
    params : np.ndarray, shape (N_samples, N_params)
        Sample points in the parameter space.
    data : np.ndarray, shape (N_samples, N_data)
        Function values at the sample points.
    p_target : callable
        Target probability density function p(theta), evaluated at each sample.
    q_sample : callable or None
        Sampling probability density function q(theta), evaluated at each sample.
        If None, estimated via KDE from params.

    Returns
    -------
    integral : np.ndarray, shape (N_data,)
        Estimated integral over parameter space.
    """
    params = np.asarray(params)
    data = np.asarray(data)
    N_samples = len(params)

    # Evaluate target PDF
    p_vals = np.asarray(p_target(params))
    
    # Evaluate or estimate sampling PDF
    if q_sample is None:
        kde = gaussian_kde(params.T)  # scipy expects shape (N_params, N_samples)
        q_vals = kde(params.T)
    else:
        q_vals = np.asarray(q_sample(params))
    
    # Importance weights
    weights = p_vals / q_vals
    weights /= np.sum(weights)  # normalize to sum to 1

    # Weighted sum
    integral = np.sum(weights[:, None] * data, axis=0)
    return integral

from scipy.stats import qmc

def qmc_integral(N_samples, N_params, data_func, p_target, bounds=None, method='sobol'):
    """
    Estimate integral of f(theta) over parameter space with target PDF using quasi-Monte Carlo.

    Parameters
    ----------
    N_samples : int
        Number of quasi-Monte Carlo points.
    N_params : int
        Dimension of the parameter space.
    data_func : callable
        Function f(theta) that returns data array, shape (N_samples, N_data) when theta has shape (N_samples, N_params).
    p_target : callable
        Target probability density function p(theta), evaluated at each theta, shape (N_samples,).
    bounds : list of tuples, optional
        [(low1, high1), ..., (lowD, highD)] for each parameter. If None, assumes [0,1]^d.
    method : str
        QMC sequence: 'sobol' (default) or 'halton'.

    Returns
    -------
    integral : np.ndarray, shape (N_data,)
        Estimated integral of f(theta) over the parameter space w.r.t p_target(theta).
    """
    # Default unit hypercube
    if bounds is None:
        bounds = [(0.0, 1.0)] * N_params
    bounds = np.array(bounds)
    lows = bounds[:,0]
    highs = bounds[:,1]

    # Generate QMC points in unit cube
    if method.lower() == 'sobol':
        sampler = qmc.Sobol(d=N_params, scramble=True)
        u = sampler.random(N_samples)
    elif method.lower() == 'halton':
        sampler = qmc.Halton(d=N_params, scramble=True)
        u = sampler.random(N_samples)
    else:
        raise ValueError("method must be 'sobol' or 'halton'")

    # Scale to bounds
    theta = lows + (highs - lows) * u  # shape (N_samples, N_params)

    # Evaluate function
    data = np.asarray(data_func(theta))  # shape (N_samples, N_data)

    # Evaluate target PDF
    p_vals = np.asarray(p_target(theta))  # shape (N_samples,)

    # Since QMC points are uniform in hypercube, weight by PDF and volume
    vol = np.prod(highs - lows)
    integral = vol * np.mean(data * p_vals[:, None], axis=0)

    return integral

def qmc_integral_importance(N_samples, N_params, data_func, p_target, q_sample, bounds=None, method='sobol'):
    """
    QMC integral with importance sampling: integrates f(theta) * p_target(theta) / q_sample(theta).

    Parameters
    ----------
    N_samples : int
        Number of QMC points.
    N_params : int
        Dimension of the parameter space.
    data_func : callable
        Function f(theta), returns shape (N_samples, N_data).
    p_target : callable
        Target PDF p(theta), evaluated at each theta.
    q_sample : callable
        Sampling PDF q(theta) used to generate the QMC points.
    bounds : list of tuples, optional
        Bounds of the parameter space [(low1, high1), ..., (lowD, highD)].
        If None, assumes [0,1]^d.
    method : str
        QMC sequence: 'sobol' (default) or 'halton'.

    Returns
    -------
    integral : np.ndarray, shape (N_data,)
        Estimated integral of f(theta) over the parameter space w.r.t p_target(theta).
    """
    # Default unit hypercube
    if bounds is None:
        bounds = [(0.0, 1.0)] * N_params
    bounds = np.array(bounds)
    lows = bounds[:,0]
    highs = bounds[:,1]

    # Generate QMC points in unit cube
    if method.lower() == 'sobol':
        sampler = qmc.Sobol(d=N_params, scramble=True)
        u = sampler.random(N_samples)
    elif method.lower() == 'halton':
        sampler = qmc.Halton(d=N_params, scramble=True)
        u = sampler.random(N_samples)
    else:
        raise ValueError("method must be 'sobol' or 'halton'")

    # Scale to bounds
    theta_unit = u
    theta = lows + (highs - lows) * theta_unit  # shape (N_samples, N_params)

    # Evaluate function
    data = np.asarray(data_func(theta))  # shape (N_samples, N_data)

    # Evaluate target PDF and sampling PDF
    p_vals = np.asarray(p_target(theta))
    q_vals = np.asarray(q_sample(theta))

    # Importance weights
    weights = p_vals / q_vals
    integral = np.mean(weights[:, None] * data, axis=0)  # weighted mean

    # Scale by volume of hypercube
    vol = np.prod(highs - lows)
    integral *= vol

    return integral


def qmc_integral_auto(N_samples, N_params, data_func, p_target, q_sample=None, bounds=None, method='sobol'):
    """
    Fully automatic QMC integrator with optional importance sampling.

    Parameters
    ----------
    N_samples : int
        Number of QMC points.
    N_params : int
        Dimension of the parameter space.
    data_func : callable
        Function f(theta), returns shape (N_samples, N_data).
    p_target : callable
        Target PDF p(theta), evaluated at each theta.
    q_sample : callable or None
        Sampling PDF q(theta). If None, QMC points are uniform in bounds and importance weights = p_target.
    bounds : list of tuples or None
        Bounds of the parameter space [(low1, high1), ..., (lowD, highD)]. Default [0,1]^d.
    method : str
        QMC sequence: 'sobol' (default) or 'halton'.

    Returns
    -------
    integral : np.ndarray, shape (N_data,)
        Estimated integral of f(theta) over the parameter space w.r.t p_target(theta).
    """
    if bounds is None:
        bounds = [(0.0, 1.0)] * N_params
    bounds = np.array(bounds)
    lows = bounds[:,0]
    highs = bounds[:,1]
    
    # Generate QMC points in unit cube
    if method.lower() == 'sobol':
        sampler = qmc.Sobol(d=N_params, scramble=True)
        u = sampler.random(N_samples)
    elif method.lower() == 'halton':
        sampler = qmc.Halton(d=N_params, scramble=True)
        u = sampler.random(N_samples)
    else:
        raise ValueError("method must be 'sobol' or 'halton'")
    
    # Scale to bounds
    theta = lows + (highs - lows) * u  # shape (N_samples, N_params)

    # Evaluate function
    data = np.asarray(data_func(theta))  # shape (N_samples, N_data)

    # Evaluate target PDF
    p_vals = np.asarray(p_target(theta))

    if q_sample is None:
        # Uniform sampling in bounds: weights = target PDF
        weights = p_vals
    else:
        # Importance sampling
        q_vals = np.asarray(q_sample(theta))
        weights = p_vals / q_vals

    # Normalize weights and take weighted mean
    weights /= np.sum(weights)
    integral = np.sum(weights[:, None] * data, axis=0)

    # Scale by hypercube volume
    vol = np.prod(highs - lows)
    integral *= vol

    return integral