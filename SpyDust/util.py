import numpy as np
from concurrent.futures import ThreadPoolExecutor

#from numba import njit
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator, interp1d

class cgsconst:
    # Stores the needed constants in cgs units
    pi = 3.1415926535897932385
    c = 2.99792458e10  # speed of light
    q = 4.8032068e-10  # elementary charge
    k = 1.380650e-16   # Boltzmann constant
    mp = 1.6726231e-24  # proton mass
    me = 9.1093898e-28  # electron mass
    h = 6.6260688e-27   # Planck constant
    debye = 1e-18       # 1 Debye in cgs units
    eV = 1.60217653e-12  # 1eV in ergs

#@njit
def maketab(xmin, xmax, Npt):
    """Creates a linearly spaced table."""
    return xmin + (xmax - xmin) / Npt * (0.5 + np.arange(Npt))

#@njit
def makelogtab(xmin, xmax, Npt):
    """Creates a logarithmically spaced table."""
    return xmin * np.exp(1 / Npt * np.log(xmax / xmin) * (0.5 + np.arange(Npt)))

#@njit
def DX_over_X(xmin, xmax, Npt):
    """Returns Dx/x for a logarithmically spaced table created by makelogtab."""
    return np.exp(0.5 / Npt * np.log(xmax / xmin)) - np.exp(-0.5 / Npt * np.log(xmax / xmin))


def biinterp_func(f_val, X, Y):
    """
    Returns a function that interpolates f given at X, Y (logarithmically spaced).

    Parameters:
    - f_val: 2D array of values at Nx * Ny grid.
    - X, Y: Arrays of logarithmically spaced values.

    Returns:
    - Function that interpolates f at Xnew, Ynew.
    """


    # Interpolation requires coordinates as pairs
    return RegularGridInterpolator((X, Y), f_val, method='linear', bounds_error=False, fill_value=None)

def coord_grid(X, Y):
    """
    Returns the grid of coordinates for X, Y.
    """
    xg, yg = np.meshgrid(X, Y, indexing='ij')
    points = np.empty((xg.shape[0], xg.shape[1], 2))
    points[:, :, 0] = xg
    points[:, :, 1] = yg
    return points

def biinterplog(f, X, Y, Xnew, Ynew):
    """
    Interpolates the 2D array `f` given at logarithmically spaced points `X` and `Y`
    at new points `Xnew` and `Ynew`.
    
    Parameters:
    - f: 2D array of shape (Nx, Ny), function values at grid points.
    - X: 1D array of length Nx, original X-axis (logarithmic spacing).
    - Y: 1D array of length Ny, original Y-axis (logarithmic spacing).
    - Xnew: 1D array, new X-axis points to interpolate to.
    - Ynew: 1D array, new Y-axis points to interpolate to.
    
    Returns:
    - Interpolated values at the specified `Xnew` and `Ynew` points.
    """
    

    # Convert X and Y to logarithmic space
    logX = np.log(X)
    logY = np.log(Y)
    
    # Create indices for interpolation based on X and Y
    Xind = np.arange(len(X))
    Yind = np.arange(len(Y))
    
    # Interpolate indices for the new X and Y in log space
    Xind_new = np.interp(np.log(Xnew), logX, Xind)
    Yind_new = np.interp(np.log(Ynew), logY, Yind)
    Xind_new = np.array(Xind_new).reshape(-1,)
    Yind_new = np.array(Yind_new).reshape(-1,)
    
    # Set up 2D grid interpolator on the original index grid
    interpolator = RegularGridInterpolator((Xind, Yind), f, method='linear')
    
    # Prepare points for interpolating to `Xind_new`, `Yind_new`
    points = np.array([[xi, yi] for xi in Xind_new for yi in Yind_new])
    
    # Interpolate and reshape result
    interpolated_values = interpolator(points)
    return interpolated_values.reshape(np.size(Xnew), np.size(Ynew))

def log_biinterplog(f, X, Y, Xnew, Ynew):
    """
    Same as biinterplog, but f is logarithmically sampled.

    Parameters:
    - f: 2D array of logarithmically sampled values at Nx * Ny grid.
    - X, Y: Arrays of logarithmically spaced values.
    - Xnew, Ynew: New values at which to interpolate f.

    Returns:
    - Interpolated value of exp(f) at Xnew, Ynew.
    """

    # Perform bi-logarithmic interpolation on log(f) and exponentiate the result
    return np.exp(biinterplog(np.log(f), X, Y, Xnew, Ynew))

def log_interp(x, x_vals, y_vals):
    log_y_vals = np.log(y_vals)
    interp_log_y = np.interp(np.log(x), np.log(x_vals), log_y_vals)
    return np.exp(interp_log_y)

def extend_to_array(data):
    """ Convert list of (a, beta_tab, prob_tab) into a single array of shape (Npoints, 3). """
    extended_data = []
    for a, beta_tab, prob_tab in data:
        for beta, prob in zip(beta_tab, prob_tab):
            extended_data.append([a, beta, prob])
    return np.array(extended_data)

def readcol(name, comment=None, format=None):
    """
    Read a free-format ASCII file with columns of data into numpy arrays.

    Parameters:
        name (str): Name of ASCII data file.
        comment (str, optional): Single character specifying comment character.
                                 Any line beginning with this character will be skipped.
                                 Default is None (no comment lines).
        fmt (str, optional): Scalar string containing a letter specifying an IDL type
                              for each column of data to be read.
                              Default is None (all columns assumed floating point).

    Returns:
        The numpy arrays containing columns of data.

    Raises:
        ValueError: If invalid format string is provided.

    Note:
        This function does not support all features of the IDL `readcol` function,
        such as /SILENT, /DEBUG, /NAN, /PRESERVE_NULL, /QUICK, /SKIPLINE, /NUMLINE,
        /DELIMITER, /COUNT, /NLINES, /STRINGSKIP, or /COMPRESS keywords.
    """

    # Open the file and read lines
    with open(name, 'r') as file:
        lines = file.readlines()

    # Initialize variables
    data = [[] for _ in range(len(lines[0].split()))]  # List to store data columns

    fmt = format.split(', ') if format else None

    # Process each line
    for line in lines:
        # Skip comment lines
        if comment and line.strip().startswith(comment):
            continue

        # Split the line into fields based on whitespace
        fields = line.strip().split()[:len(fmt)]
        
        # Convert each field based on format
        for i, field in enumerate(fields):
            if fmt:
                fmt_type = fmt[i].upper()
            else:
                fmt_type = 'F'  # Default to floating point if format not specified

            if fmt_type == 'A':
                data[i].append(field)
            elif fmt_type in ('D', 'F', 'I', 'B', 'L', 'U', 'Z'):
                data[i].append(float(field))
            elif fmt_type == 'X':
                continue  # Skip this column
            else:
                raise ValueError(f"Invalid format type: {fmt_type}")

    # Convert lists to numpy arrays
    arrays = [np.array(column) for column in data if len(column) > 0]

    return arrays

# Example usage:
# v1, v2, v3 = readcol("data.txt", fmt="F,F,A")



class ParallelBase:
    """Base class for parallelized function evaluation.
    
    Features:
    - Automatic thread pool management
    - Context manager support for resource cleanup
    - Batch evaluation of multiple points
    - Graceful error handling
    
    Parameters
    ----------
    max_workers : int, optional
        Maximum parallel threads; None uses the default setup in ThreadPoolExecutor (default: None)
    """
    
    def __init__(self, max_workers=None):
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        
    def __del__(self):
        """Safely shutdown thread pool when instance is destroyed."""
        if self.pool._threads:
            self.pool.shutdown(wait=True)

    def __enter__(self):
        """Enable context manager usage ('with') for resource control."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Automatically clean up resources when exiting context."""
        self.pool.shutdown(wait=True)
        return False

    def reset_pool(self, max_workers=None):
        """Reinitialize thread pool with new worker count.
        
        Args:
            max_workers : int, optional
                New maximum parallel threads (default: auto-scale)
        """
        self.pool.shutdown(wait=True)
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def _parallel_execute(self, func, points):
        """Execute function in parallel over points.
        
        Args:
            func: Function to execute
            points: Iterable of points to process
            
        Returns:
            List of results
            
        Raises:
            RuntimeError: If execution fails
        """
        try:
            futures = self.pool.map(func, points)
            return list(futures)
        except Exception as e:
            self.pool.shutdown(wait=False)
            raise RuntimeError(f"Parallel execution failed: {str(e)}") from e


class ParallelInterpolator(ParallelBase):
    """Parallelized interpolator for regular grid data.
    
    Parameters
    ----------
    data_grid : ndarray
        N-dimensional array of data values
    grid_points : tuple of arrays
        Tuple specifying grid coordinates for each dimension
    max_workers : int, optional
        Maximum parallel threads (default: None)
    
    Attributes
    ----------
    interp : RegularGridInterpolator
        Underlying SciPy interpolator instance

    Example
    -------
    >>> import numpy as np
    >>> grid = np.linspace(0, 1, 10)
    >>> data = np.sin(grid)
    >>> interpolator = ParallelInterpolator(data, (grid,))
    >>> interpolator([0.2, 0.5, 0.8])
    >>> del interp
    >>> # or
    >>> with ParallelInterpolator(data, (grid,), max_workers=30) as interpolator:
    >>>     points = np.array([[0.2], [0.5], [0.8]])  # Must be 2D array for 1D grid
    >>>     results = interpolator(points)
    >>>     print("Interpolated values:", results)
    """
    
    def __init__(self, data_grid, grid_points, max_workers=None):
        super().__init__(max_workers)
        self.interp = RegularGridInterpolator(grid_points, data_grid)
        
    def __call__(self, points):
        """Evaluate interpolator at multiple points in parallel.
        
        Args:
            points : array_like
                (N, D) array of N points with D dimensions. For example, 1D grid: [[x1], [x2], ...].
                
        Returns:
            ndarray: Interpolated values at input points
        """
        return self._parallel_execute(self.interp, points)


class Interpolator1D:
    def __init__(self, interp, logx=True, logy=True):
        '''
        Args:
            interp: 1D interpolator
            logx: whether the input (x) of interp is log scale
            logy: whether the output (y) of interp is log scale

        Returns:
            Interpolator1D function instance,
            whose input and output are both in ordinary scale.
        '''
        self.interp = interp
        self.logx = logx
        self.logy = logy
        # if logx and logy:
        #     self.__call__ = self.loglog
        # elif logx:
        #     self.__call__ = self.logx_uniformy
        # elif logy:
        #     self.__call__ = self.uniformx_logy
        # else:
        #     self.__call__ = self.uniformx_uniformy
        
    def loglog(self, xs):
        log_xs = np.log(xs)
        log_ys = self.interp(log_xs)
        return np.exp(log_ys)

    def logx_uniformy(self, xs):
        log_xs = np.log(xs)
        ys = self.interp(log_xs)
        return ys

    def uniformx_logy(self, xs):
        log_ys = self.interp(xs)
        return np.exp(log_ys)

    def uniformx_uniformy(self, xs):
        ys = self.interp(xs)
        return ys

    def __call__(self, xs):
        if self.logx and self.logy:
            return self.loglog(xs)
        elif self.logx:
            return self.logx_uniformy(xs)
        elif self.logy:
            return self.uniformx_logy(xs)
        else:
            return self.uniformx_uniformy(xs)


def loglog_interp_func_1d(log_x_grid, log_y_grid, kind='cubic'):
    '''
    Args:
        log_x_grid: 1D array, the log transformed coordinate grid feeded to the interpolator
        log_y_grid: 1D array, the log transformed function values
        kind: str, the interpolation method, default is 'cubic'
    Returns:
        Interpolator1D function instance, whose input and output are both in ordinary scale.
    '''
            
    # Verify monotonicity after transformations
    if not np.all(np.diff(log_x_grid) > 0):
        raise ValueError("Grid points must be monotonically increasing in log space")
        
    log_interp = interp1d(log_x_grid, log_y_grid, kind=kind, fill_value='extrapolate')
    
    return Interpolator1D(log_interp, logx=True, logy=True)

def logx_interp_func_1d(log_x_grid, y_grid, kind='cubic'):
    '''
    Args:
        log_x_grid: 1D array, the log transformed coordinate grid feeded to the interpolator
        log_y_grid: 1D array, the log transformed function values
        kind: str, the interpolation method, default is 'cubic'
    Returns:
        Interpolator1D function instance, whose input and output are both in ordinary scale.
    '''
            
    # Verify monotonicity after transformations
    if not np.all(np.diff(log_x_grid) > 0):
        raise ValueError("Grid points must be monotonically increasing in log space")
        
    logx_interp = interp1d(log_x_grid, y_grid, kind=kind, fill_value='extrapolate')
    
    return Interpolator1D(logx_interp, logx=True, logy=False)

def interp_func_1d(coord_grid, data_grid, kind='cubic', log_coord=True, log_data=True):
    '''
    Args:
        
        kind: str, the interpolation method, default is 'cubic'
    Returns:
        Interpolator1D function instance, whose input and output are both in ordinary scale.
    '''
            
    # Verify monotonicity after transformations
    if not np.all(np.diff(coord_grid) > 0):
        raise ValueError("Grid points must be monotonically increasing in log space")
        
    myinterp = interp1d(coord_grid, data_grid, kind=kind, fill_value='extrapolate')
    
    return Interpolator1D(myinterp, logx=log_coord, logy=log_data)


def homogeneous_dist(*args, **kwargs):
    """Constant distribution function that always returns 1"""
    return 1.0

