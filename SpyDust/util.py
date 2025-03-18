import numpy as np
#from numba import njit
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator

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


