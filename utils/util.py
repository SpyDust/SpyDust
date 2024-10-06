import numpy as np
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from numba import jit, njit

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

@njit
def maketab(xmin, xmax, Npt):
    """Creates a linearly spaced table."""
    return xmin + (xmax - xmin) / Npt * (0.5 + np.arange(Npt))

@njit
def makelogtab(xmin, xmax, Npt):
    """Creates a logarithmically spaced table."""
    return xmin * np.exp(1 / Npt * np.log(xmax / xmin) * (0.5 + np.arange(Npt)))

@njit
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
    Interpolates f given at X, Y (logarithmically spaced) at Xnew, Ynew.

    Parameters:
    - f: 2D array of values at Nx * Ny grid.
    - X, Y: Arrays of logarithmically spaced values.
    - Xnew, Ynew: New values at which to interpolate f.

    Returns:
    - Interpolated value of f at Xnew, Ynew.
    """

    # Generate the grids
    grid_log_y, grid_log_x = np.meshgrid(np.log(np.array(Y)), np.log(np.array(X)))

    # Flatten the grids
    flat_x = grid_log_x.flatten()
    flat_y = grid_log_y.flatten()

    # Stack them and transpose to get the shape (num, 2)
    points = np.vstack((flat_x, flat_y)).T

    grid_log_y_new, grix_log_x_new = np.meshgrid(np.log(np.array(Ynew)), np.log(np.array(Xnew)))

    # Interpolation requires coordinates as pairs
    return griddata(points, f.flatten(), (grix_log_x_new, grid_log_y_new), method='linear')

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


