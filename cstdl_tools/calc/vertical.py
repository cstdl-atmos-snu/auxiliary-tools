import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

from .. import constants

g = constants.g


def column_integral(data_array):
    """
    Compute the vertical column integral of an xarray DataArray using the midpoint method.

    Parameters:
        data_array (xarray.DataArray): The input data array which must have a vertical coordinate
                                       named "level" (in hPa).

    Returns:
        xarray.DataArray: The column-integrated value.
    """

    # Calculate pressure differences between levels in Pa (1 hPa = 100 Pa)
    dp = np.abs(data_array["level"].diff(dim="level") * 100)

    # Compute the difference of the data values along the vertical dimension
    dvalue = data_array.diff(dim="level")

    # Estimate the midpoint values using the adjacent differences
    mid_value = data_array - 0.5 * dvalue

    # Compute the column integral by summing over the 'level' dimension
    integral = (mid_value * dp).sum(dim="level") / g

    return integral


def omega(divergence):
    """
    Compute omega from divergence.

    Parameters:
        divergence (xarray.DataArray): The input divergence array which must have a vertical coordinate
                                       named "level" (in hPa).

    Returns:
        xarray.DataArray: The computed omega values.
    """
    div_sorted = divergence.sortby("level", ascending=True)

    # Calculate pressure differences between levels in Pa (1 hPa = 100 Pa)
    dp = np.abs(div_sorted["level"].diff(dim="level") * 100)

    # Compute the difference of the data values along the vertical dimension
    dvalue = div_sorted.diff(dim="level")

    # Estimate the midpoint values using the adjacent differences
    mid_value = div_sorted - 0.5 * dvalue

    # Compute the column integral by summing over the 'level' dimension
    omega = (-mid_value * dp).cumsum(dim="level")

    return omega


def interpolate_to_height(z_prof, var_prof, z_target):
    """
    Interpolate a vertical profile of a variable to a specific height using linear interpolation/extrapolation.

    Parameters:
        z_prof (array-like): The vertical profile of heights (in meters) or geopotential (in m^2/s^2).
        var_prof (array-like): The vertical profile of the variable to be interpolated.
        z_target (float): The target height (in meters) or geopotential (in m^2/s^2) at which to interpolate the variable
                          i.e., z = 0.0 for mean sea level; z = zs + 2 m =  zs + g * 2 for 2 m above the surface.

    Usage example:
    ```python
    # interpolate surface pressure (in Pa)
    sp = xr.apply_ufunc(
        interpolate_to_height,
        ds["geopotential"], # geopotential profile (in m^2/s^2)
        ds["level"] * 100,  # in Pa
        zs,
        input_core_dims=[["level"], ["level"], []],
        output_core_dims=[[]],
        vectorize=True,
        output_dtypes=[float],
    )
    ```
    """
    f = interp1d(z_prof, var_prof, fill_value="extrapolate")
    return f(z_target)


def get_mslp(z_prof, p_prof):
    """
    Mean sea level pressure
    """
    return interpolate_to_height(z_prof, p_prof, 0.0)


def get_sp(z_prof, p_prof, zs):
    """
    Surface pressure
    """
    return interpolate_to_height(z_prof, p_prof, zs)


def get_2m_var(z_prof, var_prof, zs):
    """
    Interpolate a vertical profile of a variable to 2 m above the surface using linear interpolation/extrapolation.

    Parameters:
        z_prof (array-like): The vertical profile of geopotential (in m^2/s^2).
        var_prof (array-like): The vertical profile of the variable to be interpolated.
        zs (float): The surface geopotential (in m^2/s^2).
    """
    return interpolate_to_height(z_prof, var_prof, zs + g * 2)
