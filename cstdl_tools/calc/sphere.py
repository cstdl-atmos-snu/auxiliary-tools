import numpy as np
import xarray as xr
import scipy.fft as fft

from .. import constants

Re = constants.Re
omega = constants.omega


def haversine_distance(point1, point2):
    """
    Calculate the distance between point1 and point2 using the Haversine formula.

    Parameters:
    - point1, point2: Tuples or arrays representing (longitude, latitude).

    The function handles the reshaping of points and conversion from degrees to radians.
    """

    lon1_rad, lat1_rad, lon2_rad, lat2_rad = np.radians(
        [point1[0], point1[1], point2[0], point2[1]]
    )

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a + 1e-10))

    # Calculate the distance
    distance = Re * c

    return distance


def dx_to_dlon(dx, lat):
    return dx * 360 / (2 * np.pi * Re * np.cos(np.deg2rad(lat)))


def dy_to_dlat(dy):
    return dy * 360 / (2 * np.pi * Re)


# Function to calculate a point given latitude, longitude, bearing, and distance
def calculate_point(lon, lat, bearing, distance):
    lon, lat, bearing = np.radians([lon, lat, bearing])
    new_lat = np.arcsin(
        np.sin(lat) * np.cos(distance / Re)
        + np.cos(lat) * np.sin(distance / Re) * np.cos(bearing)
    )
    new_lon = lon + np.arctan2(
        np.sin(bearing) * np.sin(distance / Re) * np.cos(lat),
        np.cos(distance / Re) - np.sin(lat) * np.sin(new_lat),
    )
    return np.degrees(new_lat), np.degrees(new_lon)


# Function to generate concentric circles
def concentric_circles(center_lon, center_lat, distances, bearing):
    circles = {}
    for distance in distances:
        circle_points = []
        for i_b in bearing:
            point = calculate_point(center_lon, center_lat, i_b, distance)
            circle_points.append(point)
        circles[distance] = circle_points
    return circles


# calculate the area in km^2 of a circle given a radius on the Earth in km
# https://en.wikipedia.org/wiki/Spherical_cap
def shperical_cap_area(radius):
    return 2 * np.pi * Re * Re * (1 - np.cos(radius / Re))


def rolling_sum_np(np_array, win_size=(41, 41), doLonPad=False):
    """
    np_array = (lat, lon)
    """
    # rolling mean 10 by 10 degree (41 by 41 grid, including center)
    lat_win_size = win_size[0]
    lon_win_size = win_size[1]

    if doLonPad:
        np_array_copy = np.zeros(
            (np_array.shape[0], np_array.shape[1] + lon_win_size - 1)
        )
        np_array_copy[:, : lon_win_size // 2] = np_array[:, -(lon_win_size // 2) :]
        np_array_copy[:, lon_win_size // 2 : -(lon_win_size // 2)] = np_array
        np_array_copy[:, -(lon_win_size // 2) :] = np_array[:, : lon_win_size // 2]
    else:
        np_array_copy = np_array.copy()

    data_bar = np.zeros(
        (
            lon_win_size * lat_win_size,
            np_array_copy.shape[0] - lat_win_size + 1,
            np_array_copy.shape[1] - lon_win_size + 1,
        )
    )

    for j in range(lat_win_size):
        for k in range(lon_win_size):
            data_bar[j * lon_win_size + k, :, :] = np_array_copy[
                j : np_array_copy.shape[0] - lat_win_size + j + 1,
                k : np_array_copy.shape[1] - lon_win_size + k + 1,
            ]

    data_bar = np.nansum(data_bar, axis=0)
    return data_bar


def dx_central(data_array, doLonPad=False):
    """
    Compute the zonal derivative (with respect to longitude) of an xarray DataArray using a central difference scheme.

    The derivative is calculated along the "longitude" dimension. The grid spacing (dx) is determined from the
    difference between adjacent longitude values (assumed to be uniform) and is adjusted for latitude by
    converting degrees to a physical distance using Earth's mean radius.

    For boundary points:
      - If do_lon_pad is True, periodic boundary conditions are applied using a roll operation.
      - Otherwise, forward and backward differences are used at the boundaries with appropriate scaling.

    Parameters:
        data (xarray.DataArray): The input data array with "longitude" and "latitude" coordinates.
        do_lon_pad (bool): If True, apply periodic boundary conditions in the longitude direction.
                           If False, use forward/backward differences at the boundaries. Defaults to False.

    Returns:
        xarray.DataArray: The computed zonal derivative (df/dx) of the input data.
    """

    # Compute the difference in longitude (assumes uniform spacing)
    dlon = data_array["longitude"][1] - data_array["longitude"][0]
    dx = np.deg2rad(dlon) * np.cos(np.deg2rad(data_array["latitude"])) * Re

    if doLonPad:
        # Apply periodic boundary conditions using roll
        dvalue = data_array.roll(longitude=-1) - data_array.roll(longitude=1)
    else:
        dvalue = data_array.shift(longitude=-1, fill_value=np.nan) - data_array.shift(
            longitude=1, fill_value=np.nan
        )
        dvalue[dict(longitude=0)] = (
            data_array.isel(longitude=1).values - data_array.isel(longitude=0).values
        ) * 2
        dvalue[dict(longitude=-1)] = (
            data_array.isel(longitude=-1).values - data_array.isel(longitude=-2).values
        ) * 2

    # (f(x+dx) - f(x-dx))/(2*dx)
    derivative = dvalue / (2 * dx)

    return derivative


def dy_central(data_array):
    """
    Compute the meridional derivative (with respect to latitude) of an xarray DataArray using a central difference scheme.

    The derivative is calculated along the "latitude" dimension. The grid spacing (dy) is determined from the
    difference between adjacent latitude values (assumed to be uniform) and is adjusted for latitude by
    converting degrees to a physical distance using Earth's mean radius.

    For boundary points:
      - Forward and backward differences are used at the boundaries with appropriate scaling.

    Parameters:
        data (xarray.DataArray): The input data array with "latitude" coordinates.

    Returns:
        xarray.DataArray: The computed zonal derivative (df/dy) of the input data.
    """

    # Compute the difference in longitude (assumes uniform spacing)
    dlat = data_array["latitude"][1] - data_array["latitude"][0]
    dy = np.deg2rad(dlat) * Re

    dvalue = data_array.shift(latitude=-1, fill_value=np.nan) - data_array.shift(
        latitude=1, fill_value=np.nan
    )
    dvalue[dict(latitude=0)] = (
        data_array.isel(latitude=1).values - data_array.isel(latitude=0).values
    ) * 2
    dvalue[dict(latitude=-1)] = (
        data_array.isel(latitude=-1).values - data_array.isel(latitude=-2).values
    ) * 2

    # (f(y+dy) - f(y-dy))/(2*dy)
    derivative = dvalue / (2 * dy)

    return derivative


def f(latitude):
    """
    Compute the Coriolis parameter (f) at a given latitude.
    f = 2 * omega * sin(latitude)
    parameters:
        latitude (xarray.DataArray): Latitude values (in degrees)
    """
    f = 2 * omega * np.sin(np.deg2rad(latitude))
    return f
